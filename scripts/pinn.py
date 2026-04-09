"""
Author: Dennies Bor
Role:   Physics-informed neural network (PINN) for GIC prediction.
Inputs: Sliding-window magnetometer and electric field tensors.
"""

from math import ceil

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm.auto import tqdm

from config.settings import setup_logger, MODELS_DIR, DEVICE

logger = setup_logger(name="tfgic.pinn")


class WindowDS(Dataset):
    """Sliding-window dataset with robust (MAD) normalisation."""

    def __init__(self, X, y, E, win=256, stats=None, y_stats=None):
        self.win = win
        L = min(len(X), len(y), len(E))
        self.X = torch.tensor(X[:L], dtype=torch.float32)
        self.y = torch.tensor(y[:L], dtype=torch.float32)
        self.E = torch.tensor(E[:L], dtype=torch.float32)

        if stats is None:
            med = torch.median(self.X, 0).values
            mad = torch.median(torch.abs(self.X - med), 0).values
            mad[mad == 0] = 1.0
            stats = (med, 1.4826 * mad)
        if y_stats is None:
            my = torch.median(self.y)
            sy = 1.4826 * torch.median(torch.abs(self.y - my))
            y_stats = (my, sy if sy > 0 else 1.0)

        self.stats, self.y_stats = stats, y_stats

    def __len__(self):
        return len(self.X) - self.win

    def __getitem__(self, i):
        sl = slice(i, i + self.win)
        x  = ((self.X[sl] - self.stats[0]) / self.stats[1]).T
        e  = self.E[sl].T
        y  = (self.y[i + self.win - 1] - self.y_stats[0]) / self.y_stats[1]
        return x, e, y


class ImpedanceLayer(nn.Module):
    """Learnable perturbation around a fixed impedance tensor Z0."""

    def __init__(self, Z0):
        super().__init__()
        self.register_buffer("Z0", torch.tensor(Z0, dtype=torch.cfloat))
        self.dZ = nn.Parameter(torch.zeros_like(self.Z0))

    def forward(self, Bx_f, By_f):
        Z    = self.Z0 + self.dZ
        Ex_f = Z[0] * Bx_f + Z[1] * By_f
        Ey_f = Z[2] * Bx_f + Z[3] * By_f
        return Ex_f, Ey_f


class GICPINN(nn.Module):
    """Physics-informed GRU network with impedance-derived electric field input."""

    def __init__(self, Z0, n_other=15, hidden=64):
        super().__init__()
        self.imp = ImpedanceLayer(Z0)

        self.gru = nn.GRU(
            input_size=2 + 2 + n_other - 2,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=2 * hidden, num_heads=4, batch_first=True, dropout=0.1
        )

        self.fc1      = nn.Linear(2 * hidden, hidden)
        self.fc2      = nn.Linear(hidden, 32)
        self.fc3      = nn.Linear(32, 1)
        self.residual = nn.Linear(2 * hidden, 1)
        self.dropout  = nn.Dropout(0.1)
        self.relu     = nn.ReLU()

    def forward(self, x):
        Bx, By = x[:, 0], x[:, 1]
        other  = x[:, 2:]

        # Compute E-field via impedance in the frequency domain
        Bx_f = torch.fft.rfft(Bx, dim=-1)[:, 1:]
        By_f = torch.fft.rfft(By, dim=-1)[:, 1:]
        Ex_f, Ey_f = self.imp(Bx_f, By_f)

        pad = lambda f: torch.cat([torch.zeros_like(f[..., :1]), f], -1)
        Ex  = torch.fft.irfft(pad(Ex_f), n=Bx.size(1), dim=-1).real
        Ey  = torch.fft.irfft(pad(Ey_f), n=By.size(1), dim=-1).real

        seq = torch.cat(
            [Bx.unsqueeze(1), By.unsqueeze(1), Ex.unsqueeze(1), Ey.unsqueeze(1), other], 1
        ).permute(0, 2, 1)

        gru_out, h     = self.gru(seq)
        attended, w    = self.attention(gru_out, gru_out, gru_out)
        final_state    = torch.cat([h[-2], h[-1]], 1)
        attended_final = (w.mean(1).unsqueeze(-1) * attended).sum(1)
        combined       = final_state + attended_final

        out = self.dropout(self.relu(self.fc1(combined)))
        out = self.dropout(self.relu(self.fc2(out)))
        gic = (self.fc3(out) + self.residual(final_state)).squeeze(-1)

        return gic, (Ex, Ey)


def assert_finite(*tensors, where=""):
    """Raise if any tensor contains non-finite values."""
    for t in tensors:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite detected {where}, shape={t.shape}")


def dump_bad(t, name, yard, step):
    """Log the first non-finite value in a tensor; return True if found."""
    bad = ~torch.isfinite(t)
    if bad.any():
        idx = bad.nonzero(as_tuple=False)
        logger.warning(
            f"{yard}: non-finite in {name} at step {step}, "
            f"idx={idx[0].tolist()}, val={t.flatten()[idx[0, 0]].item():.3g}"
        )
        return True
    return False


def save_best(state_dict, yard):
    """Persist best model weights to data/models/."""
    path = MODELS_DIR / f"GICPINN_{yard.replace(' ', '_')}_best.pt"
    torch.save(state_dict, path)
    logger.info(f"Saved best model to {path}")


def train_pinn(model, dl_tr, dl_val, epochs=40, lr=3e-4, yard="", β_E=0.2, λ_Z=1e-3, device=DEVICE):
    """Train GICPINN with composite loss and early stopping."""
    model     = model.to(device)
    opt       = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.7)

    best, patience, best_state = 1e9, 8, None

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for step, (xb, eref, yb) in enumerate(tqdm(dl_tr, desc=f"train {yard} ep{ep}", leave=False)):
            xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)

            if dump_bad(xb, "X", yard, step) or dump_bad(eref, "Eref", yard, step) or dump_bad(yb, "y", yard, step):
                break

            assert_finite(xb, yb, where="input")
            pred, (Ex, Ey) = model(xb)
            assert_finite(pred, Ex, Ey, where="output")

            L_gic    = F.mse_loss(pred, yb)
            L_E      = F.mse_loss(torch.stack([Ex, Ey], 1), eref)
            L_Z      = torch.mean(model.imp.dZ.abs() ** 2)
            L_smooth = (
                torch.mean(torch.abs(model.imp.dZ[1:] - model.imp.dZ[:-1]))
                if model.imp.dZ.shape[1] > 1 else 0
            )
            L_robust = F.huber_loss(pred, yb, delta=1.0)
            loss     = 0.7 * L_gic + 0.3 * L_robust + β_E * L_E + λ_Z * L_Z + 1e-4 * L_smooth

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, eref, yb in dl_val:
                xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)
                p, (Ex, Ey)  = model(xb)
                val_loss    += (F.mse_loss(p, yb) + β_E * F.mse_loss(torch.stack([Ex, Ey], 1), eref)).item()

        val_loss /= len(dl_val)
        scheduler.step(val_loss)
        logger.info(
            f"{yard} ep{ep:02d} | train {train_loss / len(dl_tr):.5f} "
            f"| val {val_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best - 1e-5:
            best, patience = val_loss, 8
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            save_best(best_state, yard)
        else:
            patience -= 1
            if patience == 0:
                logger.info(f"Early stopping at epoch {ep}.")
                break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, ds_te, batch_size=1024, device=DEVICE):
    """Evaluate model on a test dataset; return predictions and metrics."""
    model = model.to(device)
    dl_te = DataLoader(ds_te, batch_size=min(batch_size, len(ds_te)), shuffle=False)

    model.eval()
    preds, obs = [], []
    with torch.no_grad():
        for xb, _, yb in dl_te:
            p, _ = model(xb.to(device))
            preds.append(p.cpu())
            obs.append(yb)

    preds = torch.cat(preds).numpy()
    obs   = torch.cat(obs).numpy()
    mse   = np.mean((obs - preds) ** 2)
    pe    = 1 - mse / np.var(obs) if np.var(obs) > 0 else float("nan")

    return {
        "predictions": preds,
        "observations": obs,
        "PE":          pe,
        "MSE":         mse,
        "RMSE":        np.sqrt(mse),
        "MAE":         np.mean(np.abs(obs - preds)),
        "Correlation": np.corrcoef(obs, preds)[0, 1],
    }


def _build_dataloaders(ds_list, win, batch):
    """Concatenate datasets and split into train/val dataloaders."""
    full  = ConcatDataset(ds_list)
    n_val = max(1, int(0.1 * len(full)))
    ds_val, ds_tr = random_split(full, [n_val, len(full) - n_val])
    dl_tr  = DataLoader(ds_tr,  batch_size=min(batch, len(ds_tr)),  shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=min(batch, len(ds_val)), shuffle=False)
    return dl_tr, dl_val


def leave_one_yard_out(yards, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=DEVICE):
    """Leave-one-yard-out cross-validation across all yards."""
    dummy_Z = np.zeros((4, ceil(win / 2)), dtype=np.complex64)
    results = {}

    for test_yard in yards:
        logger.info(f"Leave-out: {test_yard}")
        ds_train = []
        for y in [y for y in yards if y != test_yard]:
            X, y_gic, E = build_feature_tensor_fn(y)
            ds_train.append(WindowDS(X, y_gic, E, win))
            logger.debug(f"{y}  windows: {len(ds_train[-1]):,}")

        X_te, y_te, E_te = build_feature_tensor_fn(test_yard)
        ds_te = WindowDS(X_te, y_te, E_te, win, stats=ds_train[0].stats, y_stats=ds_train[0].y_stats)

        dl_tr, dl_val = _build_dataloaders(ds_train, win, batch)
        model = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
        train_pinn(model, dl_tr, dl_val, epochs=epochs, lr=3e-4, yard=test_yard, device=device)

        res = evaluate_model(model, ds_te, batch_size=batch, device=device)
        results[test_yard] = res
        logger.info(f"{test_yard:<15s} PE = {res['PE']:.3f}")

    logger.info("Summary:")
    for y, res in results.items():
        logger.info(f"  {y:<15s} PE = {res['PE']:.3f}")

    return results


def train_on_all_yards(yards, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=DEVICE):
    """Train a single GICPINN on all yards combined."""
    logger.info(f"Training on all yards: {yards}")
    dummy_Z  = np.zeros((4, ceil(win / 2)), dtype=np.complex64)
    ds_train = []
    for y in yards:
        X, y_gic, E = build_feature_tensor_fn(y)
        ds_train.append(WindowDS(X, y_gic, E, win))
        logger.debug(f"{y}  windows: {len(ds_train[-1]):,}")

    dl_tr, dl_val = _build_dataloaders(ds_train, win, batch)
    model = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
    return train_pinn(model, dl_tr, dl_val, epochs=epochs, lr=3e-4, yard="all_yards", device=device)


def train_temporal_split(yard, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=DEVICE):
    """Train on days 1-2, evaluate on day 3 for a single yard."""
    X_full, y_full, E_full = build_feature_tensor_fn(yard)
    split    = 2 * (len(y_full) // 3)
    train_ds = WindowDS(X_full[:split], y_full[:split], E_full[:split], win)
    test_ds  = WindowDS(
        X_full[split:], y_full[split:], E_full[split:], win,
        stats=train_ds.stats, y_stats=train_ds.y_stats,
    )

    dl_tr, dl_val   = _build_dataloaders([train_ds], win, batch)
    dummy_Z          = np.zeros((4, ceil(win / 2)), dtype=np.complex64)
    model            = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
    trained          = train_pinn(model, dl_tr, dl_val, epochs=epochs, lr=3e-4, yard=f"{yard}_temporal", device=device)
    results          = evaluate_model(trained, test_ds, device=device)
    results["yard"]  = yard
    return results