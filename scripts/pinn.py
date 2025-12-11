"""
Physics-Informed Neural Network (PINN) for GIC prediction.

This module contains the implementation of the PINN model, including:
- Custom PyTorch layers (ImpedanceLayer)
- GICPINN model definition
- Training and evaluation functions
"""

import os
import time
from math import ceil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from tqdm.auto import tqdm

from config.settings import setup_logger

logger = setup_logger(name="tfgic.pinn")


class WindowDS(Dataset):
    """Dataset for creating sliding windows of input data."""

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
            sy = sy if sy > 0 else 1.0
            y_stats = (my, sy)
        self.stats, self.y_stats = stats, y_stats

    def __len__(self):
        return len(self.X) - self.win

    def __getitem__(self, i):
        sl = slice(i, i + self.win)
        x = ((self.X[sl] - self.stats[0]) / self.stats[1]).T
        e = self.E[sl].T
        y = (self.y[i + self.win - 1] - self.y_stats[0]) / self.y_stats[1]
        return x, e, y


class ImpedanceLayer(nn.Module):
    """Custom layer for impedance calculations."""

    def __init__(self, Z0):
        super().__init__()
        self.register_buffer("Z0", torch.tensor(Z0, dtype=torch.cfloat))
        self.dZ = nn.Parameter(torch.zeros_like(self.Z0))

    def forward(self, Bx_f, By_f):
        Z = self.Z0 + self.dZ
        Ex_f = Z[0] * Bx_f + Z[1] * By_f
        Ey_f = Z[2] * Bx_f + Z[3] * By_f
        return Ex_f, Ey_f


class GICPINN(nn.Module):
    """Physics-informed neural network for GIC prediction."""

    def __init__(self, Z0, n_other=15, hidden=64):
        super().__init__()
        self.imp = ImpedanceLayer(Z0)

        gru_input_size = 2 + 2 + n_other - 2
        self.gru = nn.GRU(
            input_size=gru_input_size,
            hidden_size=hidden,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=2 * hidden, num_heads=4, batch_first=True, dropout=0.1
        )

        self.fc1 = nn.Linear(2 * hidden, hidden)
        self.fc2 = nn.Linear(hidden, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()

        self.residual = nn.Linear(2 * hidden, 1)

    def forward(self, x):
        Bx, By = x[:, 0], x[:, 1]
        other = x[:, 2:]

        Bx_f = torch.fft.rfft(Bx, dim=-1)[:, 1:]
        By_f = torch.fft.rfft(By, dim=-1)[:, 1:]
        Ex_f, Ey_f = self.imp(Bx_f, By_f)

        pad = lambda f: torch.cat([torch.zeros_like(f[..., :1]), f], -1)
        Ex = torch.fft.irfft(pad(Ex_f), n=Bx.size(1), dim=-1).real
        Ey = torch.fft.irfft(pad(Ey_f), n=By.size(1), dim=-1).real

        seq = torch.cat(
            [Bx.unsqueeze(1), By.unsqueeze(1), Ex.unsqueeze(1), Ey.unsqueeze(1), other],
            1,
        ).permute(0, 2, 1)

        gru_out, h = self.gru(seq)

        attended_out, attn_weights = self.attention(gru_out, gru_out, gru_out)

        final_state = torch.cat([h[-2], h[-1]], 1)
        attended_final = (attn_weights.mean(1).unsqueeze(-1) * attended_out).sum(1)

        combined = final_state + attended_final

        out = self.relu(self.fc1(combined))
        out = self.dropout(out)
        out = self.relu(self.fc2(out))
        out = self.dropout(out)
        main_out = self.fc3(out)

        residual_out = self.residual(final_state)

        gic = (main_out + residual_out).squeeze(-1)

        return gic, (Ex, Ey)


def assert_finite(*tensors, where=""):
    """Check if tensors contain only finite values."""
    for t in tensors:
        if not torch.isfinite(t).all():
            raise RuntimeError(f"Non-finite detected {where}, shape={t.shape}")


def dump_bad(t, name, yard, step):
    """Check for non-finite values in tensors during training."""
    bad = ~torch.isfinite(t)
    if bad.any():
        idx = bad.nonzero(as_tuple=False)
        logger.warning(
            f"{yard} • first non-finite in {name} @ minibatch step {step} "
            f"tensor-idx={idx[0].tolist()}  value={t.flatten()[idx[0,0]].item():.3g}"
        )
        return True
    return False


def save_best(state_dict, yard):
    """Save best model for a specific yard."""
    out = Path("models")
    out.mkdir(exist_ok=True)
    f = out / f"GICPINN_{yard.replace(' ','_')}_best.pt"
    torch.save(state_dict, f)
    logger.info(f"✓ saved → {f}")


def train_pinn(
    model, dl_tr, dl_val, epochs=40, lr=3e-4, yard="", β_E=0.2, λ_Z=1e-3, device=None
):
    """Train the PINN model."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.7)

    best, patience, best_state = 1e9, 8, None

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for step, (xb, eref, yb) in enumerate(
            tqdm(dl_tr, desc=f"train {yard} ep{ep}", leave=False)
        ):
            xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)
            if (
                dump_bad(xb, "X", yard, step)
                or dump_bad(eref, "Eref", yard, step)
                or dump_bad(yb, "y", yard, step)
            ):
                break

            assert_finite(xb, yb, where="input")
            pred, (Ex, Ey) = model(xb)
            assert_finite(pred, Ex, Ey, where="output")

            L_gic = F.mse_loss(pred, yb)
            L_E = F.mse_loss(torch.stack([Ex, Ey], 1), eref)
            L_Z = torch.mean(model.imp.dZ.abs() ** 2)

            L_smooth = (
                torch.mean(torch.abs(model.imp.dZ[1:] - model.imp.dZ[:-1]))
                if model.imp.dZ.shape[1] > 1
                else 0
            )

            L_robust = F.huber_loss(pred, yb, delta=1.0)

            loss = (
                0.7 * L_gic + 0.3 * L_robust + β_E * L_E + λ_Z * L_Z + 1e-4 * L_smooth
            )

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for xb, eref, yb in dl_val:
                xb, eref, yb = xb.to(device), eref.to(device), yb.to(device)
                p, (Ex, Ey) = model(xb)
                val_loss += (
                    F.mse_loss(p, yb) + β_E * F.mse_loss(torch.stack([Ex, Ey], 1), eref)
                ).item()

        val_loss /= len(dl_val)
        scheduler.step(val_loss)

        logger.info(
            f"{yard} ep{ep:02d} | train {train_loss/len(dl_tr):.5f} | val {val_loss:.5f} | lr {opt.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best - 1e-5:
            best, patience = val_loss, 8
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            save_best(best_state, yard)
        else:
            patience -= 1

        if patience == 0:
            logger.info(f"Early stopping at epoch {ep}")
            break

    model.load_state_dict(best_state)
    return model


def evaluate_model(model, ds_te, batch_size=1024, device=None):
    """Evaluate model on test dataset."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)

    dl_te = DataLoader(ds_te, batch_size=min(batch_size, len(ds_te)), shuffle=False)

    model.eval()
    preds = []
    obs = []
    with torch.no_grad():
        for xb, _, yb in dl_te:
            p, _ = model(xb.to(device))
            preds.append(p.cpu())
            obs.append(yb)

    preds = torch.cat(preds).numpy()
    obs = torch.cat(obs).numpy()

    mse = np.mean((obs - preds) ** 2)
    rmse = np.sqrt(mse)
    var_obs = np.var(obs)
    pe = 1 - mse / var_obs if var_obs > 0 else float("nan")
    mae = np.mean(np.abs(obs - preds))

    corr = np.corrcoef(obs, preds)[0, 1]

    return {
        "predictions": preds,
        "observations": obs,
        "PE": pe,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "Correlation": corr,
    }


def leave_one_yard_out(
    yards, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=None
):
    """Perform leave-one-yard-out cross-validation."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    winF = ceil(win / 2)
    dummy_Z = np.zeros((4, winF), dtype=np.complex64)
    results = {}

    for test_yard in yards:
        logger.info(f"▶ leave-out: {test_yard}")
        train_yards = [y for y in yards if y != test_yard]

        ds_train = []
        for y in train_yards:
            X, y_gic, E = build_feature_tensor_fn(y)
            ds_train.append(WindowDS(X, y_gic, E, win))
            logger.debug(f"{y}  windows:{len(ds_train[-1]):,}")

        X_te, y_te, E_te = build_feature_tensor_fn(test_yard)
        ds_te = WindowDS(
            X_te, y_te, E_te, win, stats=ds_train[0].stats, y_stats=ds_train[0].y_stats
        )

        full = ConcatDataset(ds_train)
        n_val = max(1, int(0.1 * len(full)))
        ds_val, ds_tr = random_split(full, [n_val, len(full) - n_val])

        dl_tr = DataLoader(
            ds_tr, batch_size=min(batch, len(ds_tr)), shuffle=True, drop_last=True
        )
        dl_val = DataLoader(ds_val, batch_size=min(batch, len(ds_val)), shuffle=False)

        model = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
        train_pinn(
            model, dl_tr, dl_val, epochs=epochs, lr=3e-4, yard=test_yard, device=device
        )

        eval_results = evaluate_model(model, ds_te, batch_size=batch, device=device)
        results[test_yard] = eval_results
        pe = eval_results["PE"]
        logger.info(f"★ {test_yard:<15s} PE = {pe:.3f}")

    logger.info("\n==== summary ====")
    for y, res in results.items():
        logger.info(f"{y:<15s} {res['PE']:.3f}")

    return results


def train_on_all_yards(
    yards, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=None
):
    """Train on all given yards (no leave-one-out)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    winF = ceil(win / 2)
    dummy_Z = np.zeros((4, winF), dtype=np.complex64)

    logger.info(f"▶ Training on all yards: {yards}")

    ds_train = []
    for y in yards:
        X, y_gic, E = build_feature_tensor_fn(y)
        ds_train.append(WindowDS(X, y_gic, E, win))
        logger.debug(f"{y}  windows:{len(ds_train[-1]):,}")

    full = ConcatDataset(ds_train)
    n_val = max(1, int(0.1 * len(full)))
    ds_val, ds_tr = random_split(full, [n_val, len(full) - n_val])

    dl_tr = DataLoader(
        ds_tr, batch_size=min(batch, len(ds_tr)), shuffle=True, drop_last=True
    )
    dl_val = DataLoader(ds_val, batch_size=min(batch, len(ds_val)), shuffle=False)

    model = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
    trained_model = train_pinn(
        model, dl_tr, dl_val, epochs=epochs, lr=3e-4, yard="all_yards", device=device
    )

    return trained_model


def train_temporal_split(
    yard, build_feature_tensor_fn, win=256, batch=1024, epochs=40, device=None
):
    """Train on Days 1-2, test on Day 3 for single yard"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_full, y_full, E_full = build_feature_tensor_fn(yard)

    samples_per_day = len(y_full) // 3

    X_train = X_full[: 2 * samples_per_day]
    y_train = y_full[: 2 * samples_per_day]
    E_train = E_full[: 2 * samples_per_day]

    X_test = X_full[2 * samples_per_day :]
    y_test = y_full[2 * samples_per_day :]
    E_test = E_full[2 * samples_per_day :]

    train_ds = WindowDS(X_train, y_train, E_train, win)
    test_ds = WindowDS(
        X_test, y_test, E_test, win, stats=train_ds.stats, y_stats=train_ds.y_stats
    )

    n_val = max(1, int(0.1 * len(train_ds)))
    ds_val, ds_tr = random_split(train_ds, [n_val, len(train_ds) - n_val])

    dl_tr = DataLoader(
        ds_tr, batch_size=min(batch, len(ds_tr)), shuffle=True, drop_last=True
    )
    dl_val = DataLoader(ds_val, batch_size=min(batch, len(ds_val)), shuffle=False)

    winF = ceil(win / 2)
    dummy_Z = np.zeros((4, winF), dtype=np.complex64)
    model = GICPINN(dummy_Z, n_other=15, hidden=64).to(device)
    trained_model = train_pinn(
        model,
        dl_tr,
        dl_val,
        epochs=epochs,
        lr=3e-4,
        yard=f"{yard}_temporal",
        device=device,
    )

    results = evaluate_model(trained_model, test_ds, device=device)
    results["yard"] = yard
    return results
