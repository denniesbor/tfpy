"""
Author: Dennies Bor
Role:   CNN and GRU ensemble models for GIC prediction.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm

from config.settings import setup_logger, DEVICE

logger = setup_logger(name="tfgic.nn_models")


def feature_matrix_cnn(site, mag_data, gic_data, site_relations, start, end):
    """Build enhanced feature matrix for CNN models."""
    mag = site_relations[site]["magnetometer"]
    bx = mag_data.sel(device=mag, time=slice(start, end)).Bx.values.astype("f4")
    by = mag_data.sel(device=mag, time=slice(start, end)).By.values.astype("f4")
    g  = gic_data.gic.sel(device=site, time=slice(start, end)).values.astype("f4")

    bx -= np.median(bx)
    by -= np.median(by)

    dBx  = np.gradient(bx)
    dBy  = np.gradient(by)
    absB = np.sqrt(bx**2 + by**2)
    d2Bx = np.gradient(dBx)
    d2By = np.gradient(dBy)

    # Short and medium moving-average residuals
    bx_ma_short = np.convolve(bx, np.ones(10) / 10, mode="same")
    by_ma_short = np.convolve(by, np.ones(10) / 10, mode="same")
    bx_ma_med   = np.convolve(bx, np.ones(30) / 30, mode="same")
    by_ma_med   = np.convolve(by, np.ones(30) / 30, mode="same")

    X = np.column_stack([
        bx, by, absB, dBx, dBy, d2Bx, d2By,
        bx - bx_ma_short, by - by_ma_short,
        bx - bx_ma_med,   by - by_ma_med,
    ])
    return X, g


def feature_matrix_gru(site, mag_data, gic_data, site_relations, start, end):
    """Build feature matrix for GRU models."""
    mag = site_relations[site]["magnetometer"]
    bx = mag_data.sel(device=mag, time=slice(start, end)).Bx.values.astype("f4")
    by = mag_data.sel(device=mag, time=slice(start, end)).By.values.astype("f4")
    g  = gic_data.gic.sel(device=site, time=slice(start, end)).values.astype("f4")

    bx -= np.median(bx)
    by -= np.median(by)

    dBx  = np.gradient(bx)
    dBy  = np.gradient(by)
    absB = np.sqrt(bx**2 + by**2)

    # 30-sample rolling std via convolution
    k    = 30
    kern = np.ones(k, "f4") / k
    mu_bx = np.convolve(bx, kern, "same")
    mu_by = np.convolve(by, kern, "same")
    std30_bx = np.sqrt(np.maximum(np.convolve(bx**2, kern, "same") - mu_bx**2, 0))
    std30_by = np.sqrt(np.maximum(np.convolve(by**2, kern, "same") - mu_by**2, 0))

    X = np.column_stack([bx, by, absB, dBx, dBy, std30_bx, std30_by])
    return X, g


class WindowDataset(Dataset):
    """Sliding-window dataset with mean/std normalisation."""

    def __init__(self, X, y, win=120, mean=None, std=None):
        self.win = win
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.float32)
        if mean is None:
            mean = self.X.mean(0)
            std  = self.X.std(0)
        self.mean, self.std = mean, std

    def __len__(self):
        return self.X.shape[0] - self.win

    def __getitem__(self, idx):
        s = slice(idx, idx + self.win)
        x = ((self.X[s] - self.mean) / self.std).T
        y = self.y[idx + self.win - 1]
        return x, y


class WindowDS(Dataset):
    """Sliding-window dataset with robust (MAD) normalisation."""

    def __init__(self, X, y, win=256, stats=None, y_stats=None):
        self.win = win
        self.X   = torch.tensor(X, dtype=torch.float32)
        self.y   = torch.tensor(y, dtype=torch.float32)

        if stats is None:
            med  = torch.median(self.X, 0)[0]
            mad  = torch.median(torch.abs(self.X - med), 0)[0]
            mad[mad == 0] = 1.0
            stats = (med, 1.4826 * mad)

        if y_stats is None:
            my = torch.median(self.y)
            sy = 1.4826 * torch.median(torch.abs(self.y - my))
            y_stats = (my, sy if sy > 0 else 1.0)

        self.stats, self.y_stats = stats, y_stats

    def __len__(self):
        return self.X.shape[0] - self.win

    def __getitem__(self, i):
        sl = slice(i, i + self.win)
        x  = ((self.X[sl] - self.stats[0]) / self.stats[1]).T
        y  = (self.y[i + self.win - 1] - self.y_stats[0]) / self.y_stats[1]
        return x, y


class CNN1D(nn.Module):
    """1-D convolutional network for time-series GIC prediction."""

    def __init__(self, C):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(C, 32, 5, padding=2), nn.ReLU(),
            nn.Conv1d(32, 32, 5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class GRU1S(nn.Module):
    """Bidirectional 2-layer GRU."""

    def __init__(self, C, hidden=64):
        super().__init__()
        self.gru = nn.GRU(C, hidden, 2, batch_first=True, bidirectional=True)
        self.fc  = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(torch.cat([h[-2], h[-1]], 1)).squeeze(-1)


class GRU2S(nn.Module):
    """Deeper bidirectional GRU with dropout."""

    def __init__(self, C, hidden=64):
        super().__init__()
        self.gru = nn.GRU(C, hidden, 3, batch_first=True, bidirectional=True, dropout=0.2)
        self.fc  = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, h = self.gru(x)
        return self.fc(torch.cat([h[-2], h[-1]], 1)).squeeze(-1)


class GRUWithAttention(nn.Module):
    """Bidirectional GRU with additive attention."""

    def __init__(self, C, hidden=64):
        super().__init__()
        self.gru       = nn.GRU(C, hidden, 2, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(2 * hidden, 1)
        self.fc        = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _       = self.gru(x)
        attn_weights = F.softmax(self.attention(out), dim=1)
        context      = torch.sum(out * attn_weights, dim=1)
        return self.fc(context).squeeze(-1)


class LSTMS(nn.Module):
    """Bidirectional 2-layer LSTM."""

    def __init__(self, C, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(C, hidden, 2, batch_first=True, bidirectional=True)
        self.fc   = nn.Sequential(nn.Linear(2 * hidden, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        _, (h, _) = self.lstm(x)
        return self.fc(torch.cat([h[-2], h[-1]], 1)).squeeze(-1)


def train_model(model, dl_tr, dl_val, epochs=20, lr=3e-4, wd=1e-4, model_name="model"):
    """Train with early stopping and ReduceLROnPlateau scheduling."""
    opt       = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=3)

    best, patience, state = 1e9, 6, None

    for ep in range(epochs):
        model.train()
        for xb, yb in tqdm(dl_tr, desc=f"[{model_name}] epoch {ep + 1}", leave=False):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss   = F.mse_loss(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in dl_val:
                val_loss += F.mse_loss(model(xb.to(DEVICE)), yb.to(DEVICE)).item()
        val_loss /= len(dl_val)
        scheduler.step(val_loss)

        if val_loss < best - 1e-4:
            best, patience = val_loss, 6
            state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience == 0:
                break

    model.load_state_dict(state)
    return best


def model5a_cnn(
    site, mag_data, gic_data, site_relations,
    train_slice, test_slice, window_size=120, epochs=20,
):
    """Train and evaluate a 5-model CNN ensemble."""
    Xtr, ytr = feature_matrix_cnn(site, mag_data, gic_data, site_relations, train_slice.start, train_slice.stop)
    Xte, yte = feature_matrix_cnn(site, mag_data, gic_data, site_relations, test_slice.start,  test_slice.stop)

    ds_tr = WindowDataset(Xtr, ytr, win=window_size)
    n_val = int(0.2 * len(ds_tr))
    ds_train = Subset(ds_tr, list(range(len(ds_tr) - n_val)))
    ds_val   = Subset(ds_tr, list(range(len(ds_tr) - n_val, len(ds_tr))))
    ds_te    = WindowDataset(Xte, yte, win=window_size, mean=ds_tr.mean, std=ds_tr.std)

    dl_tr  = DataLoader(ds_train, batch_size=128,  shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val,   batch_size=128,  shuffle=False)
    dl_te  = DataLoader(ds_te,    batch_size=128,  shuffle=False)

    ensemble = []
    for i in range(5):
        torch.manual_seed(i * 100)
        net = CNN1D(C=Xtr.shape[1]).to(DEVICE)
        train_model(net, dl_tr, dl_val, epochs=epochs, model_name=f"CNN-{i}")
        ensemble.append(net)

    all_preds = []
    for net in ensemble:
        net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in dl_te:
                preds.append(net(xb.to(DEVICE)).cpu().numpy())
        all_preds.append(np.concatenate(preds))

    obs = []
    with torch.no_grad():
        for _, yb in dl_te:
            obs.append(yb.numpy())
    obs = np.concatenate(obs)

    ensemble_preds = np.mean(all_preds, axis=0)
    mse  = np.mean((obs - ensemble_preds) ** 2)
    pe   = 1 - mse / np.var(obs)

    return {
        "predictions": ensemble_preds,
        "observations": obs,
        "pe": pe,
        "rmse": np.sqrt(mse),
        "test_times": gic_data.sel(device=site, time=test_slice).time.values[
            window_size: window_size + len(obs)
        ],
    }


def model5b_gru(
    site, mag_data, gic_data, site_relations,
    train_slice, test_slice, window_size=256, epochs=40,
):
    """Train and evaluate a weighted GRU ensemble."""
    Xtr, ytr = feature_matrix_gru(site, mag_data, gic_data, site_relations, train_slice.start, train_slice.stop)
    Xte, yte = feature_matrix_gru(site, mag_data, gic_data, site_relations, test_slice.start,  test_slice.stop)

    ds_tr = WindowDS(Xtr, ytr, win=window_size)
    n_val = int(0.1 * len(ds_tr))
    ds_val, ds_train = torch.utils.data.random_split(
        ds_tr, [n_val, len(ds_tr) - n_val],
        generator=torch.Generator().manual_seed(42),
    )
    ds_te = WindowDS(Xte, yte, win=window_size, stats=ds_tr.stats, y_stats=ds_tr.y_stats)

    dl_tr  = DataLoader(ds_train, batch_size=1024, shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val,   batch_size=1024, shuffle=False)
    dl_te  = DataLoader(ds_te,    batch_size=1024, shuffle=False)

    ensemble_configs = [
        (GRU1S,           64),
        (GRU2S,           64),
        (GRUWithAttention, 64),
        (GRU1S,           128),
        (LSTMS,           64),
    ]

    ensemble_models, best_val_losses = [], []
    for i, (cls, hidden) in enumerate(ensemble_configs):
        net = cls(C=Xtr.shape[1], hidden=hidden).to(DEVICE)
        val_loss = train_model(net, dl_tr, dl_val, epochs=epochs, model_name=f"GRU-{i}")
        ensemble_models.append(net)
        best_val_losses.append(val_loss)

    # Weight models inversely by validation loss
    weights = 1.0 / (np.array(best_val_losses) + 1e-8)
    weights /= weights.sum()

    y_mean = ds_te.y_stats[0].item()
    y_std  = ds_te.y_stats[1].item()

    obs = []
    with torch.no_grad():
        for _, yb in dl_te:
            obs.append(yb)
    obs = torch.cat(obs).numpy() * y_std + y_mean

    all_preds = []
    for net in ensemble_models:
        net.eval()
        preds = []
        with torch.no_grad():
            for xb, _ in tqdm(dl_te, desc="inference", leave=False):
                preds.append(net(xb.to(DEVICE)).cpu().numpy())
        all_preds.append(np.concatenate(preds) * y_std + y_mean)

    ensemble_preds = sum(w * p for w, p in zip(weights, all_preds))
    mse = np.mean((obs - ensemble_preds) ** 2)
    pe  = 1 - mse / np.var(obs)

    return {
        "predictions": ensemble_preds,
        "observations": obs,
        "pe": pe,
        "rmse": np.sqrt(mse),
        "test_times": gic_data.sel(device=site, time=test_slice).time.values[
            window_size: window_size + len(obs)
        ],
    }