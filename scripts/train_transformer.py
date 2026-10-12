"""
Transformer-based efficiency prediction model for ABE8e gRNA design.
Two variants controlled by cfg.use_energy:
  - sequence_only (use_energy=False): 20nt gRNA one-hot -> Transformer -> efficiency
  - sequence_energy (use_energy=True): 20nt + 27 energy/phys -> Transformer + MLP -> efficiency

Compares against XGBoost baseline (Test Spearman 0.751) on identical .npz splits.
Run: python scripts/train_transformer.py
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error
import math
import os
import json


# ============================================================================
# Configuration
# ============================================================================
class Config:
    # Data layout: X = [80 one-hot seq | 24 energy | 3 phys] = 107 dims
    seq_len = 20
    seq_dim = 4          # A, T, G, C
    energy_dim = 27      # 24 energy + 3 physical

    # Model (kept small to avoid overfitting on ~15k samples)
    d_model = 64
    nhead = 4
    num_layers = 2
    dim_feedforward = 128
    dropout = 0.1
    use_energy = True    # <<< toggle: True = full features, False = sequence only

    # Training
    batch_size = 64
    lr = 1e-4
    weight_decay = 1e-4
    warmup_steps = 500
    epochs = 100
    early_stop_patience = 15
    grad_clip = 1.0
    seed = 42

    # Paths
    feature_dir = "data/features"
    output_dir = "models/transformer"
    results_dir = "results"

    @property
    def model_name(self):
        tag = "energy" if self.use_energy else "seqonly"
        return f"transformer_d{self.d_model}_l{self.num_layers}_{tag}"


cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(cfg.results_dir, exist_ok=True)


# ============================================================================
# Reproducibility
# ============================================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset
# ============================================================================
class ABEDataset(Dataset):
    def __init__(self, split, use_energy=True):
        data = np.load(f"{cfg.feature_dir}/{split}.npz")
        X, y = data["X"], data["y"]

        # Split features: [80 seq | 27 energy/phys]
        self.seq = X[:, :80].reshape(-1, cfg.seq_len, cfg.seq_dim).astype(np.float32)
        self.energy = X[:, 80:].astype(np.float32) if use_energy else None
        self.y = y.astype(np.float32)
        self.use_energy = use_energy

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        if self.use_energy:
            return self.seq[idx], self.energy[idx], self.y[idx]
        return self.seq[idx], self.y[idx]


# ============================================================================
# Model
# ============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=32):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerEfficiencyModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.use_energy = cfg.use_energy

        self.seq_embed = nn.Linear(cfg.seq_dim, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)

        if self.use_energy:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model + cfg.energy_dim, cfg.d_model),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, 1),
            )
        else:
            self.head = nn.Sequential(
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.ReLU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model // 2, 1),
            )

    def forward(self, seq, energy=None):
        x = self.seq_embed(seq)      # [B, 20, d_model]
        x = self.pos_enc(x)
        x = self.transformer(x)      # [B, 20, d_model]
        x = x.mean(dim=1)            # mean pooling -> [B, d_model]
        if self.use_energy and energy is not None:
            x = torch.cat([x, energy], dim=1)
        return self.head(x).squeeze(-1)


# ============================================================================
# Training utilities
# ============================================================================
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def evaluate(model, loader, device, use_energy):
    model.eval()
    preds, targets, total_loss = [], [], 0.0
    criterion = nn.MSELoss()
    with torch.no_grad():
        for batch in loader:
            if use_energy:
                seq, energy, y = [b.to(device) for b in batch]
                pred = model(seq, energy)
            else:
                seq, y = [b.to(device) for b in batch]
                pred = model(seq)
            loss = criterion(pred, y)
            total_loss += loss.item() * len(y)
            preds.extend(pred.cpu().numpy())
            targets.extend(y.cpu().numpy())
    preds, targets = np.array(preds), np.array(targets)
    return {
        "loss": total_loss / len(targets),
        "spearman": float(spearmanr(targets, preds)[0]),
        "pearson": float(pearsonr(targets, preds)[0]),
        "mae": float(mean_absolute_error(targets, preds)),
        "rmse": float(np.sqrt(mean_squared_error(targets, preds))),
    }


# ============================================================================
# Main
# ============================================================================
def train():
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Model: {cfg.model_name} | use_energy={cfg.use_energy}\n")

    train_ds = ABEDataset("train", use_energy=cfg.use_energy)
    val_ds = ABEDataset("val", use_energy=cfg.use_energy)
    test_ds = ABEDataset("test", use_energy=cfg.use_energy)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}\n")

    model = TransformerEfficiencyModel(cfg).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = len(train_loader) * cfg.epochs
    scheduler = get_lr_scheduler(optimizer, cfg.warmup_steps, total_steps)
    criterion = nn.MSELoss()

    best_val_spearman = -1.0
    best_epoch = -1
    patience_counter = 0
    ckpt_path = os.path.join(cfg.output_dir, f"{cfg.model_name}_best.pt")

    for epoch in range(cfg.epochs):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            if cfg.use_energy:
                seq, energy, y = [b.to(device) for b in batch]
                pred = model(seq, energy)
            else:
                seq, y = [b.to(device) for b in batch]
                pred = model(seq)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item() * len(y)
        train_loss /= len(train_ds)

        val_m = evaluate(model, val_loader, device, cfg.use_energy)
        print(
            f"Epoch {epoch+1:3d} | Loss {train_loss:.4f} | "
            f"Val Spearman {val_m['spearman']:.3f} | "
            f"Pearson {val_m['pearson']:.3f} | MAE {val_m['mae']:.3f}"
        )

        if val_m["spearman"] > best_val_spearman:
            best_val_spearman = val_m["spearman"]
            best_epoch = epoch + 1
            patience_counter = 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_spearman": best_val_spearman}, ckpt_path)
            print(f"  -> saved best (Val Spearman {best_val_spearman:.3f})")
        else:
            patience_counter += 1
            if patience_counter >= cfg.early_stop_patience:
                print(f"\nEarly stopping at epoch {epoch+1} "
                      f"(best epoch {best_epoch}, Val Spearman {best_val_spearman:.3f})")
                break

    # ---- Final evaluation on best checkpoint ----
    print("\nLoading best checkpoint for final evaluation...")
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    train_m = evaluate(model, train_loader, device, cfg.use_energy)
    val_m = evaluate(model, val_loader, device, cfg.use_energy)
    test_m = evaluate(model, test_loader, device, cfg.use_energy)

    print("\n" + "=" * 62)
    print(f"Final Results ({cfg.model_name})  [baseline Test Spearman = 0.751]")
    print("=" * 62)
    print(f"{'Split':<8}{'Spearman':>11}{'Pearson':>11}{'MAE':>10}{'RMSE':>10}")
    print("-" * 62)
    for name, m in [("Train", train_m), ("Val", val_m), ("Test", test_m)]:
        print(f"{name:<8}{m['spearman']:>11.3f}{m['pearson']:>11.3f}"
              f"{m['mae']:>10.3f}{m['rmse']:>10.3f}")
    print("=" * 62)

    results = {
        "model_name": cfg.model_name,
        "use_energy": cfg.use_energy,
        "best_epoch": best_epoch,
        "config": {k: getattr(cfg, k) for k in
                   ["d_model", "nhead", "num_layers", "dim_feedforward",
                    "dropout", "batch_size", "lr", "weight_decay", "warmup_steps"]},
        "metrics": {"train": train_m, "val": val_m, "test": test_m},
    }
    out_path = os.path.join(cfg.results_dir, f"{cfg.model_name}_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results -> {out_path}")
    print(f"Saved model   -> {ckpt_path}")


if __name__ == "__main__":
    train()