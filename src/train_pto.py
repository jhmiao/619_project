"""
Train the Predict-Then-Optimize (PTO) outcome model.

Input:
    project/data/train.csv

Features:
    group_B, x1..x5, pre_1..pre_30, A_obs

Target:
    Y_obs

Output:
    project/outputs/pto_model.pt
    project/outputs/pto_training_summary.csv

This script uses K-fold cross-validation on train.csv to choose a reasonable
number of epochs, then retrains one final model on the full training set.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# Allow running as: python src/train_pto.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models import OutcomeNet, count_parameters  # noqa: E402


SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 20
DEFAULT_N_FOLDS = 5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_feature_columns() -> list[str]:
    return (
        ["group_B"]
        + [f"x{i}" for i in range(1, 6)]
        + [f"pre_{i}" for i in range(1, 31)]
        + ["A_obs"]
    )


def load_data(data_path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find {data_path}. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(data_path)
    df["group_B"] = (df["group"] == "B").astype(float)

    feature_cols = get_feature_columns()
    missing = [col for col in feature_cols + ["Y_obs"] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    return df, feature_cols


def make_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, StandardScaler]:
    x_np = df[feature_cols].to_numpy(dtype=np.float32)
    y_np = df["Y_obs"].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        x_np = scaler.fit_transform(x_np).astype(np.float32)
    else:
        x_np = scaler.transform(x_np).astype(np.float32)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    return x, y, scaler


def train_one_model(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    seed: int,
) -> tuple[OutcomeNet, StandardScaler, dict[str, float]]:
    set_seed(seed)

    train_x, train_y, scaler = make_tensors(
        train_df, feature_cols, scaler=None, fit_scaler=True
    )
    val_x, val_y, _ = make_tensors(
        val_df, feature_cols, scaler=scaler, fit_scaler=False
    )

    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=batch_size,
        shuffle=True,
    )

    model = OutcomeNet(
        input_dim=len(feature_cols), hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_mse = float("inf")
    best_state = None
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x.to(device)).cpu()
            val_mse = loss_fn(val_pred, val_y).item()
            val_mae = torch.mean(torch.abs(val_pred - val_y)).item()

        if val_mse < best_val_mse - 1e-8:
            best_val_mse = val_mse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        train_pred = model(train_x.to(device)).cpu()
        val_pred = model(val_x.to(device)).cpu()
        train_mse = loss_fn(train_pred, train_y).item()
        train_mae = torch.mean(torch.abs(train_pred - train_y)).item()
        val_mse = loss_fn(val_pred, val_y).item()
        val_mae = torch.mean(torch.abs(val_pred - val_y)).item()

    metrics = {
        "best_epoch": float(best_epoch),
        "train_mse": train_mse,
        "train_mae": train_mae,
        "val_mse": val_mse,
        "val_mae": val_mae,
    }
    return model, scaler, metrics


def cross_validate(
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    lr: float,
    max_epochs: int,
    patience: int,
    n_folds: int,
    seed: int,
) -> pd.DataFrame:
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    rows = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(df), start=1):
        fold_train = df.iloc[train_idx].reset_index(drop=True)
        fold_val = df.iloc[val_idx].reset_index(drop=True)

        _, _, metrics = train_one_model(
            train_df=fold_train,
            val_df=fold_val,
            feature_cols=feature_cols,
            device=device,
            hidden_dim=hidden_dim,
            dropout=dropout,
            batch_size=batch_size,
            lr=lr,
            max_epochs=max_epochs,
            patience=patience,
            seed=seed + fold,
        )

        row = {"fold": fold, **metrics}
        rows.append(row)
        print(
            f"Fold {fold}/{n_folds}: "
            f"best_epoch={metrics['best_epoch']:.0f}, "
            f"val_mse={metrics['val_mse']:.5f}, "
            f"val_mae={metrics['val_mae']:.5f}"
        )

    return pd.DataFrame(rows)


def train_final_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    hidden_dim: int,
    dropout: float,
    batch_size: int,
    lr: float,
    n_epochs: int,
    seed: int,
) -> tuple[OutcomeNet, StandardScaler, dict[str, float]]:
    set_seed(seed)

    x, y, scaler = make_tensors(df, feature_cols, scaler=None, fit_scaler=True)
    loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    model = OutcomeNet(
        input_dim=len(feature_cols), hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs + 1):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        pred = model(x.to(device)).cpu()
        train_mse = loss_fn(pred, y).item()
        train_mae = torch.mean(torch.abs(pred - y)).item()

    metrics = {
        "final_epochs": float(n_epochs),
        "full_train_mse": train_mse,
        "full_train_mae": train_mae,
    }
    return model, scaler, metrics


def save_checkpoint(
    model: OutcomeNet,
    scaler: StandardScaler,
    feature_cols: list[str],
    args: argparse.Namespace,
    final_metrics: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "OutcomeNet",
        "feature_cols": feature_cols,
        "input_dim": len(feature_cols),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "args": vars(args),
        "metrics": final_metrics,
    }
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PTO outcome model.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "outputs" / "pto_model.pt")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "pto_training_summary.csv")
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df, feature_cols = load_data(args.data_path)
    print(f"Loaded {args.data_path}: {df.shape}")
    print(f"Input dim: {len(feature_cols)}")

    cv_summary = cross_validate(
        df=df,
        feature_cols=feature_cols,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        n_folds=args.n_folds,
        seed=args.seed,
    )

    mean_best_epoch = int(round(cv_summary["best_epoch"].mean()))
    final_epochs = max(1, mean_best_epoch)

    print("\nCross-validation summary:")
    print(cv_summary)
    print(
        f"\nRetraining final model on all training data for {final_epochs} epochs "
        f"(mean best epoch from CV)."
    )

    final_model, scaler, final_metrics = train_final_model(
        df=df,
        feature_cols=feature_cols,
        device=device,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        batch_size=args.batch_size,
        lr=args.lr,
        n_epochs=final_epochs,
        seed=args.seed,
    )

    print(f"Trainable parameters: {count_parameters(final_model)}")
    print("Final training metrics:")
    print(final_metrics)

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    cv_summary.to_csv(args.summary_path, index=False)
    save_checkpoint(
        model=final_model,
        scaler=scaler,
        feature_cols=feature_cols,
        args=args,
        final_metrics=final_metrics,
        output_path=args.output_path,
    )

    print(f"Saved PTO model to {args.output_path}")
    print(f"Saved CV summary to {args.summary_path}")


if __name__ == "__main__":
    main()
