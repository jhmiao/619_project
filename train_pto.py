

"""Train a predict-then-optimize neural network model.

Input file:
    pto_training_data.csv

Expected columns:
    day_1, day_2, ..., day_30, target

The script trains a small MLP using K-fold cross-validation and reports
validation metrics for each fold and overall. It also trains a final model on
all data and saves it to pto_model.pt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass(frozen=True)
class TrainConfig:
    data_path: Path = Path("pto_training_data.csv")
    model_path: Path = Path("pto_model.pt")
    n_splits: int = 5
    seed: int = 619
    batch_size: int = 128
    hidden_dim_1: int = 32
    hidden_dim_2: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20


def checkpoint_config(config: TrainConfig) -> dict[str, object]:
    """Convert config to checkpoint-safe primitive values."""

    values = asdict(config)
    return {
        key: str(value) if isinstance(value, Path) else value
        for key, value in values.items()
    }


class PTOModel(nn.Module):
    """Small MLP for predicting treatment-effect target from 30-day history."""

    def __init__(self, input_dim: int, hidden_dim_1: int, hidden_dim_2: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load feature matrix X and target vector y."""

    if not path.exists():
        raise FileNotFoundError(f"Could not find data file: {path.resolve()}")

    df = pd.read_csv(path)
    feature_cols = [f"day_{i}" for i in range(1, 31)]
    missing_cols = [col for col in feature_cols + ["target"] if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["target"].to_numpy(dtype=np.float32)
    return X, y, feature_cols


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """Create a PyTorch DataLoader from NumPy arrays."""

    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_one_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> tuple[PTOModel, dict[str, float]]:
    """Train one fold with early stopping on validation MSE."""

    model = PTOModel(
        input_dim=X_train.shape[1],
        hidden_dim_1=config.hidden_dim_1,
        hidden_dim_2=config.hidden_dim_2,
    ).to(device)

    train_loader = make_loader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = make_loader(X_val, y_val, config.batch_size, shuffle=False)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    best_state = None
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

        val_loss = evaluate_loss(model, val_loader, criterion, device)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= config.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    val_pred = predict(model, X_val, config.batch_size, device)
    metrics = compute_metrics(y_val, val_pred)
    metrics["best_val_mse_loss"] = best_val_loss
    return model, metrics


def evaluate_loss(
    model: PTOModel,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """Evaluate average loss over a DataLoader."""

    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            total_loss += float(loss.item()) * len(batch_y)
            total_count += len(batch_y)

    return total_loss / max(total_count, 1)


def predict(
    model: PTOModel,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Generate predictions for X."""

    dummy_y = np.zeros(X.shape[0], dtype=np.float32)
    loader = make_loader(X, dummy_y, batch_size, shuffle=False)

    model.eval()
    preds: list[np.ndarray] = []
    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            pred = model(batch_x).detach().cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute regression metrics."""

    mse = mean_squared_error(y_true, y_pred)
    return {
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Run K-fold cross-validation."""

    kfold = KFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)
    rows: list[dict[str, float]] = []

    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        _, metrics = train_one_fold(X_train, y_train, X_val, y_val, config, device)
        metrics["fold"] = fold_idx
        rows.append(metrics)

        print(
            f"Fold {fold_idx}: "
            f"MSE={metrics['mse']:.4f}, "
            f"RMSE={metrics['rmse']:.4f}, "
            f"MAE={metrics['mae']:.4f}, "
            f"R2={metrics['r2']:.4f}"
        )

    return pd.DataFrame(rows)


def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> PTOModel:
    """Train final model on all available data."""

    # Use a small validation slice for early stopping while still fitting a final model.
    rng = np.random.default_rng(config.seed)
    indices = rng.permutation(len(X))
    val_size = max(int(0.1 * len(X)), 1)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    model, _ = train_one_fold(
        X_train=X[train_idx],
        y_train=y[train_idx],
        X_val=X[val_idx],
        y_val=y[val_idx],
        config=config,
        device=device,
    )
    return model


def main() -> None:
    """Run cross-validation and train final PTO model."""

    config = TrainConfig()
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, feature_cols = load_data(config.data_path)
    print(f"Loaded {len(X):,} samples with {len(feature_cols)} features from {config.data_path}")

    cv_results = cross_validate(X, y, config, device)
    print("\nCross-validation summary:")
    summary = cv_results.drop(columns=["fold"]).agg(["mean", "std"])
    print(summary)

    cv_results.to_csv("pto_cv_results.csv", index=False)
    print("\nSaved cross-validation results to pto_cv_results.csv")

    final_model = train_final_model(X, y, config, device)
    torch.save(
        {
            "model_state_dict": final_model.state_dict(),
            "feature_cols": feature_cols,
            "config": checkpoint_config(config),
        },
        config.model_path,
    )
    print(f"Saved final model to {config.model_path}")


if __name__ == "__main__":
    main()
