

"""
Train a lightweight decision-focused learning (DFL) model.

Input:
    project/data/train.csv

Model:
    OutcomeNetNoSigmoid from models.py

Training idea:
    For each mini-batch, predict y0_hat and y1_hat by evaluating the same
    model twice, once with A_obs=0 and once with A_obs=1.

    Convert predicted Y values to smooth threshold-success utilities:
        S_hat = sigmoid((Y_hat - threshold) / threshold_temperature)

    Use a differentiable soft top-B allocation inside the batch:
        a_i = batch_budget * softmax(score_i / allocation_temperature)
    where score_i = S1_hat_i - S0_hat_i.

    Then optimize the downstream incremental objective:
        maximize sum_i a_i * (S1_hat_i - S0_hat_i)

    This avoids the degenerate solution where the model inflates S0_hat for
    everyone. Optionally add an MSE loss on observed Y_obs to stabilize training.

Output:
    project/outputs/dfl_model.pt
    project/outputs/dfl_training_summary.csv
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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models import OutcomeNetNoSigmoid, count_parameters  # noqa: E402


SEED = 42
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 150
DEFAULT_PATIENCE = 20
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_THRESHOLD = 0.60
DEFAULT_THRESHOLD_TEMPERATURE = 0.05
DEFAULT_ALLOCATION_TEMPERATURE = 0.10
DEFAULT_BUDGET_FRACTION = 0.20
DEFAULT_MSE_WEIGHT = 1.00
DEFAULT_FAIRNESS_WEIGHT = 0.0


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
    required_cols = feature_cols + ["Y_obs", "A_obs", "group_B"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {data_path}: {missing}")

    return df, feature_cols


def make_tensors(
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler | None = None,
    fit_scaler: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, StandardScaler]:
    x_np = df[feature_cols].to_numpy(dtype=np.float32)
    y_np = df["Y_obs"].to_numpy(dtype=np.float32)
    group_np = df["group_B"].to_numpy(dtype=np.float32)

    if scaler is None:
        scaler = StandardScaler()

    if fit_scaler:
        x_np = scaler.fit_transform(x_np).astype(np.float32)
    else:
        x_np = scaler.transform(x_np).astype(np.float32)

    x = torch.from_numpy(x_np)
    y = torch.from_numpy(y_np)
    group = torch.from_numpy(group_np)
    return x, y, group, scaler


def make_counterfactual_features(
    x: torch.Tensor,
    feature_cols: list[str],
    action_value: float,
) -> torch.Tensor:
    """Return a copy of x with the A_obs column set to action_value.

    Important: this happens after scaling. Therefore action_value must be converted
    to the scaled value using the caller's already-scaled representation.
    This function is kept for completeness, but the training code uses a safer
    unscaled-to-scaled helper below.
    """
    action_idx = feature_cols.index("A_obs")
    x_cf = x.clone()
    x_cf[:, action_idx] = action_value
    return x_cf


def make_scaled_counterfactuals(
    x_unscaled: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create scaled feature tensors with A_obs fixed to 0 and 1."""
    action_idx = feature_cols.index("A_obs")

    x0 = x_unscaled.clone()
    x1 = x_unscaled.clone()
    x0[:, action_idx] = 0.0
    x1[:, action_idx] = 1.0

    x0_scaled = (x0 - scaler_mean) / scaler_scale
    x1_scaled = (x1 - scaler_mean) / scaler_scale
    return x0_scaled, x1_scaled


def smooth_step(y_hat: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    """Differentiable approximation to 1{Y > threshold}."""
    return torch.sigmoid((y_hat - threshold) / temperature)


def soft_allocate(scores: torch.Tensor, budget_fraction: float, temperature: float) -> torch.Tensor:
    """Soft top-B allocation within a batch.

    Returns a_i in [0, approximately batch_budget]. Sum is batch_budget.
    This is a differentiable surrogate, not a feasible binary allocation.
    """
    batch_size = scores.shape[0]
    batch_budget = max(1.0, budget_fraction * batch_size)
    allocation = batch_budget * torch.softmax(scores / temperature, dim=0)
    return torch.clamp(allocation, min=0.0, max=1.0)


def group_rate_fairness_penalty(
    s_policy: torch.Tensor,
    group_b: torch.Tensor,
) -> torch.Tensor:
    """Equal-rate fairness penalty across groups A and B.

    penalty = sum_g (r_g - r_overall)^2.
    If a mini-batch happens to contain only one group, returns zero.
    """
    overall_rate = s_policy.mean()
    penalty = torch.tensor(0.0, device=s_policy.device)

    for group_value in [0.0, 1.0]:
        mask = group_b == group_value
        if mask.any():
            group_rate = s_policy[mask].mean()
            penalty = penalty + (group_rate - overall_rate) ** 2

    return penalty


def dfl_loss_for_batch(
    model: OutcomeNetNoSigmoid,
    x_scaled: torch.Tensor,
    x_unscaled: torch.Tensor,
    y_obs: torch.Tensor,
    group_b: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    threshold: float,
    threshold_temperature: float,
    allocation_temperature: float,
    budget_fraction: float,
    mse_weight: float,
    fairness_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DFL loss plus logging metrics for one batch."""
    x0_scaled, x1_scaled = make_scaled_counterfactuals(
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
    )

    y0_hat_raw = model(x0_scaled)
    y1_hat_raw = model(x1_scaled)
    y_obs_hat_raw = model(x_scaled)

    # Raw model has no sigmoid; constrain predicted Y to [0,1].
    y0_hat = torch.sigmoid(y0_hat_raw)
    y1_hat = torch.sigmoid(y1_hat_raw)
    y_obs_hat = torch.sigmoid(y_obs_hat_raw)

    s0_hat = smooth_step(y0_hat, threshold=threshold, temperature=threshold_temperature)
    s1_hat = smooth_step(y1_hat, threshold=threshold, temperature=threshold_temperature)

    scores = s1_hat - s0_hat
    a_soft = soft_allocate(
        scores=scores,
        budget_fraction=budget_fraction,
        temperature=allocation_temperature,
    )

    # Incremental decision value: only the selected mass receives credit for
    # predicted treatment benefit. This is the key DFL objective for allocation.
    incremental_objective = torch.sum(a_soft * scores)

    # For fairness, evaluate the predicted success rate under the soft policy.
    # This term is optional and currently off by default.
    s_policy = a_soft * s1_hat + (1.0 - a_soft) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)

    # Supervised stabilizer to keep counterfactual predictions grounded in Y_obs.
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)

    # Minimize negative incremental objective, plus optional stabilizers.
    loss = -incremental_objective / x_scaled.shape[0]
    loss = loss + fairness_weight * fairness_penalty
    loss = loss + mse_weight * mse_loss

    logs = {
        "loss": float(loss.detach().cpu()),
        "incremental_objective_mean": float((incremental_objective / x_scaled.shape[0]).detach().cpu()),
        "policy_success_mean": float(s_policy.mean().detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "fairness_penalty": float(fairness_penalty.detach().cpu()),
        "mean_soft_allocation": float(a_soft.mean().detach().cpu()),
        "mean_score": float(scores.mean().detach().cpu()),
        "max_score": float(scores.max().detach().cpu()),
        "min_score": float(scores.min().detach().cpu()),
    }
    return loss, logs


def train_one_epoch(
    model: OutcomeNetNoSigmoid,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.train()
    logs_accum: dict[str, list[float]] = {}

    for x_scaled, x_unscaled, y_obs, group_b in loader:
        x_scaled = x_scaled.to(device)
        x_unscaled = x_unscaled.to(device)
        y_obs = y_obs.to(device)
        group_b = group_b.to(device)

        loss, logs = dfl_loss_for_batch(
            model=model,
            x_scaled=x_scaled,
            x_unscaled=x_unscaled,
            y_obs=y_obs,
            group_b=group_b,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            threshold=args.threshold,
            threshold_temperature=args.threshold_temperature,
            allocation_temperature=args.allocation_temperature,
            budget_fraction=args.budget_fraction,
            mse_weight=args.mse_weight,
            fairness_weight=args.fairness_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        for key, value in logs.items():
            logs_accum.setdefault(key, []).append(value)

    return {key: float(np.mean(values)) for key, values in logs_accum.items()}


def evaluate_loss(
    model: OutcomeNetNoSigmoid,
    loader: DataLoader,
    device: torch.device,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> dict[str, float]:
    model.eval()
    logs_accum: dict[str, list[float]] = {}

    with torch.no_grad():
        for x_scaled, x_unscaled, y_obs, group_b in loader:
            x_scaled = x_scaled.to(device)
            x_unscaled = x_unscaled.to(device)
            y_obs = y_obs.to(device)
            group_b = group_b.to(device)

            _, logs = dfl_loss_for_batch(
                model=model,
                x_scaled=x_scaled,
                x_unscaled=x_unscaled,
                y_obs=y_obs,
                group_b=group_b,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                feature_cols=feature_cols,
                threshold=args.threshold,
                threshold_temperature=args.threshold_temperature,
                allocation_temperature=args.allocation_temperature,
                budget_fraction=args.budget_fraction,
                mse_weight=args.mse_weight,
                fairness_weight=args.fairness_weight,
            )

            for key, value in logs.items():
                logs_accum.setdefault(key, []).append(value)

    return {key: float(np.mean(values)) for key, values in logs_accum.items()}


def make_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, StandardScaler]:
    x_train_unscaled_np = train_df[feature_cols].to_numpy(dtype=np.float32)
    y_train_np = train_df["Y_obs"].to_numpy(dtype=np.float32)
    group_train_np = train_df["group_B"].to_numpy(dtype=np.float32)

    x_val_unscaled_np = val_df[feature_cols].to_numpy(dtype=np.float32)
    y_val_np = val_df["Y_obs"].to_numpy(dtype=np.float32)
    group_val_np = val_df["group_B"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled_np = scaler.fit_transform(x_train_unscaled_np).astype(np.float32)
    x_val_scaled_np = scaler.transform(x_val_unscaled_np).astype(np.float32)

    train_dataset = TensorDataset(
        torch.from_numpy(x_train_scaled_np),
        torch.from_numpy(x_train_unscaled_np),
        torch.from_numpy(y_train_np),
        torch.from_numpy(group_train_np),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(x_val_scaled_np),
        torch.from_numpy(x_val_unscaled_np),
        torch.from_numpy(y_val_np),
        torch.from_numpy(group_val_np),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, scaler


def save_checkpoint(
    model: OutcomeNetNoSigmoid,
    scaler: StandardScaler,
    feature_cols: list[str],
    args: argparse.Namespace,
    best_val_logs: dict[str, float],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "OutcomeNetNoSigmoid",
        "feature_cols": feature_cols,
        "input_dim": len(feature_cols),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "args": vars(args),
        "metrics": best_val_logs,
    }
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train lightweight DFL model.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_model.pt")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_training_summary.csv")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--threshold-temperature", type=float, default=DEFAULT_THRESHOLD_TEMPERATURE)
    parser.add_argument("--allocation-temperature", type=float, default=DEFAULT_ALLOCATION_TEMPERATURE)
    parser.add_argument("--budget-fraction", type=float, default=DEFAULT_BUDGET_FRACTION)
    parser.add_argument("--mse-weight", type=float, default=DEFAULT_MSE_WEIGHT)
    parser.add_argument("--fairness-weight", type=float, default=DEFAULT_FAIRNESS_WEIGHT)
    parser.add_argument("--val-frac", type=float, default=0.15)
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

    train_df, val_df = train_test_split(
        df,
        test_size=args.val_frac,
        random_state=args.seed,
        shuffle=True,
        stratify=df["group"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_loader, val_loader, scaler = make_loaders(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        batch_size=args.batch_size,
    )

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    model = OutcomeNetNoSigmoid(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Trainable parameters: {count_parameters(model)}")
    print(
        "DFL settings: "
        f"budget_fraction={args.budget_fraction}, "
        f"threshold={args.threshold}, "
        f"threshold_temperature={args.threshold_temperature}, "
        f"allocation_temperature={args.allocation_temperature}, "
        f"mse_weight={args.mse_weight}, "
        f"fairness_weight={args.fairness_weight}"
    )

    best_val_loss = float("inf")
    best_state = None
    best_val_logs: dict[str, float] = {}
    epochs_without_improvement = 0
    summary_rows = []

    for epoch in range(1, args.epochs + 1):
        train_logs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=args,
        )
        val_logs = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=args,
        )

        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_logs.items()})
        row.update({f"val_{k}": v for k, v in val_logs.items()})
        summary_rows.append(row)

        val_loss = val_logs["loss"]
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_logs = val_logs.copy()
            best_val_logs["best_epoch"] = float(epoch)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d}: "
                f"train_loss={train_logs['loss']:.5f}, "
                f"val_loss={val_logs['loss']:.5f}, "
                f"val_incr_obj={val_logs['incremental_objective_mean']:.5f}, "
                f"val_policy_success={val_logs['policy_success_mean']:.5f}, "
                f"val_mse={val_logs['mse_loss']:.5f}"
            )

        if epochs_without_improvement >= args.patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    summary_df = pd.DataFrame(summary_rows)
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(args.summary_path, index=False)
    save_checkpoint(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        args=args,
        best_val_logs=best_val_logs,
        output_path=args.output_path,
    )

    print("Best validation logs:")
    print(best_val_logs)
    print(f"Saved DFL model to {args.output_path}")
    print(f"Saved training summary to {args.summary_path}")


if __name__ == "__main__":
    main()