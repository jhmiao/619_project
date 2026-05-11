"""
Train all decision-focused learning model variants.

Input:
    project/data/train.csv

Models:
    OutcomeNetNoSigmoid from models.py

Training variants:
    dfl:
        Differentiable soft top-B allocation.
    rs:
        Randomized smoothing over hard top-B allocations.
    pg:
        Perturbation-gradient estimators for hard top-B allocations.

Output:
    project/outputs/dfl_model.pt
    project/outputs/dfl_training_summary.csv
    project/outputs/rs_model.pt
    project/outputs/rs_training_summary.csv
    project/outputs/pg_<estimator>_model.pt
    project/outputs/pg_<estimator>_training_summary.csv

Each variant uses K-fold cross-validation to choose the final epoch count, then
re-trains one final model on the full training set.
"""

from __future__ import annotations

import argparse
import random
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models import OutcomeNetNoSigmoid, count_parameters  # noqa: E402


SEED = 10
DEFAULT_BATCH_SIZE = 256
DEFAULT_LR = 1e-3
DEFAULT_EPOCHS = 150
DEFAULT_PATIENCE = 20
DEFAULT_N_FOLDS = 5
DEFAULT_HIDDEN_DIM = 64
DEFAULT_DROPOUT = 0.1
DEFAULT_THRESHOLD = 0.60
DEFAULT_THRESHOLD_TEMPERATURE = 0.05
DEFAULT_ALLOCATION_TEMPERATURE = 0.10
DEFAULT_BUDGET_FRACTION = 0.10
DEFAULT_MSE_WEIGHT = 1.00
DEFAULT_FAIRNESS_WEIGHT = 1.00
DEFAULT_N_SMOOTHING_SAMPLES = 10
DEFAULT_N_PERTURB_SAMPLES = 10
DEFAULT_NOISE_STD = 0.10
DEFAULT_PG_WEIGHT = 1.00
DEFAULT_PG_MSE_WEIGHT = 1.00
DEFAULT_PG_FAIRNESS_WEIGHT = DEFAULT_FAIRNESS_WEIGHT

DEFAULT_RANDOMIZE_BUDGET = False
DEFAULT_BUDGET_FRACTION_MIN = 0.001
DEFAULT_BUDGET_FRACTION_MAX = 0.10
# PG_ESTIMATORS = ["score_function", "forward", "backward", "central"]
PG_ESTIMATORS = [ "forward", "backward", "central"]
MODEL_VARIANTS = ["dfl", "rs", "pg"]

LossFn = Callable[
    [
        OutcomeNetNoSigmoid,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[str],
        argparse.Namespace,
    ],
    tuple[torch.Tensor, dict[str, float]],
]


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


def make_loader(
    df: pd.DataFrame,
    feature_cols: list[str],
    batch_size: int,
    shuffle: bool,
) -> tuple[DataLoader, StandardScaler]:
    x_unscaled_np = df[feature_cols].to_numpy(dtype=np.float32)
    y_np = df["Y_obs"].to_numpy(dtype=np.float32)
    group_np = df["group_B"].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_scaled_np = scaler.fit_transform(x_unscaled_np).astype(np.float32)

    dataset = TensorDataset(
        torch.from_numpy(x_scaled_np),
        torch.from_numpy(x_unscaled_np),
        torch.from_numpy(y_np),
        torch.from_numpy(group_np),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader, scaler


def make_scaled_counterfactuals(
    x_unscaled: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
) -> tuple[torch.Tensor, torch.Tensor]:
    action_idx = feature_cols.index("A_obs")

    x0 = x_unscaled.clone()
    x1 = x_unscaled.clone()
    x0[:, action_idx] = 0.0
    x1[:, action_idx] = 1.0

    x0_scaled = (x0 - scaler_mean) / scaler_scale
    x1_scaled = (x1 - scaler_mean) / scaler_scale
    return x0_scaled, x1_scaled


def predict_batch_components(
    model: OutcomeNetNoSigmoid,
    x_scaled: torch.Tensor,
    x_unscaled: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    threshold: float,
    threshold_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x0_scaled, x1_scaled = make_scaled_counterfactuals(
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
    )

    y0_hat = torch.sigmoid(model(x0_scaled))
    y1_hat = torch.sigmoid(model(x1_scaled))
    y_obs_hat = torch.sigmoid(model(x_scaled))
    s0_hat = smooth_step(y0_hat, threshold=threshold, temperature=threshold_temperature)
    s1_hat = smooth_step(y1_hat, threshold=threshold, temperature=threshold_temperature)
    scores = s1_hat - s0_hat
    return y_obs_hat, s0_hat, s1_hat, scores, y1_hat - y0_hat


def smooth_step(y_hat: torch.Tensor, threshold: float, temperature: float) -> torch.Tensor:
    return torch.sigmoid((y_hat - threshold) / temperature)


def soft_allocate(scores: torch.Tensor, budget_fraction: float, temperature: float) -> torch.Tensor:
    batch_size = scores.shape[0]
    batch_budget = max(1, int(round(budget_fraction * batch_size)))
    allocation = float(batch_budget) * torch.softmax(scores / temperature, dim=0)
    return torch.clamp(allocation, min=0.0, max=1.0)


def hard_top_b(scores: torch.Tensor, budget: int) -> torch.Tensor:
    allocation = torch.zeros_like(scores)
    n = scores.shape[0]
    if budget <= 0 or n == 0:
        return allocation

    k = min(budget, n)
    top_idx = torch.topk(scores, k=k, largest=True).indices
    allocation[top_idx] = 1.0
    return allocation

def get_batch_budget_fraction(args: argparse.Namespace) -> float:
    """Return the budget fraction used for the current mini-batch."""
    if getattr(args, "randomize_budget", False):
        return float(
            np.random.uniform(
                args.budget_fraction_min,
                args.budget_fraction_max,
            )
        )
    return float(args.budget_fraction)


def randomized_smoothing_allocate(
    scores: torch.Tensor,
    budget_fraction: float,
    n_samples: int,
    noise_std: float,
) -> torch.Tensor:
    batch_size = scores.shape[0]
    batch_budget = max(1, int(round(budget_fraction * batch_size)))

    allocations = []
    for _ in range(n_samples):
        noise = torch.randn_like(scores) * noise_std
        noisy_scores = scores + noise
        allocations.append(hard_top_b(noisy_scores, budget=batch_budget))

    return torch.stack(allocations, dim=0).mean(dim=0)


def objective_for_allocation(allocation: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    return torch.sum(allocation * scores)


def perturbation_gradient_surrogate(
    scores: torch.Tensor,
    budget_fraction: float,
    n_samples: int,
    noise_std: float,
    estimator: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if estimator not in PG_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}. Use one of {PG_ESTIMATORS}.")

    batch_size = scores.shape[0]
    batch_budget = max(1, int(round(budget_fraction * batch_size)))
    h = max(noise_std, 1e-8)

    scores_detached = scores.detach()
    allocation_base = hard_top_b(scores_detached, budget=batch_budget)
    objective_base = objective_for_allocation(allocation_base, scores_detached)

    eps_list = []
    objective_list = []
    grad_terms = []
    allocation_list = []

    for _ in range(n_samples):
        eps = torch.randn_like(scores_detached)
        scores_plus = scores_detached + h * eps
        scores_minus = scores_detached - h * eps

        allocation_plus = hard_top_b(scores_plus, budget=batch_budget)
        allocation_minus = hard_top_b(scores_minus, budget=batch_budget)

        objective_plus = objective_for_allocation(allocation_plus, scores_detached)
        objective_minus = objective_for_allocation(allocation_minus, scores_detached)

        if estimator == "score_function":
            directional_derivative = None
            objective_for_logging = objective_plus
            allocation_for_logging = allocation_plus
        elif estimator == "forward":
            directional_derivative = (objective_plus - objective_base) / h
            objective_for_logging = objective_plus
            allocation_for_logging = allocation_plus
        elif estimator == "backward":
            directional_derivative = (objective_base - objective_minus) / h
            objective_for_logging = objective_minus
            allocation_for_logging = allocation_minus
        elif estimator == "central":
            directional_derivative = (objective_plus - objective_minus) / (2.0 * h)
            objective_for_logging = 0.5 * (objective_plus + objective_minus)
            allocation_for_logging = 0.5 * (allocation_plus + allocation_minus)
        else:
            raise RuntimeError("Unreachable estimator branch.")

        eps_list.append(eps)
        objective_list.append(objective_for_logging)
        allocation_list.append(allocation_for_logging)
        if directional_derivative is not None:
            grad_terms.append(directional_derivative * eps)

    eps_tensor = torch.stack(eps_list, dim=0)
    objectives = torch.stack(objective_list, dim=0)
    allocations = torch.stack(allocation_list, dim=0)

    if estimator == "score_function":
        baseline = objectives.mean()
        centered_objectives = objectives - baseline
        grad_scores = torch.mean(centered_objectives[:, None] * eps_tensor, dim=0) / h
    else:
        grad_scores = torch.stack(grad_terms, dim=0).mean(dim=0)

    surrogate_loss = -torch.sum(grad_scores.detach() * scores) / batch_size

    avg_allocation = allocations.mean(dim=0)
    avg_objective = torch.sum(avg_allocation * scores_detached)

    logs = {
        "pg_surrogate_loss": float(surrogate_loss.detach().cpu()),
        "perturbed_objective_mean": float(objectives.mean().detach().cpu() / batch_size),
        "perturbed_objective_std": float(objectives.std(unbiased=False).detach().cpu() / batch_size),
        "base_objective_mean": float((objective_base / batch_size).detach().cpu()),
        "avg_allocation_mean": float(avg_allocation.mean().detach().cpu()),
        "avg_allocation_objective_mean": float((avg_objective / batch_size).detach().cpu()),
        "grad_score_norm": float(torch.norm(grad_scores).detach().cpu()),
    }
    return surrogate_loss, logs


def group_rate_fairness_penalty(
    s_policy: torch.Tensor,
    group_b: torch.Tensor,
) -> torch.Tensor:
    overall_rate = s_policy.mean()
    penalty = torch.tensor(0.0, device=s_policy.device)

    for group_value in [0.0, 1.0]:
        mask = group_b == group_value
        if mask.any():
            group_rate = s_policy[mask].mean()
            penalty = penalty + (group_rate - overall_rate) ** 2

    return penalty


def add_common_logs(
    logs: dict[str, float],
    loss: torch.Tensor,
    scores: torch.Tensor,
    s_policy: torch.Tensor,
    mse_loss: torch.Tensor,
    fairness_penalty: torch.Tensor,
) -> dict[str, float]:
    return {
        "loss": float(loss.detach().cpu()),
        "policy_success_mean": float(s_policy.mean().detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "fairness_penalty": float(fairness_penalty.detach().cpu()),
        "mean_score": float(scores.mean().detach().cpu()),
        "max_score": float(scores.max().detach().cpu()),
        "min_score": float(scores.min().detach().cpu()),
        **logs,
    }


def dfl_loss_for_batch(
    model: OutcomeNetNoSigmoid,
    x_scaled: torch.Tensor,
    x_unscaled: torch.Tensor,
    y_obs: torch.Tensor,
    group_b: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_obs_hat, s0_hat, s1_hat, scores, _ = predict_batch_components(
        model=model,
        x_scaled=x_scaled,
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
        threshold=args.threshold,
        threshold_temperature=args.threshold_temperature,
    )
    budget_fraction = get_batch_budget_fraction(args)

    a_soft = soft_allocate(
    scores=scores,
    budget_fraction=budget_fraction,
    temperature=args.allocation_temperature,
    )
    a_soft = soft_allocate(
        scores=scores,
        budget_fraction=budget_fraction,
        temperature=args.allocation_temperature,
    )

    incremental_objective = torch.sum(a_soft * scores)
    s_policy = a_soft * s1_hat + (1.0 - a_soft) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)

    loss = -incremental_objective / x_scaled.shape[0]
    loss = loss + args.fairness_weight * fairness_penalty
    loss = loss + args.mse_weight * mse_loss

    logs = {
        "incremental_objective_mean": float((incremental_objective / x_scaled.shape[0]).detach().cpu()),
        "mean_soft_allocation": float(a_soft.mean().detach().cpu()),
        "budget_fraction": float(budget_fraction),
        "batch_budget": float(max(1, int(round(budget_fraction * x_scaled.shape[0])))),
    }
    return loss, add_common_logs(logs, loss, scores, s_policy, mse_loss, fairness_penalty)


def rs_loss_for_batch(
    model: OutcomeNetNoSigmoid,
    x_scaled: torch.Tensor,
    x_unscaled: torch.Tensor,
    y_obs: torch.Tensor,
    group_b: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_obs_hat, s0_hat, s1_hat, scores, _ = predict_batch_components(
        model=model,
        x_scaled=x_scaled,
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
        threshold=args.threshold,
        threshold_temperature=args.threshold_temperature,
    )
    budget_fraction = get_batch_budget_fraction(args)
    a_rs = randomized_smoothing_allocate(
        scores=scores.detach(),
        budget_fraction=budget_fraction,
        n_samples=args.n_smoothing_samples,
        noise_std=args.noise_std,
    )

    incremental_objective = torch.sum(a_rs * scores)
    s_policy = a_rs * s1_hat + (1.0 - a_rs) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)

    loss = -incremental_objective / x_scaled.shape[0]
    loss = loss + args.fairness_weight * fairness_penalty
    loss = loss + args.mse_weight * mse_loss

    logs = {
        "incremental_objective_mean": float((incremental_objective / x_scaled.shape[0]).detach().cpu()),
        "mean_rs_allocation": float(a_rs.mean().detach().cpu()),
        "budget_fraction": float(budget_fraction),
        "batch_budget": float(max(1, int(round(budget_fraction * x_scaled.shape[0])))),
    }
    return loss, add_common_logs(logs, loss, scores, s_policy, mse_loss, fairness_penalty)


def pg_loss_for_batch(
    model: OutcomeNetNoSigmoid,
    x_scaled: torch.Tensor,
    x_unscaled: torch.Tensor,
    y_obs: torch.Tensor,
    group_b: torch.Tensor,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
) -> tuple[torch.Tensor, dict[str, float]]:
    y_obs_hat, s0_hat, s1_hat, scores, _ = predict_batch_components(
        model=model,
        x_scaled=x_scaled,
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
        threshold=args.threshold,
        threshold_temperature=args.threshold_temperature,
    )
    budget_fraction = get_batch_budget_fraction(args)
    pg_loss, pg_logs = perturbation_gradient_surrogate(
        scores=scores,
        budget_fraction=budget_fraction,
        n_samples=args.n_perturb_samples,
        noise_std=args.noise_std,
        estimator=args.pg_estimator,
    )

    batch_budget = max(1, int(round(budget_fraction * scores.shape[0])))
    deterministic_allocation = hard_top_b(scores.detach(), budget=batch_budget)
    s_policy = deterministic_allocation * s1_hat + (1.0 - deterministic_allocation) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)

    loss = args.pg_weight * pg_loss
    loss = loss + args.pg_fairness_weight * fairness_penalty
    loss = loss + args.pg_mse_weight * mse_loss

    return loss, add_common_logs(pg_logs, loss, scores, s_policy, mse_loss, fairness_penalty)


def train_or_evaluate_epoch(
    model: OutcomeNetNoSigmoid,
    loader: DataLoader,
    device: torch.device,
    scaler_mean: torch.Tensor,
    scaler_scale: torch.Tensor,
    feature_cols: list[str],
    args: argparse.Namespace,
    loss_fn: LossFn,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    model.train(optimizer is not None)
    logs_accum: dict[str, list[float]] = {}

    context = torch.enable_grad() if optimizer is not None else torch.no_grad()
    with context:
        for x_scaled, x_unscaled, y_obs, group_b in loader:
            x_scaled = x_scaled.to(device)
            x_unscaled = x_unscaled.to(device)
            y_obs = y_obs.to(device)
            group_b = group_b.to(device)

            loss, logs = loss_fn(
                model,
                x_scaled,
                x_unscaled,
                y_obs,
                group_b,
                scaler_mean,
                scaler_scale,
                feature_cols,
                args,
            )

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for key, value in logs.items():
                logs_accum.setdefault(key, []).append(value)

    return {key: float(np.mean(values)) for key, values in logs_accum.items()}


def save_checkpoint(
    model: OutcomeNetNoSigmoid,
    scaler: StandardScaler,
    feature_cols: list[str],
    args: argparse.Namespace,
    best_val_logs: dict[str, float],
    output_path: Path,
    training_method: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "OutcomeNetNoSigmoid",
        "feature_cols": feature_cols,
        "input_dim": len(feature_cols),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "training_method": training_method,
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "args": vars(args),
        "metrics": best_val_logs,
    }
    torch.save(checkpoint, output_path)


def default_output_path(args: argparse.Namespace, variant: str, estimator: str | None = None) -> Path:
    if variant == "dfl":
        return args.output_path
    if variant == "rs":
        return args.rs_output_path
    if variant == "pg":
        if estimator is None:
            return args.pg_output_path
        return args.output_dir / f"pg_{estimator}_model.pt"
    raise ValueError(f"Unknown variant: {variant}")


def default_summary_path(args: argparse.Namespace, variant: str, estimator: str | None = None) -> Path:
    if variant == "dfl":
        return args.summary_path
    if variant == "rs":
        return args.rs_summary_path
    if variant == "pg":
        if estimator is None:
            return args.pg_summary_path
        return args.output_dir / f"pg_{estimator}_training_summary.csv"
    raise ValueError(f"Unknown variant: {variant}")


def configure_variant_args(
    args: argparse.Namespace,
    variant: str,
    estimator: str | None = None,
) -> argparse.Namespace:
    variant_args = argparse.Namespace(**vars(args))
    variant_args.model_variant = variant
    variant_args.pg_estimator = estimator or args.pg_estimator

    if variant == "pg":
        variant_args.mse_weight = args.pg_mse_weight
        variant_args.fairness_weight = args.pg_fairness_weight

    return variant_args


def variant_loss_and_method(variant: str) -> tuple[LossFn, str, str]:
    if variant == "dfl":
        return dfl_loss_for_batch, "decision_focused_soft", "DFL"
    if variant == "rs":
        return rs_loss_for_batch, "randomized_smoothing", "Randomized smoothing DFL"
    if variant == "pg":
        return pg_loss_for_batch, "perturbation_gradient", "Perturbation-gradient DFL"
    raise ValueError(f"Unknown variant: {variant}")


def print_variant_settings(args: argparse.Namespace, variant: str) -> None:
    settings = [
        f"budget_fraction={args.budget_fraction}",
        f"randomize_budget={args.randomize_budget}",
        f"budget_fraction_min={args.budget_fraction_min}",
        f"budget_fraction_max={args.budget_fraction_max}",
        f"threshold={args.threshold}",
        f"threshold_temperature={args.threshold_temperature}",
    ]
    if variant == "dfl":
        settings.extend(
            [
                f"allocation_temperature={args.allocation_temperature}",
                f"mse_weight={args.mse_weight}",
                f"fairness_weight={args.fairness_weight}",
            ]
        )
    elif variant == "rs":
        settings.extend(
            [
                f"n_smoothing_samples={args.n_smoothing_samples}",
                f"noise_std={args.noise_std}",
                f"mse_weight={args.mse_weight}",
                f"fairness_weight={args.fairness_weight}",
            ]
        )
    elif variant == "pg":
        settings.extend(
            [
                f"estimator={args.pg_estimator}",
                f"n_perturb_samples={args.n_perturb_samples}",
                f"noise_std={args.noise_std}",
                f"pg_weight={args.pg_weight}",
                f"mse_weight={args.pg_mse_weight}",
                f"fairness_weight={args.pg_fairness_weight}",
            ]
        )

    _, _, display_name = variant_loss_and_method(variant)
    print(f"{display_name} settings: " + ", ".join(settings))


def train_final_variant(
    args: argparse.Namespace,
    full_df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    loss_fn: LossFn,
    n_epochs: int,
    run_name: str,
) -> tuple[OutcomeNetNoSigmoid, StandardScaler, dict[str, float]]:
    set_seed(args.seed)

    full_loader, scaler = make_loader(
        df=full_df,
        feature_cols=feature_cols,
        batch_size=args.batch_size,
        shuffle=True,
    )
    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    model = OutcomeNetNoSigmoid(
        input_dim=len(feature_cols),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    final_logs: dict[str, float] = {}
    print(f"[{run_name}] Retraining final model on all training data for {n_epochs} epochs.")
    for _ in range(1, n_epochs + 1):
        final_logs = train_or_evaluate_epoch(
            model=model,
            loader=full_loader,
            device=device,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=args,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )

    model.eval()
    return model, scaler, final_logs


def train_one_variant(
    args: argparse.Namespace,
    full_df: pd.DataFrame,
    feature_cols: list[str],
    device: torch.device,
    variant: str,
    output_path: Path,
    summary_path: Path,
    estimator: str | None = None,
) -> None:
    variant_args = configure_variant_args(args, variant=variant, estimator=estimator)
    set_seed(variant_args.seed)

    loss_fn, training_method, display_name = variant_loss_and_method(variant)
    parameter_count_model = OutcomeNetNoSigmoid(
        input_dim=len(feature_cols),
        hidden_dim=variant_args.hidden_dim,
        dropout=variant_args.dropout,
    )

    run_name = f"{variant}:{estimator}" if estimator else variant
    print(f"\nCross-validating {display_name} ({run_name})")
    print(f"Trainable parameters per fold: {count_parameters(parameter_count_model)}")
    print_variant_settings(variant_args, variant)

    kfold = KFold(n_splits=variant_args.n_folds, shuffle=True, random_state=variant_args.seed)
    summary_rows = []
    fold_best_logs = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(full_df), start=1):
        fold_args = argparse.Namespace(**vars(variant_args))
        fold_args.seed = variant_args.seed + fold
        set_seed(fold_args.seed)

        train_df = full_df.iloc[train_idx].reset_index(drop=True)
        val_df = full_df.iloc[val_idx].reset_index(drop=True)
        train_loader, val_loader, scaler = make_loaders(
            train_df=train_df,
            val_df=val_df,
            feature_cols=feature_cols,
            batch_size=fold_args.batch_size,
        )
        scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
        scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

        model = OutcomeNetNoSigmoid(
            input_dim=len(feature_cols),
            hidden_dim=fold_args.hidden_dim,
            dropout=fold_args.dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=fold_args.lr)

        best_val_loss = float("inf")
        best_val_logs: dict[str, float] = {}
        epochs_without_improvement = 0

        for epoch in range(1, fold_args.epochs + 1):
            train_logs = train_or_evaluate_epoch(
                model=model,
                loader=train_loader,
                device=device,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                feature_cols=feature_cols,
                args=fold_args,
                loss_fn=loss_fn,
                optimizer=optimizer,
            )
            val_logs = train_or_evaluate_epoch(
                model=model,
                loader=val_loader,
                device=device,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                feature_cols=feature_cols,
                args=fold_args,
                loss_fn=loss_fn,
            )

            row = {"fold": fold, "epoch": epoch, "model_variant": variant}
            if estimator is not None:
                row["estimator"] = estimator
            row.update({f"train_{k}": v for k, v in train_logs.items()})
            row.update({f"val_{k}": v for k, v in val_logs.items()})
            summary_rows.append(row)

            val_loss = val_logs["loss"]
            if val_loss < best_val_loss - 1e-8:
                best_val_loss = val_loss
                best_val_logs = val_logs.copy()
                best_val_logs["fold"] = float(fold)
                best_val_logs["best_epoch"] = float(epoch)
                best_val_logs["model_variant"] = variant
                if estimator is not None:
                    best_val_logs["pg_estimator"] = estimator
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epoch == 1 or epoch % 10 == 0:
                objective_key = (
                    "perturbed_objective_mean"
                    if variant == "pg"
                    else "incremental_objective_mean"
                )
                objective_label = "val_pg_obj" if variant == "pg" else "val_incr_obj"
                print(
                    f"[{run_name} fold {fold}/{variant_args.n_folds}] Epoch {epoch:03d}: "
                    f"train_loss={train_logs['loss']:.5f}, "
                    f"val_loss={val_logs['loss']:.5f}, "
                    f"{objective_label}={val_logs[objective_key]:.5f}, "
                    f"val_policy_success={val_logs['policy_success_mean']:.5f}, "
                    f"val_mse={val_logs['mse_loss']:.5f}"
                )

            if epochs_without_improvement >= fold_args.patience:
                print(f"[{run_name} fold {fold}/{variant_args.n_folds}] Early stopping at epoch {epoch}.")
                break

        fold_best_logs.append(best_val_logs)
        print(
            f"[{run_name} fold {fold}/{variant_args.n_folds}] "
            f"best_epoch={best_val_logs['best_epoch']:.0f}, "
            f"val_loss={best_val_logs['loss']:.5f}, "
            f"val_mse={best_val_logs['mse_loss']:.5f}"
        )

    summary_df = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)

    mean_best_epoch = int(round(float(np.mean([row["best_epoch"] for row in fold_best_logs]))))
    final_epochs = max(1, mean_best_epoch)
    final_model, final_scaler, final_train_logs = train_final_variant(
        args=variant_args,
        full_df=full_df,
        feature_cols=feature_cols,
        device=device,
        loss_fn=loss_fn,
        n_epochs=final_epochs,
        run_name=run_name,
    )

    checkpoint_logs = {
        "cv_mean_best_epoch": float(np.mean([row["best_epoch"] for row in fold_best_logs])),
        "cv_mean_val_loss": float(np.mean([row["loss"] for row in fold_best_logs])),
        "cv_mean_val_mse_loss": float(np.mean([row["mse_loss"] for row in fold_best_logs])),
        "n_folds": float(variant_args.n_folds),
        "model_variant": variant,
    }
    if estimator is not None:
        checkpoint_logs["pg_estimator"] = estimator
    checkpoint_logs["final_epochs"] = float(final_epochs)
    checkpoint_logs.update({f"full_train_{k}": v for k, v in final_train_logs.items()})

    save_checkpoint(
        model=final_model,
        scaler=final_scaler,
        feature_cols=feature_cols,
        args=variant_args,
        best_val_logs=checkpoint_logs,
        output_path=output_path,
        training_method=training_method,
    )

    if variant == "pg" and estimator == "score_function":
        save_checkpoint(
            model=final_model,
            scaler=final_scaler,
            feature_cols=feature_cols,
            args=variant_args,
            best_val_logs=checkpoint_logs,
            output_path=args.pg_output_path,
            training_method=training_method,
        )

    print(f"[{run_name}] Cross-validation best logs:")
    print(pd.DataFrame(fold_best_logs))
    print(f"[{run_name}] Final full-data training logs:")
    print(final_train_logs)
    print(f"[{run_name}] Saved final full-data model to {output_path}")
    print(f"[{run_name}] Saved training summary to {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFL, RS, and PG model variants.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs")
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_model.pt")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_training_summary.csv")
    parser.add_argument("--rs-output-path", type=Path, default=PROJECT_ROOT / "outputs" / "rs_model.pt")
    parser.add_argument("--rs-summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "rs_training_summary.csv")
    parser.add_argument("--pg-output-path", type=Path, default=PROJECT_ROOT / "outputs" / "pg_model.pt")
    parser.add_argument("--pg-summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "pg_training_summary.csv")
    parser.add_argument("--models", nargs="+", choices=MODEL_VARIANTS, default=MODEL_VARIANTS)
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
    parser.add_argument("--randomize-budget", action="store_true", default=True)
    parser.add_argument("--no-randomize-budget", dest="randomize_budget", action="store_false")
    parser.add_argument("--budget-fraction-min", type=float, default=0.001)
    parser.add_argument("--budget-fraction-max", type=float, default=0.10)
    parser.add_argument("--n-smoothing-samples", type=int, default=DEFAULT_N_SMOOTHING_SAMPLES)
    parser.add_argument("--n-perturb-samples", type=int, default=DEFAULT_N_PERTURB_SAMPLES)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--pg-weight", type=float, default=DEFAULT_PG_WEIGHT)
    parser.add_argument("--pg-mse-weight", type=float, default=DEFAULT_PG_MSE_WEIGHT)
    parser.add_argument("--pg-fairness-weight", type=float, default=DEFAULT_PG_FAIRNESS_WEIGHT)
    parser.add_argument(
        "--pg-estimator",
        choices=PG_ESTIMATORS,
        default=PG_ESTIMATORS[0],
        help="Perturbation-gradient estimator used when --single-estimator-only is set.",
    )
    parser.add_argument(
        "--single-estimator-only",
        action="store_true",
        help="Train only --pg-estimator for PG instead of all four PG estimators.",
    )
    parser.add_argument("--mse-weight", type=float, default=DEFAULT_MSE_WEIGHT)
    parser.add_argument("--fairness-weight", type=float, default=DEFAULT_FAIRNESS_WEIGHT)
    parser.add_argument("--n-folds", type=int, default=DEFAULT_N_FOLDS)
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

    for variant in args.models:
        if variant == "pg":
            estimators = [args.pg_estimator] if args.single_estimator_only else PG_ESTIMATORS
            for estimator in estimators:
                train_one_variant(
                    args=args,
                    full_df=df,
                    feature_cols=feature_cols,
                    device=device,
                    variant=variant,
                    estimator=estimator,
                    output_path=default_output_path(args, variant, estimator),
                    summary_path=default_summary_path(args, variant, estimator),
                )
        else:
            train_one_variant(
                args=args,
                full_df=df,
                feature_cols=feature_cols,
                device=device,
                variant=variant,
                output_path=default_output_path(args, variant),
                summary_path=default_summary_path(args, variant),
            )


if __name__ == "__main__":
    main()
