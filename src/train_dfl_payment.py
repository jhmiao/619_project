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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models import OutcomeNetNoSigmoid, count_parameters  # noqa: E402


SEED = 619
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
DEFAULT_FAIRNESS_WEIGHT = 5.00
DEFAULT_N_SMOOTHING_SAMPLES = 10
DEFAULT_N_PERTURB_SAMPLES = 10
DEFAULT_NOISE_STD = 0.10
DEFAULT_PG_WEIGHT = 1.00
DEFAULT_PG_MSE_WEIGHT = 0.50
DEFAULT_PG_FAIRNESS_WEIGHT = DEFAULT_FAIRNESS_WEIGHT

DEFAULT_RANDOMIZE_BUDGET = True
DEFAULT_BUDGET_FRACTION_MIN = 0.001
DEFAULT_BUDGET_FRACTION_MAX = 0.10
PG_ESTIMATORS = ["score_function", "forward", "backward", "central"]
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

def compute_expected_cost(
    y1_hat: torch.Tensor,
    payment_threshold: float = 0.60,
    alpha: float = 1.0,
) -> torch.Tensor:
    """
    Compute expected payment cost under action.

    Cost model:
        c_i = alpha * y1_hat_i * 1{y1_hat_i >= payment_threshold}

    Args:
        y1_hat:
            Predicted outcome under action.
            Shape: (batch_size,)

        payment_threshold:
            Minimum threshold required to trigger payment.

        alpha:
            Scaling factor converting wear/adherence into dollars.

    Returns:
        expected_cost:
            Tensor of shape (batch_size,)
    """
    payment_indicator = (y1_hat >= payment_threshold).float()

    expected_cost = alpha * y1_hat * payment_indicator

    return expected_cost

def compute_expected_cost_smooth(
    y1_hat: torch.Tensor,
    payment_threshold: float = 0.60,
    alpha: float = 1.0,
    temperature: float = 0.05,
) -> torch.Tensor:
    """
    Smooth differentiable expected payment cost.

    Uses:
        sigmoid((y1_hat - T) / temperature)
    instead of:
        1{y1_hat >= T}
    """
    payment_prob = torch.sigmoid(
        (y1_hat - payment_threshold) / temperature
    )

    expected_cost = alpha * y1_hat * payment_prob

    return expected_cost

def soft_knapsack_allocate(
    scores: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
    temperature: float = 0.10,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Differentiable soft knapsack allocation.

    Idea:
        Allocate more probability mass to users with high
            score / cost
        while approximately respecting:
            sum_i a_i * cost_i <= budget

    Args:
        scores:
            Utility gains.
            Shape: (batch_size,)

        costs:
            Expected payment costs.
            Shape: (batch_size,)

        budget:
            Total payment budget.

        temperature:
            Softmax temperature.

    Returns:
        a_soft:
            Soft allocation vector in [0,1].
            Shape: (batch_size,)
    """
    # Avoid division by zero.
    safe_costs = torch.clamp(costs, min=eps)

    # Value-per-cost ratio.
    density = scores / safe_costs

    # Soft ranking over density.
    weights = torch.softmax(density / temperature, dim=0)

    # Scale allocations to approximately satisfy budget.
    denom = torch.sum(weights * safe_costs)

    scale = budget / torch.clamp(denom, min=eps)

    a_soft = scale * weights

    # Optional clipping for stability.
    a_soft = torch.clamp(a_soft, min=0.0, max=1.0)

    return a_soft


def hard_knapsack_greedy(
    scores: np.ndarray,
    costs: np.ndarray,
    budget: float,
    require_positive_score: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Greedy knapsack allocation.

    Select users by descending:
        score / cost

    subject to:
        sum_i cost_i * A_i <= budget

    Args:
        scores:
            Utility gains.
            Shape: (n_users,)

        costs:
            Expected payment costs.
            Shape: (n_users,)

        budget:
            Total payment budget.

    Returns:
        allocation:
            Binary allocation vector.
    """
    scores = np.asarray(scores, dtype=float)
    costs = np.asarray(costs, dtype=float)

    n = len(scores)
    allocation = np.zeros(n, dtype=int)

    # Optional filtering.
    candidate_idx = np.arange(n)

    if require_positive_score:
        candidate_idx = candidate_idx[scores[candidate_idx] > 0]

    # Handle zero-cost users separately.
    zero_cost_mask = costs[candidate_idx] <= eps
    zero_cost_idx = candidate_idx[zero_cost_mask]
    positive_cost_idx = candidate_idx[~zero_cost_mask]

    remaining_budget = float(budget)

    # Always take beneficial zero-cost users.
    for i in zero_cost_idx:
        allocation[i] = 1

    # Greedy density ordering.
    density = scores[positive_cost_idx] / costs[positive_cost_idx]

    order = positive_cost_idx[np.argsort(-density)]

    for i in order:
        if costs[i] <= remaining_budget:
            allocation[i] = 1
            remaining_budget -= costs[i]

    return allocation


# ==== Additional Helper Functions ====
import torch

def hard_knapsack_greedy_torch(
    scores: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
    require_positive_score: bool = True,
) -> torch.Tensor:
    """Torch wrapper around hard_knapsack_greedy.

    The greedy solve is non-differentiable, so scores and costs are detached.
    The returned allocation is a tensor on the same device as scores.
    """
    allocation_np = hard_knapsack_greedy(
        scores=scores.detach().cpu().numpy(),
        costs=costs.detach().cpu().numpy(),
        budget=float(budget),
        require_positive_score=require_positive_score,
    )
    return torch.tensor(allocation_np, dtype=torch.float32, device=scores.device)


def randomized_smoothing_knapsack_allocate(
    scores: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
    n_samples: int,
    noise_std: float,
    require_positive_score: bool = True,
) -> torch.Tensor:
    """Average hard knapsack allocations over noisy score perturbations."""
    allocations = []
    for _ in range(n_samples):
        noise = torch.randn_like(scores) * noise_std
        noisy_scores = scores.detach() + noise
        allocation = hard_knapsack_greedy_torch(
            scores=noisy_scores,
            costs=costs,
            budget=budget,
            require_positive_score=require_positive_score,
        )
        allocations.append(allocation)

    return torch.stack(allocations, dim=0).mean(dim=0)


def objective_for_allocation(allocation: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
    """Incremental objective for a fixed allocation."""
    return torch.sum(allocation * scores)


def perturbation_gradient_knapsack_surrogate(
    scores: torch.Tensor,
    costs: torch.Tensor,
    budget: float,
    n_samples: int,
    noise_std: float,
    estimator: str,
    require_positive_score: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Perturbation-gradient surrogate using hard knapsack allocations.

    Supports the same estimators as train_dfl.py:
        score_function, forward, backward, central.

    Let t be the score vector and z*(t) be the hard greedy knapsack allocation.
    The finite-difference estimators approximate a gradient with respect to t.
    We then backpropagate through scores using:
        - stop_gradient(g)^T t.
    """
    if estimator not in PG_ESTIMATORS:
        raise ValueError(f"Unknown estimator: {estimator}. Use one of {PG_ESTIMATORS}.")

    batch_size = scores.shape[0]
    h = max(noise_std, 1e-8)
    scores_detached = scores.detach()

    allocation_base = hard_knapsack_greedy_torch(
        scores=scores_detached,
        costs=costs,
        budget=budget,
        require_positive_score=require_positive_score,
    )
    objective_base = objective_for_allocation(allocation_base, scores_detached)

    eps_list = []
    objective_list = []
    grad_terms = []
    allocation_list = []

    for _ in range(n_samples):
        eps = torch.randn_like(scores_detached)
        scores_plus = scores_detached + h * eps
        scores_minus = scores_detached - h * eps

        allocation_plus = hard_knapsack_greedy_torch(
            scores=scores_plus,
            costs=costs,
            budget=budget,
            require_positive_score=require_positive_score,
        )
        allocation_minus = hard_knapsack_greedy_torch(
            scores=scores_minus,
            costs=costs,
            budget=budget,
            require_positive_score=require_positive_score,
        )

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
    avg_objective = objective_for_allocation(avg_allocation, scores_detached)
    avg_cost = torch.sum(avg_allocation * costs.detach())

    logs = {
        "pg_surrogate_loss": float(surrogate_loss.detach().cpu()),
        "perturbed_objective_mean": float((objectives.mean() / batch_size).detach().cpu()),
        "perturbed_objective_std": float((objectives.std(unbiased=False) / batch_size).detach().cpu()),
        "base_objective_mean": float((objective_base / batch_size).detach().cpu()),
        "avg_allocation_mean": float(avg_allocation.mean().detach().cpu()),
        "avg_allocation_objective_mean": float((avg_objective / batch_size).detach().cpu()),
        "avg_allocation_cost": float(avg_cost.detach().cpu()),
        "grad_score_norm": float(torch.norm(grad_scores).detach().cpu()),
    }
    return surrogate_loss, logs





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


def dfl_soft_payment_loss_for_batch(
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
    """Compute DFL-soft loss using a soft knapsack/payment-budget layer.

    The learned score is:
        d_i = S1_hat_i - S0_hat_i

    The expected payment cost is:
        c_i = alpha * y1_hat_i * sigmoid((y1_hat_i - T) / tau)

    The soft allocation approximately solves:
        max sum_i a_i d_i
        s.t. sum_i a_i c_i <= B_batch

    where B_batch is sampled as:
        rho ~ Uniform(budget_fraction_min, budget_fraction_max)
        B_batch = rho * sum_i c_i

    Using a fraction of total predicted available cost makes the budget scale
    naturally with mini-batch size and predicted costs.
    """
    y_obs_hat, s0_hat, s1_hat, scores, y1_minus_y0 = predict_batch_components(
        model=model,
        x_scaled=x_scaled,
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
        threshold=args.threshold,
        threshold_temperature=args.threshold_temperature,
    )

    # Recompute y1_hat directly because predicted cost depends on the action outcome level.
    _, x1_scaled = make_scaled_counterfactuals(
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
    )
    y1_hat = torch.sigmoid(model(x1_scaled))

    costs = compute_expected_cost_smooth(
        y1_hat=y1_hat,
        payment_threshold=args.payment_threshold,
        alpha=args.alpha,
        temperature=args.threshold_temperature,
    )

    budget_fraction = get_batch_budget_fraction(args)
    total_predicted_cost = torch.sum(costs.detach())
    batch_budget = budget_fraction * total_predicted_cost

    # If all predicted costs are essentially zero, skip the decision term and rely on MSE.
    if float(total_predicted_cost.detach().cpu()) <= 1e-8:
        a_soft = torch.zeros_like(scores)
        incremental_objective = torch.tensor(0.0, device=scores.device)
    else:
        a_soft = soft_knapsack_allocate(
            scores=scores,
            costs=costs,
            budget=float(batch_budget.detach().cpu()),
            temperature=args.allocation_temperature,
        )
        incremental_objective = torch.sum(a_soft * scores)

    s_policy = a_soft * s1_hat + (1.0 - a_soft) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)

    loss = -incremental_objective / x_scaled.shape[0]
    loss = loss + args.fairness_weight * fairness_penalty
    loss = loss + args.mse_weight * mse_loss

    realized_soft_cost = torch.sum(a_soft * costs)

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
        "mean_expected_cost": float(costs.mean().detach().cpu()),
        "total_predicted_cost": float(total_predicted_cost.detach().cpu()),
        "batch_budget": float(batch_budget.detach().cpu()),
        "realized_soft_cost": float(realized_soft_cost.detach().cpu()),
        "budget_fraction": float(budget_fraction),
    }
    return loss, logs


# ==== Additional Loss Functions ====

def rs_payment_loss_for_batch(
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
    """Randomized-smoothing DFL loss with hard greedy knapsack allocations."""
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

    _, x1_scaled = make_scaled_counterfactuals(
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
    )
    y1_hat = torch.sigmoid(model(x1_scaled))
    costs = compute_expected_cost_smooth(
        y1_hat=y1_hat,
        payment_threshold=args.payment_threshold,
        alpha=args.alpha,
        temperature=args.threshold_temperature,
    )

    budget_fraction = get_batch_budget_fraction(args)
    total_predicted_cost = torch.sum(costs.detach())
    batch_budget = budget_fraction * total_predicted_cost

    if float(total_predicted_cost.detach().cpu()) <= 1e-8:
        a_rs = torch.zeros_like(scores)
        incremental_objective = torch.tensor(0.0, device=scores.device)
    else:
        a_rs = randomized_smoothing_knapsack_allocate(
            scores=scores,
            costs=costs,
            budget=float(batch_budget.detach().cpu()),
            n_samples=args.n_smoothing_samples,
            noise_std=args.noise_std,
            require_positive_score=args.require_positive_score,
        )
        incremental_objective = torch.sum(a_rs * scores)

    s_policy = a_rs * s1_hat + (1.0 - a_rs) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)
    loss = -incremental_objective / x_scaled.shape[0]
    loss = loss + args.fairness_weight * fairness_penalty
    loss = loss + args.mse_weight * mse_loss

    realized_cost = torch.sum(a_rs * costs)
    logs = {
        "loss": float(loss.detach().cpu()),
        "incremental_objective_mean": float((incremental_objective / x_scaled.shape[0]).detach().cpu()),
        "policy_success_mean": float(s_policy.mean().detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "fairness_penalty": float(fairness_penalty.detach().cpu()),
        "mean_rs_allocation": float(a_rs.mean().detach().cpu()),
        "mean_score": float(scores.mean().detach().cpu()),
        "max_score": float(scores.max().detach().cpu()),
        "min_score": float(scores.min().detach().cpu()),
        "mean_expected_cost": float(costs.mean().detach().cpu()),
        "total_predicted_cost": float(total_predicted_cost.detach().cpu()),
        "batch_budget": float(batch_budget.detach().cpu()),
        "realized_cost": float(realized_cost.detach().cpu()),
        "budget_fraction": float(budget_fraction),
    }
    return loss, logs


def pg_payment_loss_for_batch(
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
    """Perturbation-gradient DFL loss with hard greedy knapsack allocations."""
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

    _, x1_scaled = make_scaled_counterfactuals(
        x_unscaled=x_unscaled,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        feature_cols=feature_cols,
    )
    y1_hat = torch.sigmoid(model(x1_scaled))
    costs = compute_expected_cost_smooth(
        y1_hat=y1_hat,
        payment_threshold=args.payment_threshold,
        alpha=args.alpha,
        temperature=args.threshold_temperature,
    )

    budget_fraction = get_batch_budget_fraction(args)
    total_predicted_cost = torch.sum(costs.detach())
    batch_budget = budget_fraction * total_predicted_cost

    if float(total_predicted_cost.detach().cpu()) <= 1e-8:
        pg_loss = torch.tensor(0.0, device=scores.device)
        deterministic_allocation = torch.zeros_like(scores)
        pg_logs = {
            "pg_surrogate_loss": 0.0,
            "perturbed_objective_mean": 0.0,
            "perturbed_objective_std": 0.0,
            "base_objective_mean": 0.0,
            "avg_allocation_mean": 0.0,
            "avg_allocation_objective_mean": 0.0,
            "avg_allocation_cost": 0.0,
            "grad_score_norm": 0.0,
        }
    else:
        pg_loss, pg_logs = perturbation_gradient_knapsack_surrogate(
            scores=scores,
            costs=costs,
            budget=float(batch_budget.detach().cpu()),
            n_samples=args.n_perturb_samples,
            noise_std=args.noise_std,
            estimator=args.pg_estimator,
            require_positive_score=args.require_positive_score,
        )
        deterministic_allocation = hard_knapsack_greedy_torch(
            scores=scores.detach(),
            costs=costs,
            budget=float(batch_budget.detach().cpu()),
            require_positive_score=args.require_positive_score,
        )

    s_policy = deterministic_allocation * s1_hat + (1.0 - deterministic_allocation) * s0_hat
    fairness_penalty = group_rate_fairness_penalty(s_policy=s_policy, group_b=group_b)
    mse_loss = nn.functional.mse_loss(y_obs_hat, y_obs)
    loss = args.pg_weight * pg_loss
    loss = loss + args.fairness_weight * fairness_penalty
    loss = loss + args.mse_weight * mse_loss

    deterministic_cost = torch.sum(deterministic_allocation * costs)
    logs = {
        "loss": float(loss.detach().cpu()),
        "policy_success_mean": float(s_policy.mean().detach().cpu()),
        "mse_loss": float(mse_loss.detach().cpu()),
        "fairness_penalty": float(fairness_penalty.detach().cpu()),
        "mean_score": float(scores.mean().detach().cpu()),
        "max_score": float(scores.max().detach().cpu()),
        "min_score": float(scores.min().detach().cpu()),
        "mean_expected_cost": float(costs.mean().detach().cpu()),
        "total_predicted_cost": float(total_predicted_cost.detach().cpu()),
        "batch_budget": float(batch_budget.detach().cpu()),
        "deterministic_cost": float(deterministic_cost.detach().cpu()),
        "budget_fraction": float(budget_fraction),
        **pg_logs,
    }
    return loss, logs


# ==== Loss Function Dispatcher ====

def get_loss_function(model_variant: str) -> LossFn:
    if model_variant == "dfl":
        return dfl_soft_payment_loss_for_batch
    if model_variant == "rs":
        return rs_payment_loss_for_batch
    if model_variant == "pg":
        return pg_payment_loss_for_batch
    raise ValueError(f"Unknown model_variant: {model_variant}")


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
    loss_fn = get_loss_function(args.model_variant)

    for x_scaled, x_unscaled, y_obs, group_b in loader:
        x_scaled = x_scaled.to(device)
        x_unscaled = x_unscaled.to(device)
        y_obs = y_obs.to(device)
        group_b = group_b.to(device)

        loss, logs = loss_fn(
            model=model,
            x_scaled=x_scaled,
            x_unscaled=x_unscaled,
            y_obs=y_obs,
            group_b=group_b,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=args,
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
    loss_fn = get_loss_function(args.model_variant)
    with torch.no_grad():
        for x_scaled, x_unscaled, y_obs, group_b in loader:
            x_scaled = x_scaled.to(device)
            x_unscaled = x_unscaled.to(device)
            y_obs = y_obs.to(device)
            group_b = group_b.to(device)

            _, logs = loss_fn(
                model=model,
                x_scaled=x_scaled,
                x_unscaled=x_unscaled,
                y_obs=y_obs,
                group_b=group_b,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                feature_cols=feature_cols,
                args=args,
            )

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
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_class": "OutcomeNetNoSigmoid",
        "feature_cols": feature_cols,
        "input_dim": len(feature_cols),
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "training_method": f"{args.model_variant}_payment",
        "scaler_mean": scaler.mean_,
        "scaler_scale": scaler.scale_,
        "args": vars(args),
        "metrics": best_val_logs,
    }
    torch.save(checkpoint, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DFL-soft model with expected payment budget.")
    parser.add_argument("--data-path", type=Path, default=PROJECT_ROOT / "data" / "train.csv")
    parser.add_argument("--output-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_payment_model.pt")
    parser.add_argument("--summary-path", type=Path, default=PROJECT_ROOT / "outputs" / "dfl_payment_training_summary.csv")
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
    parser.add_argument("--randomize-budget", action="store_true", default=DEFAULT_RANDOMIZE_BUDGET)
    parser.add_argument("--no-randomize-budget", dest="randomize_budget", action="store_false")
    parser.add_argument("--budget-fraction-min", type=float, default=DEFAULT_BUDGET_FRACTION_MIN)
    parser.add_argument("--budget-fraction-max", type=float, default=DEFAULT_BUDGET_FRACTION_MAX)
    parser.add_argument("--mse-weight", type=float, default=DEFAULT_MSE_WEIGHT)
    parser.add_argument("--fairness-weight", type=float, default=DEFAULT_FAIRNESS_WEIGHT)
    parser.add_argument("--payment-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=SEED)
    # Additional arguments for variants
    parser.add_argument("--model-variant", choices=MODEL_VARIANTS, default="dfl")
    parser.add_argument("--train-all-variants", action="store_true", default=False)
    parser.add_argument("--pg-estimator", choices=PG_ESTIMATORS, default="central")
    parser.add_argument("--n-smoothing-samples", type=int, default=DEFAULT_N_SMOOTHING_SAMPLES)
    parser.add_argument("--n-perturb-samples", type=int, default=DEFAULT_N_PERTURB_SAMPLES)
    parser.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    parser.add_argument("--pg-weight", type=float, default=DEFAULT_PG_WEIGHT)
    parser.add_argument("--require-positive-score", action="store_true", default=True)
    parser.add_argument("--allow-nonpositive-score", dest="require_positive_score", action="store_false")
    return parser.parse_args()



# ==== Main Function With Multi-Variant Support ====

def train_one_variant(args: argparse.Namespace, model_variant: str, pg_estimator: str | None = None) -> None:
    variant_args = argparse.Namespace(**vars(args))
    variant_args.model_variant = model_variant
    if pg_estimator is not None:
        variant_args.pg_estimator = pg_estimator

    set_seed(variant_args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label = model_variant if model_variant != "pg" else f"pg:{variant_args.pg_estimator}"
    print(f"Using device: {device}")
    print(f"Training payment-budget model: {label}")

    df, feature_cols = load_data(variant_args.data_path)
    print(f"Loaded {variant_args.data_path}: {df.shape}")
    print(f"Input dim: {len(feature_cols)}")

    train_df, val_df = train_test_split(
        df,
        test_size=variant_args.val_frac,
        random_state=variant_args.seed,
        shuffle=True,
        stratify=df["group"],
    )
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_loader, val_loader, scaler = make_loaders(
        train_df=train_df,
        val_df=val_df,
        feature_cols=feature_cols,
        batch_size=variant_args.batch_size,
    )

    scaler_mean = torch.tensor(scaler.mean_, dtype=torch.float32, device=device)
    scaler_scale = torch.tensor(scaler.scale_, dtype=torch.float32, device=device)

    model = OutcomeNetNoSigmoid(
        input_dim=len(feature_cols),
        hidden_dim=variant_args.hidden_dim,
        dropout=variant_args.dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=variant_args.lr)

    print(f"Trainable parameters: {count_parameters(model)}")
    print(
        "Payment-budget settings: "
        f"variant={variant_args.model_variant}, "
        f"pg_estimator={variant_args.pg_estimator}, "
        f"randomize_budget={variant_args.randomize_budget}, "
        f"budget_fraction={variant_args.budget_fraction}, "
        f"budget_fraction_min={variant_args.budget_fraction_min}, "
        f"budget_fraction_max={variant_args.budget_fraction_max}, "
        f"threshold={variant_args.threshold}, "
        f"payment_threshold={variant_args.payment_threshold}, "
        f"alpha={variant_args.alpha}, "
        f"allocation_temperature={variant_args.allocation_temperature}, "
        f"n_smoothing_samples={variant_args.n_smoothing_samples}, "
        f"n_perturb_samples={variant_args.n_perturb_samples}, "
        f"noise_std={variant_args.noise_std}, "
        f"mse_weight={variant_args.mse_weight}, "
        f"fairness_weight={variant_args.fairness_weight}"
    )

    best_val_loss = float("inf")
    best_state = None
    best_val_logs: dict[str, float] = {}
    epochs_without_improvement = 0
    summary_rows = []

    for epoch in range(1, variant_args.epochs + 1):
        train_logs = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=variant_args,
        )
        val_logs = evaluate_loss(
            model=model,
            loader=val_loader,
            device=device,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            feature_cols=feature_cols,
            args=variant_args,
        )

        row = {"epoch": epoch, "model_variant": variant_args.model_variant, "pg_estimator": variant_args.pg_estimator}
        row.update({f"train_{k}": v for k, v in train_logs.items()})
        row.update({f"val_{k}": v for k, v in val_logs.items()})
        summary_rows.append(row)

        val_loss = val_logs["loss"]
        if val_loss < best_val_loss - 1e-8:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_val_logs = val_logs.copy()
            best_val_logs["best_epoch"] = float(epoch)
            best_val_logs["model_variant"] = variant_args.model_variant
            best_val_logs["pg_estimator"] = variant_args.pg_estimator
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            val_obj = val_logs.get("incremental_objective_mean", val_logs.get("perturbed_objective_mean", 0.0))
            val_cost = val_logs.get("realized_soft_cost", val_logs.get("realized_cost", val_logs.get("deterministic_cost", 0.0)))
            print(
                f"[{label}] Epoch {epoch:03d}: "
                f"train_loss={train_logs['loss']:.5f}, "
                f"val_loss={val_logs['loss']:.5f}, "
                f"val_obj={val_obj:.5f}, "
                f"val_policy_success={val_logs['policy_success_mean']:.5f}, "
                f"val_mse={val_logs['mse_loss']:.5f}, "
                f"val_budget={val_logs['batch_budget']:.5f}, "
                f"val_cost={val_cost:.5f}"
            )

        if epochs_without_improvement >= variant_args.patience:
            print(f"[{label}] Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    suffix = variant_args.model_variant if variant_args.model_variant != "pg" else f"pg_{variant_args.pg_estimator}"
    output_path = variant_args.output_path.parent / f"payment_{suffix}_model.pt"
    summary_path = variant_args.summary_path.parent / f"payment_{suffix}_training_summary.csv"

    summary_df = pd.DataFrame(summary_rows)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False)
    save_checkpoint(
        model=model,
        scaler=scaler,
        feature_cols=feature_cols,
        args=variant_args,
        best_val_logs=best_val_logs,
        output_path=output_path,
    )

    if variant_args.model_variant == "dfl":
        save_checkpoint(
            model=model,
            scaler=scaler,
            feature_cols=feature_cols,
            args=variant_args,
            best_val_logs=best_val_logs,
            output_path=variant_args.output_path,
        )

    print(f"[{label}] Best validation logs:")
    print(best_val_logs)
    print(f"[{label}] Saved model to {output_path}")
    print(f"[{label}] Saved training summary to {summary_path}")


def main() -> None:
    args = parse_args()

    if args.train_all_variants:
        jobs: list[tuple[str, str | None]] = [("dfl", None), ("rs", None)]
        jobs.extend(("pg", estimator) for estimator in ["forward", "backward", "central"])
    else:
        jobs = [(args.model_variant, args.pg_estimator if args.model_variant == "pg" else None)]

    for model_variant, pg_estimator in jobs:
        train_one_variant(args=args, model_variant=model_variant, pg_estimator=pg_estimator)


if __name__ == "__main__":
    main()