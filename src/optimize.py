

"""
Optimization utilities for incentive allocation.

Minimal first version: lambda_y = 0, so the optimizer selects the top-B users
by predicted treatment benefit:

    score_i = u(y1_hat_i) - u(y0_hat_i)

By default, u(y) = 1{y > threshold}, matching the step utility setup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gurobipy as gp
from gurobipy import GRB


def step_utility(y: np.ndarray, threshold: float = 0.60) -> np.ndarray:
    """Return 1{y > threshold} as a float array."""
    return (np.asarray(y) > threshold).astype(float)


def compute_scores(
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    utility: str = "step",
    threshold: float = 0.60,
) -> np.ndarray:
    """Compute predicted treatment benefit for each user.

    Args:
        y0_hat: predicted outcome under no action.
        y1_hat: predicted outcome under action.
        utility: "step" or "linear".
        threshold: threshold used when utility="step".

    Returns:
        score_i = u(y1_hat_i) - u(y0_hat_i)
    """
    y0_hat = np.asarray(y0_hat, dtype=float)
    y1_hat = np.asarray(y1_hat, dtype=float)

    if y0_hat.shape != y1_hat.shape:
        raise ValueError(f"Shape mismatch: y0_hat {y0_hat.shape}, y1_hat {y1_hat.shape}")

    if utility == "step":
        u0 = step_utility(y0_hat, threshold=threshold)
        u1 = step_utility(y1_hat, threshold=threshold)
    elif utility == "linear":
        u0 = y0_hat
        u1 = y1_hat
    else:
        raise ValueError(f"Unknown utility: {utility}. Use 'step' or 'linear'.")

    return u1 - u0

def solve_fair_allocation_action(
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    group: np.ndarray,
    budget: int,
    fairness_weight: float,
    utility: str = "step",
    threshold: float = 0.60,
    time_limit: float = 60.0,
    require_positive_score: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Solve fairness-aware incentive allocation using Gurobi MIQP.

    Objective:
        max_A sum_i A_i d_i
              - fairness_weight * sum_g (r_g(A) - r_overall(A))^2

    where:
        d_i = u(y1_hat_i) - u(y0_hat_i)

        s_i(A_i) = A_i u(y1_hat_i) + (1 - A_i) u(y0_hat_i)

        r_g(A) = mean_{i in g} s_i(A_i)
        r_overall(A) = mean_i s_i(A_i)

    This mirrors the DFL training fairness penalty:
        sum_g (group_rate - overall_rate)^2.

    Args:
        y0_hat: predicted outcome under no action.
        y1_hat: predicted outcome under action.
        group: group labels, e.g. "A"/"B" or 0/1.
        budget: maximum number of selected users.
        fairness_weight: lambda_f in the objective.
        utility: "step" or "linear".
        threshold: threshold used when utility="step".
        time_limit: Gurobi time limit in seconds. If optimality is not proven,
            the best feasible incumbent found by this time is returned.
        require_positive_score: if True, force A_i = 0 whenever d_i <= 0.
            Usually False for the fairness-aware problem because fairness may
            justify selecting some lower-score users.
        verbose: if True, print Gurobi output.

    Returns:
        allocation: binary numpy array where allocation[i] = 1 if user i is selected.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if fairness_weight < 0:
        raise ValueError("fairness_weight must be nonnegative")

    y0_hat = np.asarray(y0_hat, dtype=float)
    y1_hat = np.asarray(y1_hat, dtype=float)
    group = np.asarray(group)

    if y0_hat.shape != y1_hat.shape:
        raise ValueError(f"Shape mismatch: y0_hat {y0_hat.shape}, y1_hat {y1_hat.shape}")
    if len(group) != len(y0_hat):
        raise ValueError(f"Length mismatch: group has length {len(group)}, y has length {len(y0_hat)}")

    n = len(y0_hat)
    allocation = np.zeros(n, dtype=int)
    if n == 0 or budget == 0:
        return allocation

    if utility == "step":
        s0 = step_utility(y0_hat, threshold=threshold)
        s1 = step_utility(y1_hat, threshold=threshold)
    elif utility == "linear":
        s0 = y0_hat.astype(float)
        s1 = y1_hat.astype(float)
    else:
        raise ValueError(f"Unknown utility: {utility}. Use 'step' or 'linear'.")

    d = s1 - s0
    unique_groups = list(pd.unique(group))

    model = gp.Model("fair_incentive_allocation")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 1 if verbose else 0

    A = model.addVars(n, vtype=GRB.BINARY, name="A")
    r_overall = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r_overall")
    r_group = model.addVars(len(unique_groups), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r_group")

    model.addConstr(gp.quicksum(A[i] for i in range(n)) <= min(budget, n), name="budget")

    if require_positive_score:
        for i in range(n):
            if d[i] <= 0:
                model.addConstr(A[i] == 0, name=f"nonpositive_score_{i}")

    # s_i(A_i) = s0_i + A_i * (s1_i - s0_i) = s0_i + A_i * d_i
    model.addConstr(
        r_overall == (1.0 / n) * gp.quicksum(s0[i] + d[i] * A[i] for i in range(n)),
        name="overall_rate",
    )

    for g_idx, g_value in enumerate(unique_groups):
        idx = np.where(group == g_value)[0]
        if len(idx) == 0:
            continue
        model.addConstr(
            r_group[g_idx]
            == (1.0 / len(idx)) * gp.quicksum(s0[i] + d[i] * A[i] for i in idx),
            name=f"group_rate_{g_value}",
        )

    gain_expr = gp.quicksum(float(d[i]) * A[i] for i in range(n))
    fairness_penalty = gp.quicksum(
        (r_group[g_idx] - r_overall) * (r_group[g_idx] - r_overall)
        for g_idx in range(len(unique_groups))
    )

    model.setObjective(gain_expr - float(fairness_weight) * fairness_penalty, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        # Fall back to top-B allocation if Gurobi fails to find any feasible solution.
        return solve_top_b_allocation(
            y0_hat=y0_hat,
            y1_hat=y1_hat,
            budget=budget,
            utility=utility,
            threshold=threshold,
            require_positive_score=require_positive_score,
        )

    allocation = np.array([1 if A[i].X >= 0.5 else 0 for i in range(n)], dtype=int)
    return allocation

def solve_fair_allocation_payment(
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    group: np.ndarray,
    budget: float,
    fairness_weight: float,
    utility: str = "step",
    threshold: float = 0.60,
    payment_threshold: float | None = None,
    alpha: float = 1.0,
    time_limit: float = 180.0,
    require_positive_score: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Solve fairness-aware allocation with an expected payment budget.

    Objective:
        max_A sum_i A_i d_i
              - fairness_weight * sum_g (r_g(A) - r_overall(A))^2

    Payment budget constraint:
        sum_i expected_cost_i * A_i <= budget

    where:
        expected_cost_i = alpha * y1_hat_i * 1{y1_hat_i >= payment_threshold}

    By default, payment_threshold is the same as the utility threshold.

    The fairness penalty mirrors `solve_fair_allocation` and the DFL training
    penalty:
        sum_g (group_rate - overall_rate)^2

    with:
        s_i(A_i) = A_i u(y1_hat_i) + (1 - A_i) u(y0_hat_i)
        r_g(A) = mean_{i in g} s_i(A_i)
        r_overall(A) = mean_i s_i(A_i)

    Args:
        y0_hat: predicted outcome under no action.
        y1_hat: predicted outcome under action.
        group: group labels, e.g. "A"/"B" or 0/1.
        budget: expected payment budget.
        fairness_weight: lambda_f in the objective.
        utility: "step" or "linear".
        threshold: threshold used when utility="step".
        payment_threshold: threshold T for payment. If None, use `threshold`.
        alpha: converts wear into dollars/cost units. Default 1.
        time_limit: Gurobi time limit in seconds. If optimality is not proven,
            the best feasible incumbent found by this time is returned.
        require_positive_score: if True, force A_i = 0 whenever d_i <= 0.
        verbose: if True, print Gurobi output.

    Returns:
        allocation: binary numpy array where allocation[i] = 1 if user i is selected.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")
    if fairness_weight < 0:
        raise ValueError("fairness_weight must be nonnegative")
    if alpha < 0:
        raise ValueError("alpha must be nonnegative")

    if payment_threshold is None:
        payment_threshold = threshold

    y0_hat = np.asarray(y0_hat, dtype=float)
    y1_hat = np.asarray(y1_hat, dtype=float)
    group = np.asarray(group)

    if y0_hat.shape != y1_hat.shape:
        raise ValueError(f"Shape mismatch: y0_hat {y0_hat.shape}, y1_hat {y1_hat.shape}")
    if len(group) != len(y0_hat):
        raise ValueError(f"Length mismatch: group has length {len(group)}, y has length {len(y0_hat)}")

    n = len(y0_hat)
    allocation = np.zeros(n, dtype=int)
    if n == 0 or budget == 0:
        return allocation

    if utility == "step":
        s0 = step_utility(y0_hat, threshold=threshold)
        s1 = step_utility(y1_hat, threshold=threshold)
    elif utility == "linear":
        s0 = y0_hat.astype(float)
        s1 = y1_hat.astype(float)
    else:
        raise ValueError(f"Unknown utility: {utility}. Use 'step' or 'linear'.")

    d = s1 - s0

    expected_cost = alpha * y1_hat * (y1_hat >= payment_threshold).astype(float)
    expected_cost = np.asarray(expected_cost, dtype=float)

    unique_groups = list(pd.unique(group))

    model = gp.Model("fair_incentive_allocation_payment")
    model.Params.TimeLimit = time_limit
    model.Params.OutputFlag = 1 if verbose else 0

    A = model.addVars(n, vtype=GRB.BINARY, name="A")
    r_overall = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r_overall")
    r_group = model.addVars(len(unique_groups), lb=-GRB.INFINITY, ub=GRB.INFINITY, name="r_group")

    model.addConstr(
        gp.quicksum(float(expected_cost[i]) * A[i] for i in range(n)) <= float(budget),
        name="expected_payment_budget",
    )

    if require_positive_score:
        for i in range(n):
            if d[i] <= 0:
                model.addConstr(A[i] == 0, name=f"nonpositive_score_{i}")

    # s_i(A_i) = s0_i + A_i * (s1_i - s0_i) = s0_i + A_i * d_i
    model.addConstr(
        r_overall == (1.0 / n) * gp.quicksum(s0[i] + d[i] * A[i] for i in range(n)),
        name="overall_rate",
    )

    for g_idx, g_value in enumerate(unique_groups):
        idx = np.where(group == g_value)[0]
        if len(idx) == 0:
            continue
        model.addConstr(
            r_group[g_idx]
            == (1.0 / len(idx)) * gp.quicksum(s0[i] + d[i] * A[i] for i in idx),
            name=f"group_rate_{g_value}",
        )

    gain_expr = gp.quicksum(float(d[i]) * A[i] for i in range(n))
    fairness_penalty = gp.quicksum(
        (r_group[g_idx] - r_overall) * (r_group[g_idx] - r_overall)
        for g_idx in range(len(unique_groups))
    )

    model.setObjective(gain_expr - float(fairness_weight) * fairness_penalty, GRB.MAXIMIZE)
    model.optimize()

    if model.SolCount == 0:
        # Fallback: greedy by score per expected cost, respecting payment budget.
        candidate_idx = np.arange(n)
        if require_positive_score:
            candidate_idx = candidate_idx[d[candidate_idx] > 0]

        positive_cost = expected_cost > 1e-12
        zero_cost_idx = candidate_idx[~positive_cost[candidate_idx]]
        positive_cost_idx = candidate_idx[positive_cost[candidate_idx]]

        fallback = np.zeros(n, dtype=int)
        remaining_budget = float(budget)

        # Select beneficial zero-cost users first because they do not consume budget.
        if len(zero_cost_idx) > 0:
            zero_order = np.lexsort((zero_cost_idx, -d[zero_cost_idx]))
            for i in zero_cost_idx[zero_order]:
                if d[i] > 0 or not require_positive_score:
                    fallback[i] = 1

        if len(positive_cost_idx) > 0:
            ratio = d[positive_cost_idx] / expected_cost[positive_cost_idx]
            order = np.lexsort((positive_cost_idx, -ratio))
            for i in positive_cost_idx[order]:
                if expected_cost[i] <= remaining_budget + 1e-9:
                    fallback[i] = 1
                    remaining_budget -= expected_cost[i]

        return fallback

    allocation = np.array([1 if A[i].X >= 0.5 else 0 for i in range(n)], dtype=int)
    return allocation

def solve_top_b_allocation(
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    budget: int,
    utility: str = "step",
    threshold: float = 0.60,
    require_positive_score: bool = True,
) -> np.ndarray:
    """Select users with the top-B predicted utility gain.

    Args:
        y0_hat: predicted outcome under no action.
        y1_hat: predicted outcome under action.
        budget: maximum number of users to select.
        utility: "step" or "linear".
        threshold: threshold used when utility="step".
        require_positive_score: if True, select only users with score > 0.
            If fewer than B users have positive scores, fewer than B are selected.

    Returns:
        allocation: binary numpy array where allocation[i] = 1 if user i is selected.
    """
    if budget < 0:
        raise ValueError("budget must be nonnegative")

    scores = compute_scores(
        y0_hat=y0_hat,
        y1_hat=y1_hat,
        utility=utility,
        threshold=threshold,
    )

    n = len(scores)
    allocation = np.zeros(n, dtype=int)

    if budget == 0 or n == 0:
        return allocation

    if require_positive_score:
        candidate_idx = np.where(scores > 0)[0]
    else:
        candidate_idx = np.arange(n)

    if len(candidate_idx) == 0:
        return allocation

    k = min(budget, len(candidate_idx))

    # Stable tie-breaking: sort by descending score, then ascending user index.
    order = np.lexsort((candidate_idx, -scores[candidate_idx]))
    selected_idx = candidate_idx[order[:k]]
    allocation[selected_idx] = 1

    return allocation


def build_allocation_dataframe(
    user_id: np.ndarray,
    group: np.ndarray,
    y0_hat: np.ndarray,
    y1_hat: np.ndarray,
    allocation: np.ndarray,
    utility: str = "step",
    threshold: float = 0.60,
) -> pd.DataFrame:
    """Create a convenient dataframe containing predictions, scores, and allocation."""
    scores = compute_scores(
        y0_hat=y0_hat,
        y1_hat=y1_hat,
        utility=utility,
        threshold=threshold,
    )

    return pd.DataFrame(
        {
            "user_id": user_id,
            "group": group,
            "y0_hat": y0_hat,
            "y1_hat": y1_hat,
            "score_hat": scores,
            "selected": allocation.astype(int),
        }
    )


if __name__ == "__main__":
    # Tiny sanity check.
    y0 = np.array([0.50, 0.62, 0.08, 0.20])
    y1 = np.array([0.65, 0.70, 0.59, 0.80])
    alloc = solve_top_b_allocation(y0, y1, budget=2, utility="step", threshold=0.60)
    print("allocation:", alloc)