

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