

"""Generate synthetic Fitbit wear data from latent user groups.

Each user follows a two-state Markov chain over 30 days:
    0 = below wear threshold
    1 = above wear threshold

The script creates 1,000 users from six latent groups and saves one CSV row
per user, with day-level wear states in wide columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class UserGroup:
    """Parameters defining one latent user group."""

    name: str
    probability: float
    p01_mean: float  # P(wear tomorrow | not wear today), no incentive
    p11_mean: float  # P(wear tomorrow | wear today), no incentive
    tau01_mean: float  # incentive lift for 0 -> 1 transition
    tau11_mean: float  # incentive lift for 1 -> 1 transition


GROUPS: tuple[UserGroup, ...] = (
    # "group name", probability, p01_mean, p11_mean, tau01_mean, tau11_mean
    UserGroup("highly_compliant", 0.20, 0.75, 0.95, 0.03, 0.01),
    UserGroup("forgetful_responsive", 0.25, 0.35, 0.80, 0.25, 0.08),
    UserGroup("habit_forming_candidate", 0.20, 0.20, 0.85, 0.1, 0.03),
    UserGroup("intermittent_unstable", 0.15, 0.40, 0.50, 0.15, 0.08),
    UserGroup("disengaged_nonresponsive", 0.05, 0.01, 0.25, 0.03, 0.02),
    UserGroup("incentive_dependent", 0.15, 0.20, 0.35, 0.40, 0.18),
)

P_SD = 0.05
TAU_SD = 0.03
PARAMETER_COLUMNS = ("p01_0", "p11_0", "tau01", "tau11", "p01_1", "p11_1")


def clip_probability(x: float) -> float:
    """Keep probabilities away from exactly 0 and 1."""

    return float(np.clip(x, 0.01, 0.99))


def sample_user_parameters(group: UserGroup, rng: np.random.Generator) -> dict[str, float]:
    """Sample one user's transition parameters around group-level means.

    This is called once per person, not once per day, so each user's
    parameters stay fixed across their time series while differing slightly
    from other users in the same latent group.
    """

    p01_0 = clip_probability(rng.normal(group.p01_mean, P_SD))
    p11_0 = clip_probability(rng.normal(group.p11_mean, P_SD))
    tau01 = float(np.clip(rng.normal(group.tau01_mean, TAU_SD), 0.0, 1.0 - p01_0))
    tau11 = float(np.clip(rng.normal(group.tau11_mean, TAU_SD), 0.0, 1.0 - p11_0))

    return {
        "p01_0": p01_0,
        "p11_0": p11_0,
        "tau01": tau01,
        "tau11": tau11,
        "p01_1": clip_probability(p01_0 + tau01),
        "p11_1": clip_probability(p11_0 + tau11),
    }


def simulate_user(
    user_id: int,
    group: UserGroup,
    params: dict[str, float],
    n_days: int,
    rng: np.random.Generator,
    action: int = 0,
) -> dict[str, object]:
    """Simulate daily wear states for a single user.

    Args:
        user_id: Unique user identifier.
        group: Latent group for the user.
        params: User-level transition parameters.
        n_days: Number of days to simulate.
        rng: NumPy random generator.
        action: 0 for no incentive, 1 for incentive.

    Returns:
        Wide-format record, one row per user.
    """

    if action not in (0, 1):
        raise ValueError("action must be 0 or 1")

    # Initialize from the no-incentive stationary distribution.
    # pi_1 = p01 / (p01 + p10), where p10 = 1 - p11.
    p10_0 = 1.0 - params["p11_0"]
    init_prob_wear = params["p01_0"] / (params["p01_0"] + p10_0)
    state = int(rng.random() < init_prob_wear)

    record: dict[str, object] = {
        "user_id": user_id,
        "group": group.name,
        "action": action,
    }

    for day in range(1, n_days + 1):
        record[f"day_{day}"] = state

        transition_prob = transition_probability(state, params, action)
        state = int(rng.random() < transition_prob)

    record.update(params)
    return record


def transition_probability(state: int, params: dict[str, float], action: int) -> float:
    """Return P(next state is 1) for the current state and action."""

    if action not in (0, 1):
        raise ValueError("action must be 0 or 1")

    if state == 0:
        return params["p01_1"] if action == 1 else params["p01_0"]
    return params["p11_1"] if action == 1 else params["p11_0"]


def generate_dataset(
    n_users: int = 1000,
    n_days: int = 30,
    action: int = 0,
    seed: int = 619,
) -> pd.DataFrame:
    """Generate the full synthetic dataset."""

    rng = np.random.default_rng(seed)
    group_probs = np.array([group.probability for group in GROUPS], dtype=float)
    group_probs = group_probs / group_probs.sum()
    sampled_groups = rng.choice(GROUPS, size=n_users, p=group_probs)

    all_records: list[dict[str, object]] = []
    for user_id, group in enumerate(sampled_groups, start=1):
        # Person-level randomization: sample once, then reuse for all days.
        params = sample_user_parameters(group, rng)
        all_records.append(
            simulate_user(
                user_id=user_id,
                group=group,
                params=params,
                n_days=n_days,
                rng=rng,
                action=action,
            )
        )

    return pd.DataFrame(all_records)


def generate_next_month(
    input_path: str | Path,
    action: int,
    output_path: str | Path | None = None,
    n_days: int = 30,
    seed: int = 620,
) -> pd.DataFrame:
    """Generate the next month from an existing wide-format CSV.

    The input CSV must contain user_id, group, a day-30 state column, and the
    parameter columns p01_0, p11_0, tau01, tau11, p01_1, p11_1. The returned
    and saved DataFrame has the same wide row format as generate_dataset.
    """

    if action not in (0, 1):
        raise ValueError("action must be 0 or 1")

    input_path = Path(input_path)
    df = pd.read_csv(input_path)
    initial_state_column = "day_30"

    required_columns = ("user_id", "group", *PARAMETER_COLUMNS)
    missing_columns = [column for column in required_columns if column not in df.columns]
    if initial_state_column is None:
        missing_columns.append("day_30")
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    rng = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        params = {column: float(getattr(row, column)) for column in PARAMETER_COLUMNS}
        state = int(getattr(row, initial_state_column))
        record: dict[str, object] = {
            "user_id": getattr(row, "user_id"),
            "group": getattr(row, "group"),
            "action": action,
        }

        for day in range(1, n_days + 1):
            transition_prob = transition_probability(state, params, action)
            state = int(rng.random() < transition_prob)
            record[f"day_{day}"] = state

        record.update(params)
        records.append(record)

    next_month_df = pd.DataFrame(records)
    if output_path is not None:
        next_month_df.to_csv(output_path, index=False)

    return next_month_df


def main() -> None:
    """Generate and save synthetic data."""

    output_path = Path("synthetic_fitbit_wear_test.csv")
    df = generate_dataset(n_users=10000, n_days=30, action=0, seed=6199)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} user rows to {output_path.resolve()}")
    print(df.groupby("group")["user_id"].nunique().sort_values(ascending=False))

    action = 1
    df_next = generate_next_month(
        "synthetic_fitbit_wear_test.csv",
        action=action,
        output_path=f"synthetic_fitbit_wear_next_month_{action}_test.csv",
    )
    action = 0
    df_next = generate_next_month(
        "synthetic_fitbit_wear_test.csv",
        action=action,
        output_path=f"synthetic_fitbit_wear_next_month_{action}_test.csv",
    )
if __name__ == "__main__":
    main()
