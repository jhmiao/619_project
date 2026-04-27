"""Compare PTO and pair-rank performance against the oracle ranking."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt


PTO_RESULTS_PATH = Path("results_test.csv")
PAIR_RANK_RESULTS_PATH = Path("pair_rank_results.csv")
CURVE_OUTPUT_PATH = Path("compare_results_curve.csv")
PLOT_OUTPUT_PATH = Path("compare_results.png")
RANDOM_SEED = 619


def read_results(path: str | Path, score_column: str) -> pd.DataFrame:
    """Read a result CSV and validate the needed columns."""

    df = pd.read_csv(path)
    required_columns = {"user_id", "target", score_column}
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return df[["user_id", "target", score_column]]


def cumsum_by_score(df: pd.DataFrame, score_column: str, target_column: str) -> np.ndarray:
    """Return cumulative target after ranking rows by score descending."""

    return df.sort_values(score_column, ascending=False)[target_column].cumsum().to_numpy(dtype=float)


def build_comparison_curve(
    pto_results_path: str | Path = PTO_RESULTS_PATH,
    pair_rank_results_path: str | Path = PAIR_RANK_RESULTS_PATH,
) -> pd.DataFrame:
    """Build top-K regret curves for PTO, pair-rank, and random ranking."""

    pto_df = read_results(pto_results_path, "pto_pred")
    pair_rank_df = read_results(pair_rank_results_path, "pair_rank_score")
    df = pto_df.merge(
        pair_rank_df,
        on="user_id",
        suffixes=("_pto", "_pair_rank"),
        validate="one_to_one",
    )
    if not np.allclose(df["target_pto"], df["target_pair_rank"]):
        raise ValueError("Target values differ between PTO and pair-rank result files.")

    df["target"] = df["target_pto"]
    oracle_cumsum = df.sort_values("target", ascending=False)["target"].cumsum().to_numpy(dtype=float)
    pto_cumsum = cumsum_by_score(df, "pto_pred", "target_pto")
    pair_rank_cumsum = cumsum_by_score(df, "pair_rank_score", "target_pair_rank")
    random_cumsum = (
        df.sample(frac=1.0, random_state=RANDOM_SEED)["target"]
        .cumsum()
        .to_numpy(dtype=float)
    )
    pto_regret = oracle_cumsum - pto_cumsum
    pair_rank_regret = oracle_cumsum - pair_rank_cumsum
    random_regret = oracle_cumsum - random_cumsum

    return pd.DataFrame(
        {
            "rank": np.arange(1, len(df) + 1),
            "oracle_cumsum": oracle_cumsum,
            "pto_cumsum": pto_cumsum,
            "pair_rank_cumsum": pair_rank_cumsum,
            "random_cumsum": random_cumsum,
            "pto_regret": pto_regret,
            "pair_rank_regret": pair_rank_regret,
            "random_regret": random_regret,
        }
    )


def plot_comparison_curve(curve_df: pd.DataFrame, output_path: str | Path = PLOT_OUTPUT_PATH) -> None:
    """Save a line plot of top-K regret relative to oracle."""

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(curve_df["rank"], curve_df["pto_regret"], linewidth=1.8, label="PTO")
    ax.plot(
        curve_df["rank"],
        curve_df["pair_rank_regret"],
        linewidth=1.8,
        label="Pair rank",
    )
    ax.plot(
        curve_df["rank"],
        curve_df["random_regret"],
        linewidth=1.8,
        label="Random",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Oracle top-K target - model top-K target")
    ax.set_title("Top-K Regret vs Oracle")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    """Create comparison CSV and plot."""

    curve_df = build_comparison_curve(PTO_RESULTS_PATH, PAIR_RANK_RESULTS_PATH)
    curve_df.to_csv(CURVE_OUTPUT_PATH, index=False)
    plot_comparison_curve(curve_df, PLOT_OUTPUT_PATH)
    print(f"Saved comparison curve to {CURVE_OUTPUT_PATH.resolve()}")
    print(f"Saved line plot to {PLOT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
