"""Build PTO training data from simulated Fitbit wear CSVs."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


DAY_COLUMNS = [f"day_{day}" for day in range(1, 31)]
DEFAULT_ORIGINAL_PATH = Path("synthetic_fitbit_wear_test.csv")
DEFAULT_ACTION_0_PATH = Path("synthetic_fitbit_wear_next_month_0_test.csv")
DEFAULT_ACTION_1_PATH = Path("synthetic_fitbit_wear_next_month_1_test.csv")
DEFAULT_OUTPUT_PATH = Path("pto_training_data_test.csv")


def read_wear_csv(path: str | Path) -> pd.DataFrame:
    """Read a wide wear CSV and validate the columns used here."""

    df = pd.read_csv(path)
    required_columns = ["user_id", *DAY_COLUMNS]
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {missing_columns}")

    return df


def build_pto_training_data(
    original_path: str | Path = DEFAULT_ORIGINAL_PATH,
    action_0_path: str | Path = DEFAULT_ACTION_0_PATH,
    action_1_path: str | Path = DEFAULT_ACTION_1_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Create PTO training data with treatment-effect target by user."""

    original_df = read_wear_csv(original_path)
    action_0_df = read_wear_csv(action_0_path)
    action_1_df = read_wear_csv(action_1_path)

    action_0_sum = action_0_df.set_index("user_id")[DAY_COLUMNS].sum(axis=1)
    action_1_sum = action_1_df.set_index("user_id")[DAY_COLUMNS].sum(axis=1)
    target = (action_1_sum - action_0_sum).rename("target")

    training_df = original_df[["user_id", *DAY_COLUMNS]].merge(
        target,
        left_on="user_id",
        right_index=True,
        how="inner",
        validate="one_to_one",
    )

    training_df.to_csv(output_path, index=False)
    return training_df


def main() -> None:
    """Build and save the PTO training data CSV."""

    training_df = build_pto_training_data()
    print(f"Saved {len(training_df):,} rows to {DEFAULT_OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
