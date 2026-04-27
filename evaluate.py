

"""Evaluate a trained PTO or pair-rank model on training/evaluation data.

Inputs:
    pto_model.pt or pair_rank_model.pt
    pto_training_data_test.csv if it exists, otherwise pto_training_data.csv

Expected data columns:
    user_id, day_1, day_2, ..., day_30, target

Output:
    pto_results.csv with columns: user_id, pto_pred, target, or
    pair_rank_results.csv with columns: user_id, pair_rank_score, target
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


FEATURE_COLS = [f"day_{i}" for i in range(1, 31)]


class ScoreModel(nn.Module):
    """Small MLP matching the architecture used in train_pto.py."""

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


def choose_data_path() -> Path:
    """Use test data if present; otherwise fall back to training data."""

    test_path = Path("pto_training_data_test.csv")
    train_path = Path("pto_training_data.csv")

    if test_path.exists():
        return test_path
    if train_path.exists():
        return train_path
    raise FileNotFoundError("Could not find pto_training_data_test.csv or pto_training_data.csv.")


def infer_model_type(model_path: Path, checkpoint: dict[str, object]) -> str:
    """Infer which result score column to write."""

    if "pair_rank" in model_path.name or "args" in checkpoint:
        return "pair_rank"
    return "pto"


def infer_model_dims(state_dict: dict[str, torch.Tensor]) -> tuple[int, int, int]:
    """Infer input and hidden dimensions from a saved ScoreModel state dict."""

    try:
        input_dim = int(state_dict["net.0.weight"].shape[1])
        hidden_dim_1 = int(state_dict["net.0.weight"].shape[0])
        hidden_dim_2 = int(state_dict["net.2.weight"].shape[0])
    except KeyError as exc:
        raise ValueError("Checkpoint does not match the expected MLP architecture.") from exc

    return input_dim, hidden_dim_1, hidden_dim_2


def default_output_path(model_type: str) -> Path:
    """Return the default output path for a model type."""

    if model_type == "pair_rank":
        return Path("pair_rank_results.csv")
    return Path("pto_results.csv")


def load_pair_rank_scaler(model_path: Path, scaler_path: Path | None) -> tuple[np.ndarray, np.ndarray]:
    """Load the StandardScaler statistics saved by train_pair_rank.py."""

    if scaler_path is None:
        scaler_path = model_path.parent / "pair_rank_scaler.npz"
    if not scaler_path.exists():
        raise FileNotFoundError(
            f"Could not find pair-rank scaler file: {scaler_path.resolve()}. "
            "Pass --scaler-path or evaluate with the scaler saved by train_pair_rank.py."
        )

    scaler = np.load(scaler_path, allow_pickle=True)
    return scaler["mean"].astype(np.float32), scaler["scale"].astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTO or pair-rank model on PTO-format data.")
    parser.add_argument("--model-path", type=Path, default=Path("pto_model.pt"))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--scaler-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    """Load model, generate predictions, and save result scores."""

    args = parse_args()
    model_path = args.model_path
    data_path = args.data_path if args.data_path is not None else choose_data_path()

    # This project writes a local checkpoint that includes training metadata
    # such as pathlib.Path values in config. PyTorch 2.6+ defaults
    # weights_only=True, which rejects those non-tensor objects.
    if not model_path.exists():
        raise FileNotFoundError(f"Could not find model file: {model_path.resolve()}")
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model_type = infer_model_type(model_path, checkpoint)
    output_path = args.output_path if args.output_path is not None else default_output_path(model_type)
    score_col = "pair_rank_score" if model_type == "pair_rank" else "pto_pred"
    feature_cols = checkpoint.get("feature_cols", FEATURE_COLS)
    state_dict = checkpoint["model_state_dict"]
    input_dim, hidden_dim_1, hidden_dim_2 = infer_model_dims(state_dict)
    if input_dim != len(feature_cols):
        raise ValueError(
            f"Checkpoint expects {input_dim} input features, but feature_cols has {len(feature_cols)} columns."
        )

    df = pd.read_csv(data_path)
    required_cols = ["user_id", "target", *feature_cols]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in {data_path}: {missing_cols}")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    if model_type == "pair_rank":
        mean, scale = load_pair_rank_scaler(model_path, args.scaler_path)
        if len(mean) != X.shape[1] or len(scale) != X.shape[1]:
            raise ValueError("Pair-rank scaler dimensions do not match input features.")
        X = ((X - mean) / scale).astype(np.float32)

    x_tensor = torch.from_numpy(X)

    model = ScoreModel(
        input_dim=input_dim,
        hidden_dim_1=hidden_dim_1,
        hidden_dim_2=hidden_dim_2,
    )
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        preds = model(x_tensor).cpu().numpy()

    results = pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(),
            score_col: preds,
            "target": df["target"].to_numpy(),
        }
    )
    results.to_csv(output_path, index=False)
    print(f"Loaded {model_type} model from {model_path}")
    print(f"Loaded data from {data_path}")
    print(f"Saved {len(results):,} rows to {output_path.resolve()}")


if __name__ == "__main__":
    main()
