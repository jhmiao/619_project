

"""
Train a neural network to rank users by predicted treatment effect using
pairwise ranking loss.

Input CSV: pto_training_data.csv
Required columns:
    user_id, day_1, ..., day_30, target

The model learns a scalar score for each user. Within each mini-batch, we form
pairwise comparisons using target differences, but we subsample pairs so the
number of pairs does not explode.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, Dataset


FEATURE_COLS = [f"day_{i}" for i in range(1, 31)]
TARGET_COL = "target"
USER_ID_COL = "user_id"


class FitbitRankDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, user_ids: np.ndarray | None = None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.user_ids = user_ids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        if self.user_ids is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.y[idx], self.user_ids[idx]


class RankNet(nn.Module):
    """Small MLP matching the architecture used in train_pto.py."""

    def __init__(self, input_dim: int = 30, hidden_dim_1: int = 32, hidden_dim_2: int = 16):
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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pairwise_ranking_loss(
    scores: torch.Tensor,
    targets: torch.Tensor,
    max_pairs: int = 4096,
    min_target_diff: float = 0.0,
) -> torch.Tensor:
    """
    RankNet-style pairwise logistic loss.

    For each valid pair (i, j) with target_i > target_j, encourage
    score_i > score_j.

    To avoid O(batch_size^2) memory/compute becoming too large, this function
    randomly samples up to max_pairs valid pairs within the batch.
    """
    n = targets.shape[0]
    if n < 2:
        return scores.new_tensor(0.0, requires_grad=True)

    # valid_pairs[k] = [i, j], where target_i should rank above target_j.
    diff_targets = targets[:, None] - targets[None, :]
    valid_pairs = torch.nonzero(diff_targets > min_target_diff, as_tuple=False)

    if valid_pairs.numel() == 0:
        return scores.new_tensor(0.0, requires_grad=True)

    if valid_pairs.shape[0] > max_pairs:
        chosen = torch.randperm(valid_pairs.shape[0], device=valid_pairs.device)[:max_pairs]
        valid_pairs = valid_pairs[chosen]

    i = valid_pairs[:, 0]
    j = valid_pairs[:, 1]
    score_diff = scores[i] - scores[j]

    # softplus(-x) = log(1 + exp(-x)); small when score_i >> score_j.
    return torch.nn.functional.softplus(-score_diff).mean()


@torch.no_grad()
def predict_scores(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 1024) -> np.ndarray:
    model.eval()
    scores: List[np.ndarray] = []
    tensor_x = torch.tensor(X, dtype=torch.float32)
    loader = DataLoader(tensor_x, batch_size=batch_size, shuffle=False)
    for xb in loader:
        xb = xb.to(device)
        scores.append(model(xb).cpu().numpy())
    return np.concatenate(scores)


def top_b_metrics(scores: np.ndarray, targets: np.ndarray, budgets: List[int]) -> Dict[str, float]:
    """Decision-focused metrics: value and regret for top-B selection."""
    out: Dict[str, float] = {}
    n = len(targets)
    for b in budgets:
        b_eff = min(b, n)
        model_idx = np.argsort(scores)[-b_eff:]
        oracle_idx = np.argsort(targets)[-b_eff:]

        model_value = float(targets[model_idx].sum())
        oracle_value = float(targets[oracle_idx].sum())
        random_expected = float(b_eff * targets.mean())
        regret = oracle_value - model_value
        overlap = len(set(model_idx.tolist()).intersection(set(oracle_idx.tolist()))) / b_eff

        out[f"top_{b_eff}_model_value"] = model_value
        out[f"top_{b_eff}_oracle_value"] = oracle_value
        out[f"top_{b_eff}_random_expected_value"] = random_expected
        out[f"top_{b_eff}_regret"] = regret
        out[f"top_{b_eff}_oracle_overlap"] = float(overlap)
    return out


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device, budgets: List[int]) -> Dict[str, float]:
    scores = predict_scores(model, X, device)
    mse = float(np.mean((scores - y) ** 2))
    mae = float(np.mean(np.abs(scores - y)))

    # Spearman correlation without requiring scipy.
    pred_rank = pd.Series(scores).rank(method="average").to_numpy()
    true_rank = pd.Series(y).rank(method="average").to_numpy()
    spearman = float(np.corrcoef(pred_rank, true_rank)[0, 1])

    metrics = {
        "mse": mse,
        "mae": mae,
        "spearman": spearman,
    }
    metrics.update(top_b_metrics(scores, y, budgets))
    return metrics


def train_one_split(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[nn.Module, Dict[str, float]]:
    train_ds = FitbitRankDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = RankNet(
        input_dim=X_train.shape[1],
        hidden_dim_1=args.hidden_dim_1,
        hidden_dim_2=args.hidden_dim_2,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    mse_fn = nn.MSELoss()

    best_state = None
    best_val_score = -np.inf
    best_metrics: Dict[str, float] = {}
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_rank_loss = 0.0
        total_mse_loss = 0.0
        num_batches = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            scores = model(xb)
            rank_loss = pairwise_ranking_loss(
                scores,
                yb,
                max_pairs=args.max_pairs_per_batch,
                min_target_diff=args.min_target_diff,
            )
            mse_loss = mse_fn(scores, yb)
            loss = rank_loss + args.mse_weight * mse_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += float(loss.item())
            total_rank_loss += float(rank_loss.item())
            total_mse_loss += float(mse_loss.item())
            num_batches += 1

        val_metrics = evaluate(model, X_val, y_val, device, args.budgets)

        # Model selection: prioritize the first budget's top-B value.
        primary_budget = min(args.budgets[0], len(y_val))
        val_score = val_metrics[f"top_{primary_budget}_model_value"]

        if val_score > best_val_score:
            best_val_score = val_score
            best_metrics = val_metrics
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1

        if args.verbose:
            avg_loss = total_loss / max(num_batches, 1)
            avg_rank = total_rank_loss / max(num_batches, 1)
            avg_mse = total_mse_loss / max(num_batches, 1)
            print(
                f"Epoch {epoch:03d} | "
                f"train_loss={avg_loss:.4f} rank={avg_rank:.4f} mse={avg_mse:.4f} | "
                f"val_spearman={val_metrics['spearman']:.4f} | "
                f"val_top{primary_budget}={val_score:.2f} | "
                f"patience={patience_left}"
            )

        if patience_left <= 0:
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find input CSV: {csv_path}")

    df = pd.read_csv(csv_path)
    required = [USER_ID_COL, *FEATURE_COLS, TARGET_COL]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df[required].copy()
    df[FEATURE_COLS] = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).reset_index(drop=True)
    return df


def run_cross_validation(df: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> pd.DataFrame:
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df[TARGET_COL].to_numpy(dtype=np.float32)

    kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    fold_rows: List[Dict[str, float]] = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        print(f"\n===== Fold {fold}/{args.cv_folds} =====")
        X_train_raw, X_val_raw = X_all[train_idx], X_all[val_idx]
        y_train, y_val = y_all[train_idx], y_all[val_idx]

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
        X_val = scaler.transform(X_val_raw).astype(np.float32)

        _, metrics = train_one_split(X_train, y_train, X_val, y_val, args, device)
        metrics = {"fold": fold, **metrics}
        fold_rows.append(metrics)
        print(json.dumps(metrics, indent=2))

    cv_df = pd.DataFrame(fold_rows)
    return cv_df


def train_final_model(df: pd.DataFrame, args: argparse.Namespace, device: torch.device) -> None:
    X_all = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y_all = df[TARGET_COL].to_numpy(dtype=np.float32)
    user_ids = df[USER_ID_COL].to_numpy()

    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.val_size,
        random_state=args.seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[train_idx]).astype(np.float32)
    X_val = scaler.transform(X_all[val_idx]).astype(np.float32)
    X_all_scaled = scaler.transform(X_all).astype(np.float32)

    model, val_metrics = train_one_split(
        X_train,
        y_all[train_idx],
        X_val,
        y_all[val_idx],
        args,
        device,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "pair_rank_model.pt"
    scaler_path = args.output_dir / "pair_rank_scaler.npz"
    metrics_path = args.output_dir / "pair_rank_metrics.json"
    pred_path = args.output_dir / "pair_rank_training_predictions.csv"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
            "args": vars(args),
        },
        model_path,
    )

    np.savez(
        scaler_path,
        mean=scaler.mean_,
        scale=scaler.scale_,
        feature_cols=np.array(FEATURE_COLS),
    )

    all_scores = predict_scores(model, X_all_scaled, device)
    pred_df = pd.DataFrame(
        {
            USER_ID_COL: user_ids,
            "pair_rank_score": all_scores,
            TARGET_COL: y_all,
        }
    )
    pred_df.to_csv(pred_path, index=False)

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)

    print("\nSaved final artifacts:")
    print(f"  model:   {model_path}")
    print(f"  scaler:  {scaler_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  preds:   {pred_path}")
    print("\nValidation metrics:")
    print(json.dumps(val_metrics, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train pairwise ranking NN for PTO data.")
    parser.add_argument("--data", type=Path, default=Path("pto_training_data.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs_pair_rank"))
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim-1", type=int, default=32)
    parser.add_argument("--hidden-dim-2", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=20)

    # Pair control: batch_size=256 has ~65k possible ordered pairs, but this
    # samples at most 4096 valid pairs per batch by default.
    parser.add_argument("--max-pairs-per-batch", type=int, default=4096)
    parser.add_argument("--min-target-diff", type=float, default=0.0)

    # Hybrid loss. Set to 0.0 for pure pairwise ranking.
    parser.add_argument("--mse-weight", type=float, default=0.1)

    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--skip-cv", action="store_true")
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 500, 1000])
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    df = load_data(args.data)
    print(f"Loaded {len(df)} rows from {args.data}")

    if not args.skip_cv:
        cv_df = run_cross_validation(df, args, device)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cv_path = args.output_dir / "pair_rank_cv_results.csv"
        cv_df.to_csv(cv_path, index=False)
        print("\nCross-validation summary:")
        print(cv_df.drop(columns=["fold"]).agg(["mean", "std"]))
        print(f"Saved CV results to {cv_path}")

    print("\n===== Training final model =====")
    train_final_model(df, args, device)


if __name__ == "__main__":
    main()
