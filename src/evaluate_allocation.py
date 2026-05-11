"""
Evaluate the trained PTO model.

Steps:
1. Load project/data/test.csv.
2. Load project/outputs/pto_model.pt.
3. For each test user, predict:
       y0_hat = f(X, pre, A=0)
       y1_hat = f(X, pre, A=1)
4. Call solve_top_b_allocation from optimize.py.
5. Evaluate selected users using true potential outcomes Y0_true and Y1_true.

Outputs:
    project/outputs/pto_test_allocations.csv
    project/outputs/pto_test_metrics.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from models import OutcomeNet, OutcomeNetNoSigmoid  # noqa: E402
from optimize import (  # noqa: E402
    compute_scores,
    solve_top_b_allocation,
    solve_fair_allocation_action,
    solve_fair_allocation_payment,
    step_utility,
)

DEFAULT_BUDGET_PCT = 0.01
DEFAULT_BUDGET = int(DEFAULT_BUDGET_PCT * 1500) # Based on test set size of 1500 users.
DEFAULT_THRESHOLD = 0.60
DEFAULT_UTILITY = "step"


def load_checkpoint(model_path: Path, device: torch.device) -> tuple[torch.nn.Module, list[str], StandardScaler, str]:
    if not model_path.exists():
        raise FileNotFoundError(
            f"Could not find {model_path}. Train the corresponding model first."
        )

    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    feature_cols = checkpoint["feature_cols"]
    model_class = checkpoint.get("model_class", "OutcomeNet")

    if model_class == "OutcomeNet":
        model = OutcomeNet(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        ).to(device)
    elif model_class == "OutcomeNetNoSigmoid":
        model = OutcomeNetNoSigmoid(
            input_dim=checkpoint["input_dim"],
            hidden_dim=checkpoint["hidden_dim"],
            dropout=checkpoint["dropout"],
        ).to(device)
    else:
        raise ValueError(f"Unsupported model_class in checkpoint: {model_class}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler = StandardScaler()
    scaler.mean_ = checkpoint["scaler_mean"]
    scaler.scale_ = checkpoint["scaler_scale"]
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = len(feature_cols)

    return model, feature_cols, scaler, model_class


def load_test_data(test_path: Path, feature_cols: list[str]) -> pd.DataFrame:
    if not test_path.exists():
        raise FileNotFoundError(
            f"Could not find {test_path}. Run `python src/generate_data.py` first."
        )

    df = pd.read_csv(test_path)
    df = pd.concat(
        [
            df,
            pd.DataFrame({"group_B": (df["group"] == "B").astype(float)}),
        ],
        axis=1,
    ).copy()

    required_cols = feature_cols + ["user_id", "group", "Y0_true", "Y1_true"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {test_path}: {missing}")

    return df


def predict_counterfactuals(
    model: torch.nn.Module,
    model_class: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    scaler: StandardScaler,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    df0 = df.copy()
    df1 = df.copy()
    df0["A_obs"] = 0
    df1["A_obs"] = 1

    x0 = scaler.transform(df0[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)
    x1 = scaler.transform(df1[feature_cols].to_numpy(dtype=np.float32)).astype(np.float32)

    x0_tensor = torch.from_numpy(x0).to(device)
    x1_tensor = torch.from_numpy(x1).to(device)

    model.eval()
    with torch.no_grad():
        y0 = model(x0_tensor)
        y1 = model(x1_tensor)

        # PTO OutcomeNet already has a final sigmoid. DFL OutcomeNetNoSigmoid does not.
        if model_class == "OutcomeNetNoSigmoid":
            y0 = torch.sigmoid(y0)
            y1 = torch.sigmoid(y1)

        y0_hat = y0.cpu().numpy()
        y1_hat = y1.cpu().numpy()

    return y0_hat, y1_hat


def evaluate_allocation(
    df: pd.DataFrame,
    allocation: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    allocation = np.asarray(allocation).astype(int)
    y0_true = df["Y0_true"].to_numpy(dtype=float)
    y1_true = df["Y1_true"].to_numpy(dtype=float)

    y_policy = allocation * y1_true + (1 - allocation) * y0_true
    y_no_action = y0_true
    y_all_action = y1_true

    s_policy = step_utility(y_policy, threshold=threshold)
    s_no_action = step_utility(y_no_action, threshold=threshold)
    s_all_action = step_utility(y_all_action, threshold=threshold)

    true_gain_continuous = np.sum(allocation * (y1_true - y0_true))
    true_gain_step = np.sum(
        allocation
        * (
            step_utility(y1_true, threshold=threshold)
            - step_utility(y0_true, threshold=threshold)
        )
    )
    actual_payment_cost = allocation * y1_true * (y1_true >= threshold).astype(float)
    actual_payment_budget_used = np.sum(actual_payment_cost)

    metrics = {
        "n_users": float(len(df)),
        "budget_used": float(allocation.sum()),
        "actual_payment_budget_used": float(actual_payment_budget_used),
        "mean_y_policy": float(y_policy.mean()),
        "mean_y_no_action": float(y_no_action.mean()),
        "mean_y_all_action": float(y_all_action.mean()),
        "success_rate_policy": float(s_policy.mean()),
        "success_rate_no_action": float(s_no_action.mean()),
        "success_rate_all_action": float(s_all_action.mean()),
        "true_gain_continuous_selected": float(true_gain_continuous),
        "true_gain_step_selected": float(true_gain_step),
    }

    eval_df = pd.concat(
        [
            df[["group"]].copy(),
            pd.DataFrame(
                {
                    "selected": allocation,
                    "y_policy": y_policy,
                    "s_policy": s_policy,
                    "actual_payment_cost": actual_payment_cost,
                },
                index=df.index,
            ),
        ],
        axis=1,
    )
    for group_name, group_df in eval_df.groupby("group"):
        idx = group_df.index.to_numpy()
        metrics[f"group_{group_name}_n"] = float(len(group_df))
        metrics[f"group_{group_name}_selected"] = float(allocation[idx].sum())
        metrics[f"group_{group_name}_selection_rate"] = float(allocation[idx].mean())
        metrics[f"group_{group_name}_actual_payment_budget_used"] = float(actual_payment_cost[idx].sum())
        metrics[f"group_{group_name}_mean_y_policy"] = float(y_policy[idx].mean())
        metrics[f"group_{group_name}_success_rate_policy"] = float(s_policy[idx].mean())

    return metrics


def oracle_top_b_allocation(
    df: pd.DataFrame,
    budget: int,
    utility: str,
    threshold: float,
) -> np.ndarray:
    y0_true = df["Y0_true"].to_numpy(dtype=float)
    y1_true = df["Y1_true"].to_numpy(dtype=float)
    return solve_top_b_allocation(
        y0_hat=y0_true,
        y1_hat=y1_true,
        budget=budget,
        utility=utility,
        threshold=threshold,
        require_positive_score=True,
    )


def random_allocation(df: pd.DataFrame, budget: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = len(df)
    allocation = np.zeros(n, dtype=int)
    k = min(budget, n)
    selected = rng.choice(np.arange(n), size=k, replace=False)
    allocation[selected] = 1
    return allocation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PTO allocation on test set.")
    parser.add_argument("--test-path", type=Path, default=PROJECT_ROOT / "data" / "test.csv")
    parser.add_argument("--pto-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "pto_model.pt")
    parser.add_argument("--dfl-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "payment_dfl_model.pt")
    parser.add_argument("--rs-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "payment_rs_model.pt")
    parser.add_argument("--pg-forward-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "payment_pg_forward_model.pt")
    parser.add_argument("--pg-backward-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "payment_pg_backward_model.pt")
    parser.add_argument("--pg-central-model-path", type=Path, default=PROJECT_ROOT / "outputs" / "payment_pg_central_model.pt")
    parser.add_argument("--allocation-path", type=Path, default=PROJECT_ROOT / "outputs" / "test_allocations.csv")
    parser.add_argument("--metrics-path", type=Path, default=PROJECT_ROOT / "outputs" / "test_metrics.csv")
    parser.add_argument("--budget", type=int, default=DEFAULT_BUDGET)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--utility", choices=["step", "linear"], default=DEFAULT_UTILITY)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    pto_model, pto_feature_cols, pto_scaler, pto_model_class = load_checkpoint(args.pto_model_path, device)
    dfl_model, dfl_feature_cols, dfl_scaler, dfl_model_class = load_checkpoint(args.dfl_model_path, device)
    rs_model, rs_feature_cols, rs_scaler, rs_model_class = load_checkpoint(args.rs_model_path, device)
    pg_forward_model, pg_forward_feature_cols, pg_forward_scaler, pg_forward_model_class = load_checkpoint(args.pg_forward_model_path, device)
    pg_backward_model, pg_backward_feature_cols, pg_backward_scaler, pg_backward_model_class = load_checkpoint(args.pg_backward_model_path, device)
    pg_central_model, pg_central_feature_cols, pg_central_scaler, pg_central_model_class = load_checkpoint(args.pg_central_model_path, device)

    if pto_feature_cols != dfl_feature_cols:
        raise ValueError("PTO and DFL checkpoints use different feature columns.")

    if pto_feature_cols != rs_feature_cols:
        raise ValueError("PTO and RS checkpoints use different feature columns.")
    
    # if pto_feature_cols != pg_feature_cols:
    #     raise ValueError("PTO and PG checkpoints use different feature columns.")
    
    if pto_feature_cols != pg_forward_feature_cols:
        raise ValueError("PTO and PG forward checkpoints use different feature columns.")
    if pto_feature_cols != pg_backward_feature_cols:
        raise ValueError("PTO and PG backward checkpoints use different feature columns.")
    if pto_feature_cols != pg_central_feature_cols:
        raise ValueError("PTO and PG central checkpoints use different feature columns.")

    feature_cols = pto_feature_cols
    df = load_test_data(args.test_path, feature_cols)
    print(f"Loaded test data: {df.shape}")
    print(f"Budget: {args.budget}, utility: {args.utility}, threshold: {args.threshold}")
    print(f"Loaded PTO model class: {pto_model_class}")
    print(f"Loaded DFL model class: {dfl_model_class}")
    print(f"Loaded RS model class: {rs_model_class}")
    # print(f"Loaded PG model class: {pg_model_class}")
    print(f"Loaded PG forward model class: {pg_forward_model_class}")
    print(f"Loaded PG backward model class: {pg_backward_model_class}")
    print(f"Loaded PG central model class: {pg_central_model_class}")

    pto_y0_hat, pto_y1_hat = predict_counterfactuals(
        model=pto_model,
        model_class=pto_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=pto_scaler,
        device=device,
    )

    dfl_y0_hat, dfl_y1_hat = predict_counterfactuals(
        model=dfl_model,
        model_class=dfl_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=dfl_scaler,
        device=device,
    )

    rs_y0_hat, rs_y1_hat = predict_counterfactuals(
        model=rs_model,
        model_class=rs_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=rs_scaler,
        device=device,
    )

    # pg_y0_hat, pg_y1_hat = predict_counterfactuals(
    #     model= pg_model,
    #     model_class=pg_model_class,
    #     df=df,
    #     feature_cols=feature_cols,
    #     scaler=pg_scaler,
    #     device=device,
    # )

    pgf_y0_hat, pgf_y1_hat = predict_counterfactuals(
        model= pg_forward_model,
        model_class=pg_forward_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=pg_forward_scaler,
        device=device,
    )

    pgb_y0_hat, pgb_y1_hat = predict_counterfactuals(
        model= pg_backward_model,
        model_class=pg_backward_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=pg_backward_scaler,
        device=device,
    )

    pgc_y0_hat, pgc_y1_hat = predict_counterfactuals(
        model= pg_central_model,
        model_class=pg_central_model_class,
        df=df,
        feature_cols=feature_cols,
        scaler=pg_central_scaler,
        device=device,
    )

    # pto_allocation = solve_top_b_allocation(
    #     y0_hat=pto_y0_hat,
    #     y1_hat=pto_y1_hat,
    #     budget=args.budget,
    #     utility=args.utility,
    #     threshold=args.threshold,
    #     require_positive_score=False,
    # )

    # pto_allocation = solve_fair_allocation_action(
    #     y0_hat=pto_y0_hat,
    #     y1_hat=pto_y1_hat,
    #     group=df["group"].to_numpy(),
    #     budget=args.budget,
    #     fairness_weight=1.0,
    #     utility=args.utility,
    #     threshold=args.threshold,
    #     require_positive_score=False,
    #     verbose=True,
    # )
    
    pto_allocation = solve_fair_allocation_payment(
        y0_hat=pto_y0_hat,
        y1_hat=pto_y1_hat,
        group=df["group"].to_numpy(),
        budget=args.budget,
        fairness_weight=10.0,
        utility=args.utility,
        threshold=args.threshold,
        payment_threshold = args.threshold,
        require_positive_score=False,
        verbose=True,
    )

    dfl_allocation = solve_top_b_allocation(
        y0_hat=dfl_y0_hat,
        y1_hat=dfl_y1_hat,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
        require_positive_score=False,
    )

    rs_allocation = solve_top_b_allocation(
        y0_hat=rs_y0_hat,
        y1_hat=rs_y1_hat,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
        require_positive_score=False,
    )

    # pg_allocation = solve_top_b_allocation(
    #     y0_hat=pg_y0_hat,
    #     y1_hat=pg_y1_hat,
    #     budget=args.budget,
    #     utility=args.utility,
    #     threshold=args.threshold,
    #     require_positive_score=False,  # PG can select negative scores since it's just a risk score.
    # )

    pgf_allocation = solve_top_b_allocation(
        y0_hat=pgf_y0_hat,
        y1_hat=pgf_y1_hat,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
        require_positive_score=False,
    )

    pgb_allocation = solve_top_b_allocation(
        y0_hat=pgb_y0_hat,
        y1_hat=pgb_y1_hat,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
        require_positive_score=False,
    )

    pgc_allocation = solve_top_b_allocation(
        y0_hat=pgc_y0_hat,
        y1_hat=pgc_y1_hat,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
        require_positive_score=False,
    )

    oracle_allocation = oracle_top_b_allocation(
        df=df,
        budget=args.budget,
        utility=args.utility,
        threshold=args.threshold,
    )
    rand_allocation = random_allocation(df=df, budget=args.budget, seed=args.seed)

    allocation_df = pd.DataFrame(
        {
            "user_id": df["user_id"].to_numpy(),
            "group": df["group"].to_numpy(),
            "pto_y0_hat": pto_y0_hat,
            "pto_y1_hat": pto_y1_hat,
            "pto_score_hat": compute_scores(
                pto_y0_hat,
                pto_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "pto_selected": pto_allocation,

            "dfl_y0_hat": dfl_y0_hat,
            "dfl_y1_hat": dfl_y1_hat,
            "dfl_score_hat": compute_scores(
                dfl_y0_hat,
                dfl_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "dfl_selected": dfl_allocation,

            "rs_y0_hat": rs_y0_hat,
            "rs_y1_hat": rs_y1_hat,
            "rs_score_hat": compute_scores(
                rs_y0_hat,
                rs_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "rs_selected": rs_allocation,

            # "pg_y0_hat": pg_y0_hat,
            # "pg_y1_hat": pg_y1_hat,
            # "pg_score_hat": compute_scores(
            #     pg_y0_hat,
            #     pg_y1_hat,
            #     utility=args.utility,
            #     threshold=args.threshold,
            # ),
            # "pg_selected": pg_allocation,

            "pgf_y0_hat": pgf_y0_hat,
            "pgf_y1_hat": pgf_y1_hat,
            "pgf_score_hat": compute_scores(
                pgf_y0_hat,
                pgf_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "pgf_selected": pgf_allocation,
            "pgb_y0_hat": pgb_y0_hat,
            "pgb_y1_hat": pgb_y1_hat,
            "pgb_score_hat": compute_scores(
                pgb_y0_hat,
                pgb_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "pgb_selected": pgb_allocation,
            "pgc_y0_hat": pgc_y0_hat,
            "pgc_y1_hat": pgc_y1_hat,
            "pgc_score_hat": compute_scores(
                pgc_y0_hat,
                pgc_y1_hat,
                utility=args.utility,
                threshold=args.threshold,
            ),
            "pgc_selected": pgc_allocation,

        }
    )
    y0_true = df["Y0_true"].to_numpy(dtype=float)
    y1_true = df["Y1_true"].to_numpy(dtype=float)
    true_payment_cost = y1_true * (y1_true >= args.threshold).astype(float)
    allocation_df = pd.concat(
        [
            allocation_df,
            pd.DataFrame(
                {
                    "Y0_true": y0_true,
                    "Y1_true": y1_true,
                    "true_payment_cost": true_payment_cost,
                    "true_score_step": compute_scores(
                        y0_true,
                        y1_true,
                        utility="step",
                        threshold=args.threshold,
                    ),
                    "true_score_linear": compute_scores(
                        y0_true,
                        y1_true,
                        utility="linear",
                        threshold=args.threshold,
                    ),
                    "oracle_selected": oracle_allocation,
                    "random_selected": rand_allocation,
                }
            ),
        ],
        axis=1,
    ).copy()

    metrics_rows = []
    for policy_name, allocation in [
        ("pto", pto_allocation),
        ("dfl", dfl_allocation),
        ("rs", rs_allocation),
        # ("pg", pg_allocation),
        ("pgf", pgf_allocation),
        ("pgb", pgb_allocation),
        ("pgc", pgc_allocation),
        ("oracle", oracle_allocation),
        ("random", rand_allocation),
        ("no_action", np.zeros(len(df), dtype=int)),
    ]:
        row = {"policy": policy_name, "allocated_budget": float(args.budget)}
        row.update(evaluate_allocation(df=df, allocation=allocation, threshold=args.threshold))
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)

    args.allocation_path.parent.mkdir(parents=True, exist_ok=True)
    allocation_df.to_csv(args.allocation_path, index=False)
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    append_metrics = args.metrics_path.exists() and args.metrics_path.stat().st_size > 0
    if append_metrics:
        existing_metrics_df = pd.read_csv(args.metrics_path)
        existing_columns = list(existing_metrics_df.columns)
        new_columns = list(metrics_df.columns)
        if existing_columns == new_columns:
            metrics_df.to_csv(args.metrics_path, mode="a", header=False, index=False)
        else:
            all_columns = existing_columns + [
                col for col in new_columns if col not in existing_columns
            ]
            existing_metrics_df = existing_metrics_df.reindex(columns=all_columns)
            metrics_df = metrics_df.reindex(columns=all_columns)
            pd.concat([existing_metrics_df, metrics_df], ignore_index=True).to_csv(
                args.metrics_path,
                index=False,
            )
    else:
        metrics_df.to_csv(args.metrics_path, index=False)
    print("\nMetrics:")
    print(metrics_df[[
        "policy",
        "allocated_budget",
        "budget_used",
        "actual_payment_budget_used",
        # "mean_y_policy",
        # "success_rate_policy",
        # "true_gain_continuous_selected",
        "true_gain_step_selected",
    ]])
    print(f"\nSaved allocations to {args.allocation_path}")
    print(f"{'Appended' if append_metrics else 'Saved'} metrics to {args.metrics_path}")

    print((dfl_y1_hat > 0.6).mean())
    print((dfl_y0_hat > 0.6).mean())
    print(((dfl_y0_hat <= 0.6) & (dfl_y1_hat > 0.6)).sum())
    print((dfl_y1_hat - dfl_y0_hat).min(), (dfl_y1_hat - dfl_y0_hat).max())


if __name__ == "__main__":
    main()
