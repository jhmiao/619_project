"""
Generate simulated data for the incentive allocation experiment.

Each user has:
- covariates X
- group label A/B
- pre-intervention wear sequence
- baseline transition probabilities p=P(1->1), q=P(0->0)
- treatment response parameters tau_p, tau_q
- potential outcomes Y0_true and Y1_true
- randomized observed action A_obs and observed outcome Y_obs

The script saves train/val/test CSV files under project/data.
"""

from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
N_USERS = 10_000
N_PRE_DAYS = 30
N_POST_DAYS = 30
TRAIN_FRAC = 0.70

VAL_FRAC = 0.15

# Standard deviation of latent user-level noise added to transition logits.
# Larger values make treatment effects harder to predict from observed covariates.
TRANSITION_NOISE_STD = 0.50


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for numpy arrays."""
    return 1.0 / (1.0 + np.exp(-x))


def simulate_markov_sequence(
    rng: np.random.Generator,
    p11: float,
    p00: float,
    n_days: int,
    init_prob_one: float,
) -> np.ndarray:
    """
    Simulate a binary Markov chain.

    p11 = P(V_t = 1 | V_{t-1} = 1)
    p00 = P(V_t = 0 | V_{t-1} = 0)
    """
    seq = np.zeros(n_days, dtype=int)
    seq[0] = int(rng.random() < init_prob_one)

    for t in range(1, n_days):
        if seq[t - 1] == 1:
            seq[t] = int(rng.random() < p11)
        else:
            seq[t] = int(rng.random() >= p00)

    return seq


def build_transition_probabilities(
    x: np.ndarray,
    group: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = TRANSITION_NOISE_STD,
) -> pd.DataFrame:
    """
    Map covariates and group membership to baseline and treated transition probabilities.

    The coefficients are chosen to create heterogeneous baseline adherence and treatment response.
    Group B is slightly harder at baseline but somewhat more responsive to action.

    In addition, latent Gaussian noise is added to the transition logits so that
    users with identical observed covariates can still behave differently.
    This makes the downstream decision problem less deterministic and generally
    more challenging for PTO.
    """
    group_b = (group == "B").astype(float)

    x1 = x[:, 0]
    x2 = x[:, 1]
    x3 = x[:, 2]
    x4 = x[:, 3]
    x5 = x[:, 4]

    n_users = x.shape[0]

    # Latent user-level heterogeneity.
    # These noises affect the logits before sigmoid transformation.
    eps_p = rng.normal(loc=0.0, scale=noise_std, size=n_users)
    eps_q = rng.normal(loc=0.0, scale=noise_std, size=n_users)
    eps_tau_p = rng.normal(loc=0.0, scale=noise_std, size=n_users)
    eps_tau_q = rng.normal(loc=0.0, scale=noise_std, size=n_users)

    # Baseline persistence probabilities.
    p = sigmoid(
        0.70 + 0.85 * x1 - 0.35 * x2 + 0.25 * x3 - 0.45 * group_b + eps_p
    )
    q = sigmoid(
        0.35 - 0.65 * x1 + 0.45 * x2 - 0.25 * x4 + 0.35 * group_b + eps_q
    )

    # Treatment responsiveness. These are not direct probability increases;
    # they are scaled through p' = p + (1-p) tau_p and q' = q + (1-q) tau_q.
    tau_p = sigmoid(
        -1.55 + 0.65 * x2 - 0.30 * x3 + 0.55 * group_b + eps_tau_p
    )
    tau_q = sigmoid(
        -1.75 + 0.50 * x4 + 0.25 * x5 + 0.45 * group_b + eps_tau_q
    )

    p_treated = p + (1.0 - p) * tau_p
    q_treated = q * (1.0 - tau_q)

    return pd.DataFrame(
        {
            "p": p,
            "q": q,
            "tau_p": tau_p,
            "tau_q": tau_q,
            "p_treated": p_treated,
            "q_treated": q_treated,
        }
    )


def generate_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    user_id = np.arange(N_USERS)

    # Two groups with unequal sizes to make group-rate fairness meaningful.
    group = rng.choice(["A", "B"], size=N_USERS, p=[0.65, 0.35])

    # Five covariates.
    x = rng.normal(loc=0.0, scale=1.0, size=(N_USERS, 5))
    transition_df = build_transition_probabilities(x, group, rng=rng, noise_std=TRANSITION_NOISE_STD)

    rows = []
    for i in range(N_USERS):
        p_i = float(transition_df.loc[i, "p"])
        q_i = float(transition_df.loc[i, "q"])
        p1_i = float(transition_df.loc[i, "p_treated"])
        q1_i = float(transition_df.loc[i, "q_treated"])

        # Initial probability roughly follows the stationary probability of the baseline chain.
        # For p=P(1->1), q=P(0->0), stationary P(1)=(1-q)/(2-p-q).
        init_prob_one = (1.0 - q_i) / max(2.0 - p_i - q_i, 1e-8)
        init_prob_one = float(np.clip(init_prob_one, 0.05, 0.95))

        pre_seq = simulate_markov_sequence(rng, p_i, q_i, N_PRE_DAYS, init_prob_one)
        y0_seq = simulate_markov_sequence(rng, p_i, q_i, N_POST_DAYS, pre_seq[-1])
        y1_seq = simulate_markov_sequence(rng, p1_i, q1_i, N_POST_DAYS, pre_seq[-1])

        y0_true = float(y0_seq.mean())
        y1_true = float(y1_seq.mean())

        a_obs = int(rng.random() < 0.5)
        y_obs = y1_true if a_obs == 1 else y0_true

        row = {
            "user_id": int(user_id[i]),
            "group": group[i],
            "A_obs": a_obs,
            "Y_obs": y_obs,
            "Y0_true": y0_true,
            "Y1_true": y1_true,
            "S0_true": int(y0_true > 0.60),
            "S1_true": int(y1_true > 0.60),
            "delta_true": y1_true - y0_true,
            "delta_step_true": int(y1_true > 0.60) - int(y0_true > 0.60),
            "p": p_i,
            "q": q_i,
            "tau_p": float(transition_df.loc[i, "tau_p"]),
            "tau_q": float(transition_df.loc[i, "tau_q"]),
            "p_treated": p1_i,
            "q_treated": q1_i,
        }

        for j in range(x.shape[1]):
            row[f"x{j + 1}"] = float(x[i, j])

        for t in range(N_PRE_DAYS):
            row[f"pre_{t + 1}"] = int(pre_seq[t])

        # These sequence columns are for debugging/oracle analysis only.
        # Training should normally use X, group, pre-sequence, A_obs, and Y_obs.
        for t in range(N_POST_DAYS):
            row[f"post0_{t + 1}"] = int(y0_seq[t])
            row[f"post1_{t + 1}"] = int(y1_seq[t])

        rows.append(row)

    return pd.DataFrame(rows)


def split_and_save(df: pd.DataFrame) -> None:
    rng = np.random.default_rng(SEED)
    shuffled = df.iloc[rng.permutation(len(df))].reset_index(drop=True)

    n_train = int(TRAIN_FRAC * len(shuffled))
    n_val = int(VAL_FRAC * len(shuffled))

    train = shuffled.iloc[:n_train].reset_index(drop=True)
    val = shuffled.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test = shuffled.iloc[n_train + n_val :].reset_index(drop=True)

    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    train.to_csv(data_dir / "train.csv", index=False)
    val.to_csv(data_dir / "val.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)

    print(f"Saved train: {train.shape} -> {data_dir / 'train.csv'}")
    print(f"Saved val:   {val.shape} -> {data_dir / 'val.csv'}")
    print(f"Saved test:  {test.shape} -> {data_dir / 'test.csv'}")
    print("Group counts:")
    print(shuffled["group"].value_counts().sort_index())
    print("Mean potential outcomes:")
    print(shuffled[["Y0_true", "Y1_true", "delta_true", "S0_true", "S1_true", "delta_step_true"]].mean())

    # plot histograms of Y0_true, Y1_true, delta_true, and delta_step_true by group for sanity check
    import matplotlib.pyplot as plt
    # import seaborn as sns

    for col in ["Y0_true", "Y1_true", "delta_true", "delta_step_true"]:
        plt.figure(figsize=(10, 6))
        for group in shuffled["group"].unique():
            data = shuffled[shuffled["group"] == group][col]
            plt.hist(data, alpha=0.5, label=f"Group {group}", bins=20)
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.title(f"Histogram of {col} by Group")
        plt.legend()
        # save to outputs directory
        outputs_dir = project_root / "outputs" / "histograms for sanity check"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(outputs_dir / f"{col}_histogram.png")
        plt.close()


def main() -> None:
    df = generate_dataset()
    split_and_save(df)


if __name__ == "__main__":
    main()