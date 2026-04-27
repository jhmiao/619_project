# PTO Fitbit Wear Simulation Project

This project simulates Fitbit wear behavior and compares ranking policies for deciding which users should receive an intervention. The main target is the treatment effect of action `1` versus action `0`: the difference in total wear days over the next 30 days.

## Data Format

Most generated CSVs use one row per user:

```text
user_id, group, action, day_1, ..., day_30, p01_0, p11_0, tau01, tau11, p01_1, p11_1
```

`day_i` is a binary wear indicator:

- `0`: below wear threshold
- `1`: above wear threshold

The PTO training file uses:

```text
user_id, day_1, ..., day_30, target
```

where:

```text
target = sum(next-month days under action=1) - sum(next-month days under action=0)
```

## Scripts

### `generate_data.py`

Generates synthetic Fitbit wear data from latent user groups. Each user is assigned a group and person-level transition parameters:

- `p01_0`: probability of moving from not-wear to wear without action
- `p11_0`: probability of staying in wear without action
- `tau01`: action lift for `0 -> 1`
- `tau11`: action lift for `1 -> 1`
- `p01_1`: action transition probability for `0 -> 1`
- `p11_1`: action transition probability for `1 -> 1`

Important functions:

- `generate_dataset(...)`: creates the first month of simulated data.
- `generate_next_month(...)`: reads a CSV with a `day_31` state and simulates the next 30 days under a chosen action.

Running the script writes `synthetic_fitbit_wear.csv`.

### `pto_training_data.py`

Builds the supervised learning dataset for PTO training.

It reads:

- `synthetic_fitbit_wear_test.csv`
- `synthetic_fitbit_wear_next_month_0_test.csv`
- `synthetic_fitbit_wear_next_month_1_test.csv`

For each `user_id`, it computes:

```text
target = action_1_next_month_day_sum - action_0_next_month_day_sum
```

It writes `pto_training_data_test.csv`.

### `train_pto.py`

Trains a predict-then-optimize style neural network to predict `target` from the 30-day wear history.

Model architecture:

```text
30 inputs -> Linear(32) -> ReLU -> Linear(16) -> ReLU -> Linear(1)
```

It performs K-fold cross-validation, trains a final model, and saves:

- `pto_model.pt`
- `pto_cv_results.csv`

### `train_pair_rank.py`

Trains a pairwise ranking model using the same neural network architecture as `train_pto.py`, but with a RankNet-style pairwise loss.

The model learns a scalar score intended to rank users by treatment effect, rather than directly minimizing prediction error.

It saves artifacts under `outputs_pair_rank/`, including:

- `pair_rank_model.pt`
- `pair_rank_scaler.npz`
- `pair_rank_metrics.json`
- `pair_rank_training_predictions.csv`
- `pair_rank_cv_results.csv`

### `evaluate.py`

Evaluates either model type on PTO-format data.

It supports:

- PTO model checkpoints such as `pto_model.pt`
- Pair-rank checkpoints such as `outputs_pair_rank/pair_rank_model.pt`

For PTO models, it writes predictions with column:

```text
pto_pred
```

For pair-rank models, it writes scores with column:

```text
pair_rank_score
```

Example commands:

```bash
python evaluate.py --model-path pto_model.pt --output-path pto_results.csv
python evaluate.py --model-path outputs_pair_rank/pair_rank_model.pt --output-path pair_rank_results.csv
```

Pair-rank evaluation also loads the saved scaler from `outputs_pair_rank/pair_rank_scaler.npz`.

### `compare_results.py`

Compares ranking policies against an oracle ranking.

It reads:

- `results_test.csv` or PTO results with `pto_pred`
- `pair_rank_results.csv` with `pair_rank_score`

For every top `K`, it computes:

```text
regret = oracle top-K target sum - model top-K target sum
```

It compares:

- PTO ranking
- Pair-rank ranking
- Random baseline

It writes:

- `compare_results_curve.csv`
- `compare_results.png`

## Typical Workflow

From the `project/` directory:

```bash
python generate_data.py
python pto_training_data.py

python train_pto.py
python train_pair_rank.py

python evaluate.py --model-path pto_model.pt --output-path pto_results.csv
python evaluate.py --model-path outputs_pair_rank/pair_rank_model.pt --output-path pair_rank_results.csv

python compare_results.py
```

The final comparison plot is `compare_results.png`.

## Notes

- The simulation is synthetic and uses Markov transition probabilities.
- Person-level randomization means each user has fixed transition parameters across days, but different users have slightly different parameters.
- `target` is known only because this is synthetic data and both potential next-month outcomes are simulated.
- The oracle curve ranks by the true `target`, so it is an upper bound for learned ranking methods.
