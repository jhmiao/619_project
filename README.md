# Incentive Allocation: PTO and DFL

This project simulates a treatment allocation problem and compares predict-then-optimize (PTO) against several decision-focused learning (DFL) variants. There are two allocation settings:

- **Cardinality / nudging:** allocate to a fixed number of users.
- **Payment / knapsack:** allocate under an expected payment budget, where payment is triggered by crossing a wear threshold.

Run commands from the `project/` directory.

## Pipeline

```bash
python src/generate_data.py
python src/train_pto.py
python src/train_dfl.py
python src/evaluate.py

python src/train_dfl_payment.py --train-all-variants
python src/evaluate_allocation.py
```

## Data

`src/generate_data.py` writes:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Each row contains group label, five baseline covariates, 30 pre-intervention wear indicators, observed action `A_obs`, observed outcome `Y_obs`, and true potential outcomes `Y0_true`, `Y1_true`.

## Shared Model

All learned methods use the same feature vector:

```text
[group_B, x1, ..., x5, pre_1, ..., pre_30, A_obs]
```

This gives 37 input features. The shared architecture is a two-hidden-layer MLP:

```text
Linear(input_dim, 64)
ReLU
Dropout(0.1)
Linear(64, 64)
ReLU
Dropout(0.1)
Linear(64, 1)
```

`OutcomeNet` includes a final sigmoid and is used by PTO. `OutcomeNetNoSigmoid` omits the final sigmoid and is used by DFL-family methods, which apply `torch.sigmoid` explicitly.

## Training

### PTO

`src/train_pto.py` trains a supervised outcome model using MSE on `Y_obs`. It uses 5-fold cross-validation to select the number of epochs, then retrains a final model on all training data.

Outputs:

- `outputs/pto_model.pt`
- `outputs/pto_training_summary.csv`

### Cardinality DFL

`src/train_dfl.py` trains:

- `dfl`
- `rs`
- `pg:forward`
- `pg:backward`
- `pg:central`

The default `MODEL_VARIANTS` and `PG_ESTIMATORS` run all of these combinations:

```bash
python src/train_dfl.py
```

Each variant uses 5-fold cross-validation to select the epoch count, then retrains a final model on all training data.

Main outputs:

- `outputs/dfl_model.pt`
- `outputs/rs_model.pt`
- `outputs/pg_forward_model.pt`
- `outputs/pg_backward_model.pt`
- `outputs/pg_central_model.pt`
- matching `*_training_summary.csv` files

### Payment-Budget DFL

`src/train_dfl_payment.py` trains the same DFL variants under an expected payment budget. Payment cost is modeled as:

```text
C_i = alpha * Y_i * 1{Y_i >= T}
```

Run all payment variants with:

```bash
python src/train_dfl_payment.py --train-all-variants
```

Like `train_dfl.py`, each payment variant uses 5-fold cross-validation for epoch selection and then retrains on all training data.

Main outputs:

- `outputs/payment_dfl_model.pt`
- `outputs/payment_rs_model.pt`
- `outputs/payment_pg_forward_model.pt`
- `outputs/payment_pg_backward_model.pt`
- `outputs/payment_pg_central_model.pt`
- matching `payment_*_training_summary.csv` files

## Evaluation

### Cardinality Evaluation

`src/evaluate.py` compares PTO, DFL, RS, PG variants, oracle, random, and no-action policies on `data/test.csv`. It predicts counterfactual outcomes, solves the allocation rule, and appends metrics to:

- `outputs/test_metrics.csv`

It also writes the latest allocation table to:

- `outputs/test_allocations.csv`

Example budget sweep:

```bash
for b in 7 15 75 150; do
  python src/evaluate.py \
    --budget "$b" \
    --metrics-path outputs/test_metrics_budget_sweep.csv \
    --allocation-path "outputs/test_allocations_budget_${b}.csv"
done
```

### Payment Evaluation

`src/evaluate_allocation.py` loads the payment-trained DFL models and records realized payment budget. It reports both:

- `budget_used`: number of selected users
- `actual_payment_budget_used`: realized payment cost, computed from true outcomes as `A_i * Y1_true_i * 1{Y1_true_i >= T}`

It appends metrics to `--metrics-path` and reconciles columns if older metric files do not yet contain newer fields.

## Main Files

- `src/generate_data.py`: simulate users, actions, observed outcomes, and potential outcomes
- `src/models.py`: shared MLP architectures
- `src/train_pto.py`: supervised PTO outcome model
- `src/train_dfl.py`: cardinality-budget DFL, RS, and PG training
- `src/train_dfl_payment.py`: payment-budget DFL, RS, and PG training
- `src/optimize.py`: top-B, fairness-aware, and payment-budget allocation utilities
- `src/evaluate.py`: cardinality-policy evaluation
- `src/evaluate_allocation.py`: payment-policy evaluation
