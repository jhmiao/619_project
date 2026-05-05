# Incentive Allocation: PTO and DFL

This project simulates a treatment allocation problem and compares two neural
approaches:

- **PTO**, predict-then-optimize: learn an outcome model first, then solve a
  top-budget allocation problem using predicted treatment effects.
- **DFL**, decision-focused learning: train the same outcome architecture through
  a differentiable surrogate of the downstream allocation objective.

The intended pipeline is:

```bash
python src/generate_data.py
python src/train_pto.py
python src/train_dfl.py
python src/evaluate.py
```

Run these commands from the `project/` directory.

## Data Generation

`src/generate_data.py` creates a simulated user population and writes:

- `data/train.csv`
- `data/val.csv`
- `data/test.csv`

Each user has:

- group label `A` or `B`
- five covariates `x1` through `x5`
- 30 pre-intervention binary wear indicators `pre_1` through `pre_30`
- randomized observed treatment `A_obs`
- observed outcome `Y_obs`
- true potential outcomes `Y0_true` and `Y1_true`

The post-intervention outcomes are generated from binary Markov chains. The
baseline transition probabilities are heterogeneous by covariates and group, and
treatment changes those transition probabilities through user-specific response
parameters. The outcome `Y` is the average post-intervention wear rate, so
`Y0_true` is the no-action outcome and `Y1_true` is the action outcome.

## Model

`src/models.py` defines the shared neural network structure used by PTO and DFL.
The input feature vector is:

```text
[group_B, x1, ..., x5, pre_1, ..., pre_30, A_obs]
```

This gives an input dimension of 37.

The shared architecture is a two-hidden-layer MLP:

```text
Linear(input_dim, hidden_dim)
ReLU
Dropout
Linear(hidden_dim, hidden_dim)
ReLU
Dropout
Linear(hidden_dim, 1)
```

`OutcomeNet` adds a final sigmoid and is used by PTO. `OutcomeNetNoSigmoid` uses
the same structure without the final sigmoid and is used by DFL; DFL applies
`torch.sigmoid` explicitly during training and evaluation.

## PTO: Predict Then Optimize

`src/train_pto.py` trains `OutcomeNet` to predict the observed outcome `Y_obs`
from the observed action and user features.

The supervised training objective is mean squared error:

```text
min_theta (1/n) sum_i (f_theta(x_i, A_obs_i) - Y_obs_i)^2
```

The script uses K-fold cross-validation to choose a reasonable number of epochs,
then retrains one final model on all training data. It writes:

- `outputs/pto_model.pt`
- `outputs/pto_training_summary.csv`

After training, PTO uses the model counterfactually:

```text
y0_hat_i = f_theta(x_i, A=0)
y1_hat_i = f_theta(x_i, A=1)
```

`src/optimize.py` then solves the budgeted allocation problem. For utility
function `u`, define each user's predicted gain:

```text
score_i = u(y1_hat_i) - u(y0_hat_i)
```

The default utility is the threshold success utility:

```text
u(y) = 1{y > 0.60}
```

The PTO allocation objective is:

```text
max_a sum_i a_i [u(y1_hat_i) - u(y0_hat_i)]
subject to sum_i a_i <= B
           a_i in {0, 1}
           a_i = 0 if score_i <= 0  (default require_positive_score=True)
```

Because the objective is linear in the binary allocation, the optimizer selects
the top `B` users with positive predicted scores. Ties are broken stably by user
index.

## DFL: Decision-Focused Learning

`src/train_dfl.py` trains `OutcomeNetNoSigmoid` with a differentiable surrogate
for the allocation objective.

For each mini-batch, the same model is evaluated twice:

```text
y0_hat_i = sigmoid(f_theta(x_i, A=0))
y1_hat_i = sigmoid(f_theta(x_i, A=1))
```

The hard threshold utility is replaced by a smooth approximation:

```text
s0_hat_i = sigmoid((y0_hat_i - threshold) / threshold_temperature)
s1_hat_i = sigmoid((y1_hat_i - threshold) / threshold_temperature)
score_i = s1_hat_i - s0_hat_i
```

The hard top-`B` allocation is replaced by a soft batch allocation:

```text
a_i = batch_budget * softmax(score_i / allocation_temperature)
batch_budget = max(1, budget_fraction * batch_size)
```

The differentiable DFL decision objective is:

```text
max_theta sum_i a_i [s1_hat_i - s0_hat_i]
```

The implemented loss minimizes the negative mean decision objective plus optional
regularizers:

```text
min_theta
    - (1/m) sum_i a_i [s1_hat_i - s0_hat_i]
    + mse_weight * (1/m) sum_i (sigmoid(f_theta(x_i, A_obs_i)) - Y_obs_i)^2
    + fairness_weight * sum_g (r_g - r_overall)^2
```

Here `m` is the mini-batch size and `r_g` is the predicted soft-policy success
rate for group `g`. The fairness term is available but defaults to zero.

The DFL script writes:

- `outputs/dfl_model.pt`
- `outputs/dfl_training_summary.csv`

## Evaluation

`src/evaluate.py` loads the PTO and DFL checkpoints, predicts counterfactual
outcomes on `data/test.csv`, solves the same top-budget allocation for each
model, and compares:

- `pto`
- `dfl`
- `oracle`
- `random`
- `no_action`

The oracle policy uses the true potential outcomes:

```text
score_i^oracle = u(Y1_true_i) - u(Y0_true_i)
```

Evaluation reports:

- budget used
- mean policy outcome
- policy success rate
- true continuous gain among selected users
- true threshold-success gain among selected users
- group-level counts, selection rates, and success rates

Outputs:

- `outputs/test_allocations.csv`
- `outputs/test_metrics.csv`

## Main Files

- `src/generate_data.py`: simulate users, potential outcomes, and train/val/test splits
- `src/models.py`: shared MLP architectures
- `src/train_pto.py`: supervised outcome-model training for PTO
- `src/optimize.py`: top-budget allocation utilities
- `src/train_dfl.py`: decision-focused training objective
- `src/evaluate.py`: test-set policy comparison
