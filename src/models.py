

"""
Neural network models for PTO/DFL.

We use a shared OutcomeNet that takes features:
[group_B, x1..x5, pre_1..pre_30, A]
and outputs predicted Y in [0, 1].
"""

import torch
import torch.nn as nn


class OutcomeNet(nn.Module):
    """Simple 2-layer MLP for outcome prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # Y is proportion in [0,1]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: shape (batch_size, input_dim)

        Returns:
            y_hat: shape (batch_size,)
        """
        return self.net(x).squeeze(-1)


class OutcomeNetNoSigmoid(nn.Module):
    """Same architecture but without final sigmoid (optional for DFL)."""

    def __init__(self, input_dim: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def count_parameters(model: nn.Module) -> int:
    """Utility to count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # quick sanity check
    input_dim = 37
    model = OutcomeNet(input_dim)
    x = torch.randn(4, input_dim)
    y = model(x)
    print("Output shape:", y.shape)
    print("Num params:", count_parameters(model))
