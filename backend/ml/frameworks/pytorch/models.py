from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

TrainingMode = Literal[
    "mlp_dense",
    "linear_glm_baseline",
    "tabresnet",
    "imbalance_aware",
    "calibrated_classifier",
    "tree_teacher_distillation",
]


class MLP(nn.Module):
    """Plain dense network used for the default PyTorch tabular modes.

    Layout:
    - repeated hidden blocks: Linear -> BatchNorm -> ReLU -> optional Dropout
    - final linear head that maps hidden features to output logits/predictions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []

        in_features = input_dim
        for _ in range(max(1, num_hidden_layers)):
            # Affine projection expands raw tabular features into hidden space.
            layers.append(nn.Linear(in_features, hidden_dim))
            # BatchNorm stabilizes feature scale between blocks.
            layers.append(nn.BatchNorm1d(hidden_dim))
            # ReLU provides the non-linearity that lets the network model interactions.
            layers.append(nn.ReLU())
            if dropout > 0:
                # Dropout is optional regularization; omit it entirely when p == 0.
                layers.append(nn.Dropout(dropout))
            in_features = hidden_dim

        # Final head emits class logits or regression values.
        layers.append(nn.Linear(in_features, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LinearBaseline(nn.Module):
    """Single affine layer used as the simplest interpretable baseline."""

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """TabResNet hidden block with an identity skip connection.

    The skip path preserves the incoming representation so deeper stacks can
    learn refinements instead of relearning the full mapping each time.
    """

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save the incoming activation for the residual addition.
        identity = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = torch.relu(out)
        return out


class TabResNet(nn.Module):
    """Residual MLP for tabular data.

    Stages:
    - input projection lifts raw features into a fixed hidden width
    - residual block stack repeatedly refines the hidden representation
    - output head maps the refined hidden state to logits/predictions
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        num_hidden_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.input_bn = nn.BatchNorm1d(hidden_dim)
        blocks = [ResidualBlock(hidden_dim, dropout) for _ in range(max(1, num_hidden_layers))]
        self.blocks = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project raw features once before the residual stack.
        out = self.input(x)
        out = self.input_bn(out)
        out = torch.relu(out)
        # Residual blocks preserve gradient flow in deeper tabular networks.
        out = self.blocks(out)
        # Head converts hidden features into the task output dimension.
        out = self.head(out)
        return out


def build_model(
    input_dim: int,
    output_dim: int,
    training_mode: str,
    hidden_dim: int,
    num_hidden_layers: int,
    dropout: float,
) -> nn.Module:
    """Route a training mode to its concrete architecture implementation."""
    if training_mode == "linear_glm_baseline":
        return LinearBaseline(input_dim=input_dim, output_dim=output_dim)
    if training_mode == "tabresnet":
        return TabResNet(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=dropout,
        )
    return MLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
        num_hidden_layers=num_hidden_layers,
        dropout=dropout,
    )


def compute_loss(model: nn.Module, x: torch.Tensor, y: torch.Tensor, criterion: nn.Module) -> torch.Tensor:
    """Run one forward pass and delegate loss math to the supplied criterion."""
    logits = model(x)
    return criterion(logits, y)


def compute_class_weights(y_train: torch.Tensor, output_dim: int, device: torch.device) -> torch.Tensor:
    """Inverse-frequency class weighting for imbalanced classification."""
    counts = torch.bincount(y_train.detach().cpu(), minlength=output_dim).float()
    safe_counts = torch.clamp(counts, min=1.0)
    total = float(torch.sum(safe_counts).item())
    weights = total / (float(output_dim) * safe_counts)
    return weights.to(device)


def model_hidden_dim(model: nn.Module) -> int:
    """Infer the hidden width used by the current model instance."""
    if isinstance(model, LinearBaseline):
        return model.net.out_features
    if isinstance(model, TabResNet):
        return model.input.out_features
    for module in model.modules():
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    return 128


def model_num_hidden_layers(model: nn.Module) -> int:
    """Infer how many hidden transformations the model contains."""
    if isinstance(model, LinearBaseline):
        return 1
    if isinstance(model, TabResNet):
        return max(1, len(model.blocks))
    if isinstance(model, MLP):
        return max(1, sum(1 for layer in model.net if isinstance(layer, nn.Linear)) - 1)
    return 2


def model_dropout(model: nn.Module) -> float:
    """Infer the first configured dropout probability, if any."""
    if isinstance(model, LinearBaseline):
        return 0.0
    if isinstance(model, TabResNet):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                return float(module.p)
        return 0.0
    if isinstance(model, MLP):
        for layer in model.net:
            if isinstance(layer, nn.Dropout):
                return float(layer.p)
    return 0.0


def model_training_mode(model: nn.Module) -> TrainingMode:
    """Map a live model instance back to its persisted training mode label."""
    if isinstance(model, LinearBaseline):
        return "linear_glm_baseline"
    if isinstance(model, TabResNet):
        return "tabresnet"
    return "mlp_dense"
