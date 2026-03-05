import sys
from pathlib import Path

import torch

AI_ROOT = Path(__file__).resolve().parents[4]
if str(AI_ROOT) not in sys.path:
    sys.path.append(str(AI_ROOT))

from ml.frameworks.pytorch import models


def test_build_model_respects_mode_and_shape():
    model = models.build_model(4, 2, "linear_glm_baseline", 16, 2, 0.1)
    out = model(torch.zeros((3, 4), dtype=torch.float32))
    assert tuple(out.shape) == (3, 2)


def test_compute_class_weights_returns_per_class_weights():
    y = torch.tensor([0, 0, 1, 2], dtype=torch.long)
    weights = models.compute_class_weights(y, output_dim=3, device=torch.device("cpu"))
    assert tuple(weights.shape) == (3,)


def test_model_introspection_helpers_match_model_type():
    model = models.build_model(4, 2, "tabresnet", 32, 3, 0.2)
    assert models.model_training_mode(model) == "tabresnet"
    assert models.model_hidden_dim(model) == 32
    assert models.model_num_hidden_layers(model) >= 1
    assert models.model_dropout(model) == 0.2


def test_build_model_defaults_to_mlp_for_non_special_mode():
    model = models.build_model(4, 1, "mlp_dense", 16, 0, 0.0)
    out = model(torch.zeros((3, 4), dtype=torch.float32))
    assert isinstance(model, models.MLP)
    assert tuple(out.shape) == (3, 1)
    assert models.model_hidden_dim(model) == 16
    assert models.model_num_hidden_layers(model) == 1
    assert models.model_dropout(model) == 0.0
    assert models.model_training_mode(model) == "mlp_dense"


def test_residual_block_preserves_hidden_shape():
    block = models.ResidualBlock(hidden_dim=8, dropout=0.0)
    out = block(torch.ones((3, 8), dtype=torch.float32))
    assert tuple(out.shape) == (3, 8)


def test_compute_loss_uses_model_outputs_with_supplied_criterion():
    model = models.build_model(4, 1, "linear_glm_baseline", 16, 1, 0.0)
    x = torch.zeros((3, 4), dtype=torch.float32)
    y = torch.zeros((3, 1), dtype=torch.float32)
    loss = models.compute_loss(model, x, y, torch.nn.MSELoss())
    assert float(loss.item()) >= 0.0


def test_model_introspection_fallbacks_for_unknown_module():
    class _Unknown(torch.nn.Module):
        def forward(self, x):
            return x

    model = _Unknown()
    assert models.model_hidden_dim(model) == 128
    assert models.model_num_hidden_layers(model) == 2
    assert models.model_dropout(model) == 0.0
    assert models.model_training_mode(model) == "mlp_dense"
