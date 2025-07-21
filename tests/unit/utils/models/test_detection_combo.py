"""Test module for differentiable detection combination utilities."""

import pytest
import torch
import torch.nn as nn

from shok.utils.models.detection_combo import (
    AdaptiveDetectionEnsemble,
    DifferentiableDetectionEnsemble,
    HierarchicalDetectionEnsemble,
)


class MockDetectionModel(torch.nn.Module):
    """Mock detection model for testing."""

    def __init__(self, predictions=None, losses=None):
        super().__init__()
        self.predictions = predictions or []
        self.losses = losses or {}
        self.backbone = nn.Conv2d(3, 256, 3, padding=1)  # Mock backbone

    def forward(self, images, targets=None):
        if self.training and targets is not None:
            return self.losses
        return self.predictions


class TestDifferentiableDetectionEnsemble:
    """Test DifferentiableDetectionEnsemble class."""

    def test_init_basic(self):
        """Test basic initialization."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models)

        assert len(ensemble.models) == 2
        assert ensemble.combination_method == "soft_nms"
        assert ensemble.num_models == 2
        assert isinstance(ensemble.model_weights, nn.Parameter)

    def test_init_empty_models(self):
        """Test initialization with empty models list."""
        with pytest.raises(ValueError, match="At least one model must be provided"):
            DifferentiableDetectionEnsemble([])

    def test_init_with_attention(self):
        """Test initialization with attention mechanism."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models, combination_method="attention")

        assert hasattr(ensemble, "attention")
        assert hasattr(ensemble, "feature_projection")

    def test_normalized_weights_property(self):
        """Test normalized weights property."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models)

        weights = ensemble.normalized_weights
        assert torch.allclose(weights.sum(), torch.tensor(1.0))
        assert weights.shape == (2,)

    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        losses = {
            "loss_classifier": torch.tensor(0.5),
            "loss_box_reg": torch.tensor(0.3),
            "loss_objectness": torch.tensor(0.2),
            "loss_rpn_box_reg": torch.tensor(0.1),
        }

        models = [MockDetectionModel(losses=losses), MockDetectionModel(losses=losses)]
        ensemble = DifferentiableDetectionEnsemble(models)
        ensemble.train()

        images = [torch.randn(3, 224, 224)]
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([1])}]

        result = ensemble(images, targets)

        assert isinstance(result, dict)
        assert "loss_classifier" in result
        assert all(isinstance(v, torch.Tensor) for v in result.values())

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        pred = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]

        models = [MockDetectionModel(predictions=pred), MockDetectionModel(predictions=pred)]
        ensemble = DifferentiableDetectionEnsemble(models)
        ensemble.eval()

        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "boxes" in result[0]
        assert "scores" in result[0]
        assert "labels" in result[0]

    def test_forward_eval_empty_predictions(self):
        """Test forward pass with empty predictions."""
        empty_pred = [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
            }
        ]

        models = [MockDetectionModel(predictions=empty_pred), MockDetectionModel(predictions=empty_pred)]
        ensemble = DifferentiableDetectionEnsemble(models)
        ensemble.eval()

        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["boxes"].shape == (0, 4)
        assert result[0]["scores"].shape == (0,)
        assert result[0]["labels"].shape == (0,)

    def test_set_combination_method(self):
        """Test setting combination method."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models)

        ensemble.set_combination_method("weighted_avg")
        assert ensemble.combination_method == "weighted_avg"

        with pytest.raises(ValueError):
            ensemble.set_combination_method("invalid_method")

    def test_get_model_weights(self):
        """Test getting model weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models)

        weights = ensemble.get_model_weights()
        assert isinstance(weights, torch.Tensor)
        assert weights.shape == (2,)
        assert torch.allclose(weights.sum(), torch.tensor(1.0))

    def test_freeze_models(self):
        """Test freezing constituent models."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = DifferentiableDetectionEnsemble(models, freeze_models=True)

        # Check that model parameters are frozen
        for model in ensemble.models:
            for param in model.parameters():
                assert not param.requires_grad


class TestAdaptiveDetectionEnsemble:
    """Test AdaptiveDetectionEnsemble class."""

    def test_init_basic(self):
        """Test basic initialization."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = AdaptiveDetectionEnsemble(models)

        assert len(ensemble.models) == 2
        assert ensemble.num_models == 2
        assert hasattr(ensemble, "router")

    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        losses = {"loss_classifier": torch.tensor(0.5), "loss_box_reg": torch.tensor(0.3)}

        models = [MockDetectionModel(losses=losses), MockDetectionModel(losses=losses)]
        ensemble = AdaptiveDetectionEnsemble(models)
        ensemble.train()

        images = [torch.randn(3, 224, 224)]
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([1])}]

        result = ensemble(images, targets)

        assert isinstance(result, dict)
        assert "loss_classifier" in result

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        pred = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]

        models = [MockDetectionModel(predictions=pred), MockDetectionModel(predictions=pred)]
        ensemble = AdaptiveDetectionEnsemble(models)
        ensemble.eval()

        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "boxes" in result[0]

    def test_get_routing_weights(self):
        """Test getting routing weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = AdaptiveDetectionEnsemble(models)

        images = [torch.randn(3, 224, 224)]
        weights = ensemble.get_routing_weights(images)

        assert weights.shape == (1, 2)  # batch_size=1, num_models=2
        assert torch.allclose(weights.sum(dim=1), torch.tensor([1.0]))


class TestHierarchicalDetectionEnsemble:
    """Test HierarchicalDetectionEnsemble class."""

    def test_init_basic(self):
        """Test basic initialization."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = HierarchicalDetectionEnsemble(models)

        assert len(ensemble.models) == 2
        assert len(ensemble.scale_factors) == 2
        assert all(sf == 1.0 for sf in ensemble.scale_factors)

    def test_init_with_scale_factors(self):
        """Test initialization with custom scale factors."""
        models = [MockDetectionModel(), MockDetectionModel()]
        scale_factors = [0.5, 1.5]
        ensemble = HierarchicalDetectionEnsemble(models, scale_factors=scale_factors)

        assert ensemble.scale_factors == scale_factors

    def test_forward_training_mode(self):
        """Test forward pass in training mode."""
        losses = {"loss_classifier": torch.tensor(0.5), "loss_box_reg": torch.tensor(0.3)}

        models = [MockDetectionModel(losses=losses), MockDetectionModel(losses=losses)]
        ensemble = HierarchicalDetectionEnsemble(models)
        ensemble.train()

        images = [torch.randn(3, 224, 224)]
        targets = [{"boxes": torch.tensor([[10, 10, 50, 50]]), "labels": torch.tensor([1])}]

        result = ensemble(images, targets)

        assert isinstance(result, dict)
        assert "loss_classifier" in result

    def test_forward_eval_mode(self):
        """Test forward pass in evaluation mode."""
        pred = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]

        models = [MockDetectionModel(predictions=pred), MockDetectionModel(predictions=pred)]
        ensemble = HierarchicalDetectionEnsemble(models)
        ensemble.eval()

        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "boxes" in result[0]

    def test_scale_aware_fusion(self):
        """Test scale-aware fusion method."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = HierarchicalDetectionEnsemble(models, fusion_method="scale_aware")

        assert ensemble.fusion_method == "scale_aware"
        assert hasattr(ensemble, "scale_fusion")

    def test_forward_with_different_scales(self):
        """Test forward pass with different scale factors."""
        pred = [
            {
                "boxes": torch.tensor([[10, 10, 50, 50]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]

        models = [MockDetectionModel(predictions=pred), MockDetectionModel(predictions=pred)]
        scale_factors = [0.5, 1.0]
        ensemble = HierarchicalDetectionEnsemble(models, scale_factors=scale_factors)
        ensemble.eval()

        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert isinstance(result, list)
        assert len(result) == 1
        assert "boxes" in result[0]


if __name__ == "__main__":
    pytest.main([__file__])
