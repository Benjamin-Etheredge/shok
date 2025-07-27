"""Test module for Fast R-CNN combination utilities."""

import pytest
import torch

from shok.utils.models.combo import FastRCNNConsensus, FastRCNNEnsemble, FastRCNNWeightedEnsemble


class MockDetectionModel(torch.nn.Module):
    """Mock detection model for testing."""

    def __init__(self, predictions=None):
        super().__init__()
        self.predictions = predictions or []

    def forward(self, images, targets=None):
        return self.predictions


class TestFastRCNNEnsemble:
    """Test FastRCNNEnsemble class."""

    def test_init_empty_models(self):
        """Test initialization with empty models list."""
        with pytest.raises(ValueError, match="At least one model must be provided"):
            FastRCNNEnsemble([])

    def test_init_default_weights(self):
        """Test initialization with default weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = FastRCNNEnsemble(models)

        assert ensemble.weights == [0.5, 0.5]
        assert ensemble.combination_method == "average"

    def test_init_custom_weights(self):
        """Test initialization with custom weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        weights = [0.3, 0.7]
        ensemble = FastRCNNEnsemble(models, weights=weights)

        assert ensemble.weights == weights

    def test_init_weights_normalization(self):
        """Test weight normalization."""
        models = [MockDetectionModel(), MockDetectionModel()]
        weights = [1.0, 3.0]  # Sum = 4.0
        ensemble = FastRCNNEnsemble(models, weights=weights)

        assert ensemble.weights == [0.25, 0.75]

    def test_init_mismatched_weights(self):
        """Test initialization with mismatched weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        weights = [0.5]  # Only one weight for two models

        with pytest.raises(ValueError, match="Number of weights must match number of models"):
            FastRCNNEnsemble(models, weights=weights)

    def test_forward_average(self):
        """Test forward pass with average combination."""
        # Create mock models with predictions
        pred1 = [
            {
                "boxes": torch.tensor([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=torch.float32),
                "scores": torch.tensor([0.9, 0.8], dtype=torch.float32),
                "labels": torch.tensor([1, 2], dtype=torch.long),
            }
        ]
        pred2 = [
            {
                "boxes": torch.tensor([[15, 25, 35, 45]], dtype=torch.float32),
                "scores": torch.tensor([0.7], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            }
        ]

        model1 = MockDetectionModel(pred1)
        model2 = MockDetectionModel(pred2)

        ensemble = FastRCNNEnsemble([model1, model2], combination_method="average")

        # Test forward pass
        images = [torch.randn(3, 224, 224)]
        result = ensemble(images)

        assert len(result) == 1
        assert "boxes" in result[0]
        assert "scores" in result[0]
        assert "labels" in result[0]

    def test_apply_nms_empty_boxes(self):
        """Test NMS with empty boxes."""
        models = [MockDetectionModel()]
        ensemble = FastRCNNEnsemble(models)

        empty_boxes = torch.empty((0, 4), dtype=torch.float32)
        empty_scores = torch.empty((0,), dtype=torch.float32)
        empty_labels = torch.empty((0,), dtype=torch.long)

        result = ensemble._apply_nms(empty_boxes, empty_scores, empty_labels)

        assert len(result) == 0
        assert result.dtype == torch.long

    def test_train_mode_override(self):
        """Test that models stay in eval mode during training."""
        model1 = MockDetectionModel()
        model2 = MockDetectionModel()

        ensemble = FastRCNNEnsemble([model1, model2])
        ensemble.train(True)

        # Models should be in eval mode
        assert not model1.training
        assert not model2.training


class TestFastRCNNWeightedEnsemble:
    """Test FastRCNNWeightedEnsemble class."""

    def test_init_learnable_weights(self):
        """Test initialization with learnable weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = FastRCNNWeightedEnsemble(models, learn_weights=True)

        assert isinstance(ensemble.raw_weights, torch.nn.Parameter)
        assert ensemble.raw_weights.requires_grad

    def test_init_fixed_weights(self):
        """Test initialization with fixed weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = FastRCNNWeightedEnsemble(models, learn_weights=False)

        assert not ensemble.raw_weights.requires_grad

    def test_weights_property(self):
        """Test weights property returns softmax normalized weights."""
        models = [MockDetectionModel(), MockDetectionModel()]
        ensemble = FastRCNNWeightedEnsemble(models, initial_weights=[1.0, 2.0])

        weights = ensemble.weights
        assert torch.allclose(weights, torch.softmax(torch.tensor([1.0, 2.0]), dim=0))

    def test_forward_empty_predictions(self):
        """Test forward pass with empty predictions."""
        empty_pred = [
            {
                "boxes": torch.empty((0, 4), dtype=torch.float32),
                "scores": torch.empty((0,), dtype=torch.float32),
                "labels": torch.empty((0,), dtype=torch.long),
            }
        ]

        model = MockDetectionModel(empty_pred)
        ensemble = FastRCNNWeightedEnsemble([model])
        images = [torch.randn(3, 224, 224)]

        result = ensemble(images)

        assert len(result) == 1
        assert result[0]["boxes"].shape == (0, 4)
        assert result[0]["scores"].shape == (0,)
        assert result[0]["labels"].shape == (0,)


class TestFastRCNNConsensus:
    """Test FastRCNNConsensus class."""

    def test_init_parameters(self):
        """Test initialization with custom parameters."""
        models = [MockDetectionModel(), MockDetectionModel()]
        consensus = FastRCNNConsensus(models, min_consensus=2, iou_threshold=0.6, score_aggregation="max")

        assert consensus.min_consensus == 2
        assert consensus.iou_threshold == 0.6
        assert consensus.score_aggregation == "max"

    def test_calculate_iou(self):
        """Test IoU calculation."""
        models = [MockDetectionModel()]
        consensus = FastRCNNConsensus(models)

        boxes1 = torch.tensor([[0, 0, 10, 10]], dtype=torch.float32)
        boxes2 = torch.tensor([[0, 0, 10, 10], [5, 5, 15, 15]], dtype=torch.float32)

        iou = consensus._calculate_iou(boxes1, boxes2)

        assert iou.shape == (2,)
        assert torch.allclose(iou[0], torch.tensor(1.0))  # Perfect overlap
        assert iou[1] > 0 and iou[1] < 1  # Partial overlap

    def test_find_consensus_empty(self):
        """Test consensus finding with empty predictions."""
        models = [MockDetectionModel()]
        consensus = FastRCNNConsensus(models)

        result = consensus._find_consensus([])

        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == (0,)
        assert result["labels"].shape == (0,)

    def test_find_consensus_no_matches(self):
        """Test consensus finding with no matching detections."""
        models = [MockDetectionModel()]
        consensus = FastRCNNConsensus(models, min_consensus=2)

        predictions = [
            {
                "boxes": torch.tensor([[0, 0, 10, 10]], dtype=torch.float32),
                "scores": torch.tensor([0.9], dtype=torch.float32),
                "labels": torch.tensor([1], dtype=torch.long),
            },
            {
                "boxes": torch.tensor([[50, 50, 60, 60]], dtype=torch.float32),
                "scores": torch.tensor([0.8], dtype=torch.float32),
                "labels": torch.tensor([2], dtype=torch.long),
            },
        ]

        result = consensus._find_consensus(predictions)

        # Should have no consensus detections
        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == (0,)
        assert result["labels"].shape == (0,)
