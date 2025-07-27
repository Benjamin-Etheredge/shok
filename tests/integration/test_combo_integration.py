"""Integration test for Fast R-CNN combination modules with real models."""

import pytest
import torch
import torchvision.models.detection as detection

from shok.utils.models.combo import FastRCNNConsensus, FastRCNNEnsemble, FastRCNNWeightedEnsemble


@pytest.mark.integration
def test_real_model_ensemble():
    """Test ensemble with real Fast R-CNN models."""
    # Create small models for testing
    model1 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model2 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)

    # Create ensemble
    ensemble = FastRCNNEnsemble([model1, model2])

    # Test forward pass
    dummy_image = torch.randn(3, 224, 224)
    images = [dummy_image]

    with torch.no_grad():
        predictions = ensemble(images)

    # Verify output format
    assert len(predictions) == 1
    assert "boxes" in predictions[0]
    assert "scores" in predictions[0]
    assert "labels" in predictions[0]

    # Verify tensor shapes
    assert predictions[0]["boxes"].shape[1] == 4  # x1, y1, x2, y2
    assert predictions[0]["scores"].shape[0] == predictions[0]["labels"].shape[0]
    assert predictions[0]["boxes"].shape[0] == predictions[0]["labels"].shape[0]


@pytest.mark.integration
def test_weighted_ensemble_optimization():
    """Test that weighted ensemble weights can be optimized."""
    # Create models
    model1 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model2 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)

    # Create weighted ensemble
    ensemble = FastRCNNWeightedEnsemble([model1, model2], learn_weights=True)

    # Get initial weights
    initial_weights = ensemble.weights.clone()

    # Create optimizer
    optimizer = torch.optim.SGD(ensemble.parameters(), lr=0.1)

    # Dummy training step
    dummy_image = torch.randn(3, 224, 224)
    images = [dummy_image]

    # Simple dummy loss calculation
    predictions = ensemble(images)
    loss = predictions[0]["scores"].sum()  # Dummy loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Check that weights changed
    new_weights = ensemble.weights
    assert not torch.allclose(initial_weights, new_weights, atol=1e-6)


@pytest.mark.integration
def test_consensus_ensemble():
    """Test consensus ensemble with real models."""
    # Create models
    model1 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)
    model2 = detection.fasterrcnn_resnet50_fpn(weights=None, num_classes=2)

    # Create consensus ensemble
    ensemble = FastRCNNConsensus([model1, model2], min_consensus=1)

    # Test forward pass
    dummy_image = torch.randn(3, 224, 224)
    images = [dummy_image]

    with torch.no_grad():
        predictions = ensemble(images)

    # Verify output format
    assert len(predictions) == 1
    assert "boxes" in predictions[0]
    assert "scores" in predictions[0]
    assert "labels" in predictions[0]


if __name__ == "__main__":
    # Run tests manually
    test_real_model_ensemble()
    test_weighted_ensemble_optimization()
    test_consensus_ensemble()
    print("All integration tests passed!")
