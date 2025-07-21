"""
Combination modules for torchvision Fast R-CNN models.

This module provides utilities for combining outputs from multiple Fast R-CNN models,
including ensemble methods and weighted combination strategies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import nms


class FastRCNNEnsemble(nn.Module):
    """
    Ensemble module that combines outputs from multiple Fast R-CNN models.

    This module runs multiple Fast R-CNN models in parallel and combines their outputs
    using various strategies like averaging, weighted combination, or non-maximum suppression.

    Args:
        models: List of Fast R-CNN models to ensemble
        combination_method: Method for combining outputs ('average', 'weighted', 'nms')
        weights: Optional weights for each model (used with 'weighted' method)
        nms_threshold: IoU threshold for NMS when using 'nms' method
        score_threshold: Minimum score threshold for detections
        max_detections: Maximum number of detections to keep per image

    """

    def __init__(
        self,
        models: list[nn.Module],
        combination_method: str = "average",
        weights: list[float] | None = None,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.1,
        max_detections: int = 100,
    ):
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.models = nn.ModuleList(models)
        self.combination_method = combination_method
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections

        # Set up weights for weighted combination
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            if len(weights) != len(models):
                raise ValueError("Number of weights must match number of models")
            # Normalize weights to sum to 1
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]

        # Register weights as buffer so they're saved with the model
        self.register_buffer("model_weights", torch.tensor(self.weights))

        # Set all models to eval mode by default
        for model in self.models:
            model.eval()

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Forward pass through the ensemble.

        Args:
            images: List of input images
            targets: Optional list of target dictionaries (for training)

        Returns:
            List of prediction dictionaries, one per image

        """
        # Get predictions from all models
        all_predictions = []

        for model in self.models:
            with torch.no_grad():
                predictions = model(images, targets)
                all_predictions.append(predictions)

        # Combine predictions based on the specified method
        if self.combination_method == "average":
            return self._average_predictions(all_predictions)
        elif self.combination_method == "weighted":
            return self._weighted_predictions(all_predictions)
        elif self.combination_method == "nms":
            return self._nms_predictions(all_predictions)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _average_predictions(
        self, all_predictions: list[list[dict[str, torch.Tensor]]]
    ) -> list[dict[str, torch.Tensor]]:
        """Average predictions from multiple models."""
        num_images = len(all_predictions[0])
        combined_predictions = []

        for img_idx in range(num_images):
            # Collect all predictions for this image
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Combine boxes, scores, and labels
            all_boxes = torch.cat([pred["boxes"] for pred in img_predictions], dim=0)
            all_scores = torch.cat([pred["scores"] for pred in img_predictions], dim=0)
            all_labels = torch.cat([pred["labels"] for pred in img_predictions], dim=0)

            # Apply NMS to remove duplicates
            keep_indices = self._apply_nms(all_boxes, all_scores, all_labels)

            combined_predictions.append(
                {
                    "boxes": all_boxes[keep_indices],
                    "scores": all_scores[keep_indices],
                    "labels": all_labels[keep_indices],
                }
            )

        return combined_predictions

    def _weighted_predictions(
        self, all_predictions: list[list[dict[str, torch.Tensor]]]
    ) -> list[dict[str, torch.Tensor]]:
        """Combine predictions using weighted averaging."""
        num_images = len(all_predictions[0])
        combined_predictions = []

        for img_idx in range(num_images):
            # Collect all predictions for this image with weights
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Weight the scores
            weighted_boxes = []
            weighted_scores = []
            weighted_labels = []

            for pred, weight in zip(img_predictions, self.weights, strict=False):
                weighted_boxes.append(pred["boxes"])
                weighted_scores.append(pred["scores"] * weight)
                weighted_labels.append(pred["labels"])

            # Combine all weighted predictions
            all_boxes = torch.cat(weighted_boxes, dim=0)
            all_scores = torch.cat(weighted_scores, dim=0)
            all_labels = torch.cat(weighted_labels, dim=0)

            # Apply NMS to remove duplicates
            keep_indices = self._apply_nms(all_boxes, all_scores, all_labels)

            combined_predictions.append(
                {
                    "boxes": all_boxes[keep_indices],
                    "scores": all_scores[keep_indices],
                    "labels": all_labels[keep_indices],
                }
            )

        return combined_predictions

    def _nms_predictions(self, all_predictions: list[list[dict[str, torch.Tensor]]]) -> list[dict[str, torch.Tensor]]:
        """Combine predictions by concatenating and applying NMS."""
        return self._average_predictions(all_predictions)  # Same as average method

    def _apply_nms(self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply Non-Maximum Suppression to remove duplicate detections."""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        # Filter by score threshold
        score_mask = scores > self.score_threshold
        if not score_mask.any():
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        boxes = boxes[score_mask]
        scores = scores[score_mask]
        labels = labels[score_mask]

        # Apply NMS per class
        keep_indices = []
        unique_labels = torch.unique(labels)

        for label in unique_labels:
            label_mask = labels == label
            if not label_mask.any():
                continue

            label_boxes = boxes[label_mask]
            label_scores = scores[label_mask]

            # Apply NMS for this class
            keep = nms(label_boxes, label_scores, self.nms_threshold)

            # Map back to original indices
            original_indices = torch.nonzero(label_mask, as_tuple=False).squeeze(1)
            keep_indices.extend(original_indices[keep].tolist())

        # Sort by score and limit number of detections
        keep_indices = torch.tensor(keep_indices, dtype=torch.long, device=boxes.device)
        if len(keep_indices) > self.max_detections:
            sorted_indices = torch.argsort(scores[keep_indices], descending=True)
            keep_indices = keep_indices[sorted_indices[: self.max_detections]]

        return keep_indices

    def train(self, mode: bool = True):
        """Override train to keep models in eval mode during training."""
        super().train(mode)
        # Keep all models in eval mode
        for model in self.models:
            model.eval()
        return self


class FastRCNNWeightedEnsemble(nn.Module):
    """
    Learnable weighted ensemble of Fast R-CNN models.

    This module learns optimal weights for combining multiple Fast R-CNN models
    during training. The weights are learned parameters that can be optimized.

    Args:
        models: List of Fast R-CNN models to ensemble
        learn_weights: Whether to learn the ensemble weights
        initial_weights: Initial weights for the models
        temperature: Temperature parameter for softmax normalization of weights

    """

    def __init__(
        self,
        models: list[nn.Module],
        learn_weights: bool = True,
        initial_weights: list[float] | None = None,
        temperature: float = 1.0,
    ):
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.models = nn.ModuleList(models)
        self.temperature = temperature

        # Initialize weights
        if initial_weights is None:
            initial_weights = [1.0] * len(models)

        if learn_weights:
            self.raw_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
        else:
            self.register_buffer("raw_weights", torch.tensor(initial_weights, dtype=torch.float32))

        # Set all models to eval mode
        for model in self.models:
            model.eval()

    @property
    def weights(self) -> torch.Tensor:
        """Get normalized weights using softmax."""
        return F.softmax(self.raw_weights / self.temperature, dim=0)

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Forward pass through the weighted ensemble.

        Args:
            images: List of input images
            targets: Optional list of target dictionaries (for training)

        Returns:
            List of prediction dictionaries, one per image

        """
        # Get predictions from all models
        all_predictions = []
        weights = self.weights

        for model in self.models:
            with torch.no_grad():
                predictions = model(images, targets)
                all_predictions.append(predictions)

        # Combine predictions using learned weights
        num_images = len(all_predictions[0])
        combined_predictions = []

        for img_idx in range(num_images):
            # Collect all predictions for this image
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Weight the scores and combine
            weighted_boxes = []
            weighted_scores = []
            weighted_labels = []

            for pred, weight in zip(img_predictions, weights, strict=False):
                weighted_boxes.append(pred["boxes"])
                weighted_scores.append(pred["scores"] * weight)
                weighted_labels.append(pred["labels"])

            # Combine all weighted predictions
            all_boxes = torch.cat(weighted_boxes, dim=0)
            all_scores = torch.cat(weighted_scores, dim=0)
            all_labels = torch.cat(weighted_labels, dim=0)

            # Sort by score and limit detections
            if len(all_scores) > 0:
                sorted_indices = torch.argsort(all_scores, descending=True)
                max_dets = min(100, len(sorted_indices))  # Limit to top 100 detections
                sorted_indices = sorted_indices[:max_dets]

                combined_predictions.append(
                    {
                        "boxes": all_boxes[sorted_indices],
                        "scores": all_scores[sorted_indices],
                        "labels": all_labels[sorted_indices],
                    }
                )
            else:
                combined_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=images[0].device),
                        "scores": torch.empty((0,), device=images[0].device),
                        "labels": torch.empty((0,), dtype=torch.long, device=images[0].device),
                    }
                )

        return combined_predictions

    def train(self, mode: bool = True):
        """Override train to keep models in eval mode during training."""
        super().train(mode)
        # Keep all models in eval mode
        for model in self.models:
            model.eval()
        return self


class FastRCNNConsensus(nn.Module):
    """
    Consensus-based combination of Fast R-CNN models.

    This module combines predictions by requiring consensus among multiple models.
    Only detections that are predicted by a minimum number of models are kept.

    Args:
        models: List of Fast R-CNN models
        min_consensus: Minimum number of models that must agree on a detection
        iou_threshold: IoU threshold for matching detections across models
        score_aggregation: How to aggregate scores ('mean', 'max', 'min')

    """

    def __init__(
        self,
        models: list[nn.Module],
        min_consensus: int = 2,
        iou_threshold: float = 0.5,
        score_aggregation: str = "mean",
    ):
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.models = nn.ModuleList(models)
        self.min_consensus = min_consensus
        self.iou_threshold = iou_threshold
        self.score_aggregation = score_aggregation

        # Set all models to eval mode
        for model in self.models:
            model.eval()

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]]:
        """
        Forward pass through the consensus ensemble.

        Args:
            images: List of input images
            targets: Optional list of target dictionaries (for training)

        Returns:
            List of prediction dictionaries, one per image

        """
        # Get predictions from all models
        all_predictions = []

        for model in self.models:
            with torch.no_grad():
                predictions = model(images, targets)
                all_predictions.append(predictions)

        # Find consensus predictions
        num_images = len(all_predictions[0])
        consensus_predictions = []

        for img_idx in range(num_images):
            img_predictions = [pred[img_idx] for pred in all_predictions]
            consensus_pred = self._find_consensus(img_predictions)
            consensus_predictions.append(consensus_pred)

        return consensus_predictions

    def _find_consensus(self, predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Find consensus detections among multiple model predictions."""
        if not predictions:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

        # Collect all detections
        all_boxes = []
        all_scores = []
        all_labels = []

        for pred in predictions:
            all_boxes.append(pred["boxes"])
            all_scores.append(pred["scores"])
            all_labels.append(pred["labels"])

        if not all_boxes or len(all_boxes[0]) == 0:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

        # Find matching detections across models
        consensus_boxes = []
        consensus_scores = []
        consensus_labels = []

        # Start with first model's detections
        for _i, (box, score, label) in enumerate(zip(all_boxes[0], all_scores[0], all_labels[0], strict=False)):
            matching_scores = [score]
            matching_count = 1

            # Check other models for matching detections
            for j in range(1, len(predictions)):
                if len(all_boxes[j]) == 0:
                    continue

                # Calculate IoU with all detections from model j
                ious = self._calculate_iou(box.unsqueeze(0), all_boxes[j])

                # Find best matching detection
                best_match_idx = torch.argmax(ious)
                best_iou = ious[best_match_idx]

                # Check if it matches and has same label
                if best_iou > self.iou_threshold and all_labels[j][best_match_idx] == label:
                    matching_scores.append(all_scores[j][best_match_idx])
                    matching_count += 1

            # Keep detection if it has enough consensus
            if matching_count >= self.min_consensus:
                consensus_boxes.append(box)
                consensus_labels.append(label)

                # Aggregate scores
                if self.score_aggregation == "mean":
                    consensus_scores.append(torch.mean(torch.stack(matching_scores)))
                elif self.score_aggregation == "max":
                    consensus_scores.append(torch.max(torch.stack(matching_scores)))
                elif self.score_aggregation == "min":
                    consensus_scores.append(torch.min(torch.stack(matching_scores)))

        if consensus_boxes:
            return {
                "boxes": torch.stack(consensus_boxes),
                "scores": torch.stack(consensus_scores),
                "labels": torch.stack(consensus_labels),
            }
        else:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

    def _calculate_iou(self, boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
        """Calculate IoU between two sets of boxes."""
        # Calculate intersection
        lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # left-top
        rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # right-bottom

        wh = (rb - lt).clamp(min=0)  # width-height
        inter = wh[:, :, 0] * wh[:, :, 1]  # intersection area

        # Calculate union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2 - inter

        # IoU
        iou = inter / union.clamp(min=1e-6)
        return iou.squeeze(0)

    def train(self, mode: bool = True):
        """Override train to keep models in eval mode during training."""
        super().train(mode)
        # Keep all models in eval mode
        for model in self.models:
            model.eval()
        return self
