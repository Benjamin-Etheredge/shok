"""
Differentiable combination module for torchvision detection models.

This module provides utilities for combining outputs from multiple torchvision detection models
in a differentiable manner, allowing for end-to-end training and gradient flow through ensembles.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms


class DifferentiableDetectionEnsemble(nn.Module):
    """
    Differentiable ensemble for torchvision detection models.

    This module combines multiple detection models while maintaining gradient flow,
    enabling end-to-end training of the entire ensemble. It supports various
    combination strategies and can work with any torchvision detection model.

    Args:
        models: List of detection models to ensemble
        combination_method: How to combine predictions ('soft_nms', 'weighted_avg', 'attention')
        learnable_weights: Whether to learn combination weights
        temperature: Temperature for softmax normalization
        nms_threshold: IoU threshold for NMS
        score_threshold: Minimum score threshold
        max_detections: Maximum number of detections per image
        freeze_models: Whether to freeze the constituent models

    """

    def __init__(
        self,
        models: list[nn.Module],
        combination_method: str = "soft_nms",
        learnable_weights: bool = True,
        temperature: float = 1.0,
        nms_threshold: float = 0.5,
        score_threshold: float = 0.05,
        max_detections: int = 100,
        freeze_models: bool = False,
    ):
        super().__init__()

        if not models:
            raise ValueError("At least one model must be provided")

        self.models = nn.ModuleList(models)
        self.combination_method = combination_method
        self.temperature = temperature
        self.nms_threshold = nms_threshold
        self.score_threshold = score_threshold
        self.max_detections = max_detections
        self.num_models = len(models)

        # Initialize combination weights
        if learnable_weights:
            self.model_weights = nn.Parameter(torch.ones(self.num_models))
        else:
            self.register_buffer("model_weights", torch.ones(self.num_models))

        # Attention mechanism for combining features
        if combination_method == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=256,  # Typical feature dimension
                num_heads=8,
                batch_first=True,
            )
            self.feature_projection = nn.Linear(256, 256)

        # Freeze models if requested
        if freeze_models:
            for model in self.models:
                for param in model.parameters():
                    param.requires_grad = False

    @property
    def normalized_weights(self) -> torch.Tensor:
        """Get softmax normalized model weights."""
        return F.softmax(self.model_weights / self.temperature, dim=0)

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        """
        Forward pass through the differentiable ensemble.

        Args:
            images: List of input images
            targets: Optional list of target dictionaries (for training)

        Returns:
            Combined predictions or losses (depending on training mode)

        """
        if self.training and targets is not None:
            return self._forward_train(images, targets)
        else:
            return self._forward_eval(images)

    def _forward_train(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        """Training forward pass with loss combination."""
        all_losses = []
        weights = self.normalized_weights

        # Get losses from all models
        for i, model in enumerate(self.models):
            model_losses = model(images, targets)

            # Weight the losses
            weighted_losses = {}
            for loss_name, loss_value in model_losses.items():
                weighted_losses[loss_name] = loss_value * weights[i]

            all_losses.append(weighted_losses)

        # Combine losses
        combined_losses = {}
        for loss_name in all_losses[0].keys():
            combined_losses[loss_name] = sum(losses[loss_name] for losses in all_losses)

        return combined_losses

    def _forward_eval(self, images: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Evaluation forward pass with prediction combination."""
        # Get predictions from all models
        all_predictions = []

        for model in self.models:
            with torch.no_grad():
                predictions = model(images)
                all_predictions.append(predictions)

        # Combine predictions based on method
        if self.combination_method == "soft_nms":
            return self._combine_soft_nms(all_predictions)
        elif self.combination_method == "weighted_avg":
            return self._combine_weighted_average(all_predictions)
        elif self.combination_method == "attention":
            return self._combine_attention(all_predictions)
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")

    def _combine_soft_nms(self, all_predictions: list[list[dict[str, torch.Tensor]]]) -> list[dict[str, torch.Tensor]]:
        """Combine predictions using soft NMS approach."""
        num_images = len(all_predictions[0])
        combined_predictions = []
        weights = self.normalized_weights

        # Get device from first prediction
        device = (
            all_predictions[0][0]["boxes"].device if all_predictions[0][0]["boxes"].numel() > 0 else torch.device("cpu")
        )

        for img_idx in range(num_images):
            # Collect all predictions for this image
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Combine boxes, scores, and labels with weights
            all_boxes = []
            all_scores = []
            all_labels = []

            for pred, weight in zip(img_predictions, weights, strict=False):
                all_boxes.append(pred["boxes"])
                # Apply soft weighting to scores
                weighted_scores = pred["scores"] * weight
                all_scores.append(weighted_scores)
                all_labels.append(pred["labels"])

            if all_boxes and len(all_boxes[0]) > 0:
                # Concatenate all predictions
                combined_boxes = torch.cat(all_boxes, dim=0)
                combined_scores = torch.cat(all_scores, dim=0)
                combined_labels = torch.cat(all_labels, dim=0)

                # Apply soft NMS
                keep_indices = self._soft_nms(combined_boxes, combined_scores, combined_labels)

                combined_predictions.append(
                    {
                        "boxes": combined_boxes[keep_indices],
                        "scores": combined_scores[keep_indices],
                        "labels": combined_labels[keep_indices],
                    }
                )
            else:
                combined_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )

        return combined_predictions

    def _combine_weighted_average(
        self, all_predictions: list[list[dict[str, torch.Tensor]]]
    ) -> list[dict[str, torch.Tensor]]:
        """Combine predictions using weighted averaging."""
        num_images = len(all_predictions[0])
        combined_predictions = []
        weights = self.normalized_weights

        # Get device from first prediction
        device = (
            all_predictions[0][0]["boxes"].device if all_predictions[0][0]["boxes"].numel() > 0 else torch.device("cpu")
        )

        for img_idx in range(num_images):
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Weighted combination of predictions
            if img_predictions and len(img_predictions[0]["boxes"]) > 0:
                # Use clustering approach for box averaging
                combined_pred = self._cluster_and_average_boxes(img_predictions, weights)
                combined_predictions.append(combined_pred)
            else:
                combined_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )

        return combined_predictions

    def _combine_attention(self, all_predictions: list[list[dict[str, torch.Tensor]]]) -> list[dict[str, torch.Tensor]]:
        """Combine predictions using attention mechanism."""
        # For attention-based combination, we need to extract features
        # This is a simplified version - in practice, you'd want to use
        # intermediate features from the backbone
        num_images = len(all_predictions[0])
        combined_predictions = []

        # Get device from first prediction
        device = (
            all_predictions[0][0]["boxes"].device if all_predictions[0][0]["boxes"].numel() > 0 else torch.device("cpu")
        )

        for img_idx in range(num_images):
            img_predictions = [pred[img_idx] for pred in all_predictions]

            if img_predictions and len(img_predictions[0]["boxes"]) > 0:
                # Create feature representations for each prediction
                features = []
                for pred in img_predictions:
                    # Simple feature representation based on box coordinates and scores
                    box_features = pred["boxes"]  # [N, 4]
                    score_features = pred["scores"].unsqueeze(1)  # [N, 1]

                    # Pad to 256 dimensions
                    pred_features = torch.cat([box_features, score_features], dim=1)
                    if pred_features.shape[1] < 256:
                        pad_size = 256 - pred_features.shape[1]
                        pred_features = F.pad(pred_features, (0, pad_size))

                    features.append(pred_features)

                # Stack features and apply attention
                stacked_features = torch.stack(features, dim=1)  # [N, num_models, 256]
                attended_features, _ = self.attention(stacked_features, stacked_features, stacked_features)

                # Combine using attended features
                combined_pred = self._combine_attended_features(img_predictions, attended_features)
                combined_predictions.append(combined_pred)
            else:
                combined_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=device),
                        "scores": torch.empty((0,), device=device),
                        "labels": torch.empty((0,), dtype=torch.long, device=device),
                    }
                )

        return combined_predictions

    def _soft_nms(
        self, boxes: torch.Tensor, scores: torch.Tensor, labels: torch.Tensor, sigma: float = 0.5
    ) -> torch.Tensor:
        """Apply soft NMS to reduce overlapping detections."""
        if len(boxes) == 0:
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        # Filter by score threshold
        score_mask = scores > self.score_threshold
        if not score_mask.any():
            return torch.tensor([], dtype=torch.long, device=boxes.device)

        # Apply batched NMS per class
        keep_indices = batched_nms(boxes[score_mask], scores[score_mask], labels[score_mask], self.nms_threshold)

        # Map back to original indices
        original_indices = torch.nonzero(score_mask, as_tuple=False).squeeze(1)
        keep_indices = original_indices[keep_indices]

        # Limit number of detections
        if len(keep_indices) > self.max_detections:
            keep_indices = keep_indices[: self.max_detections]

        return keep_indices

    def _cluster_and_average_boxes(
        self, predictions: list[dict[str, torch.Tensor]], weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Cluster similar boxes and average them."""
        # Simple implementation - in practice, you'd use more sophisticated clustering
        all_boxes = torch.cat([pred["boxes"] for pred in predictions], dim=0)
        all_scores = torch.cat(
            [pred["scores"] * weight for pred, weight in zip(predictions, weights, strict=False)], dim=0
        )
        all_labels = torch.cat([pred["labels"] for pred in predictions], dim=0)

        # Apply NMS to get final detections
        keep_indices = self._soft_nms(all_boxes, all_scores, all_labels)

        return {
            "boxes": all_boxes[keep_indices],
            "scores": all_scores[keep_indices],
            "labels": all_labels[keep_indices],
        }

    def _combine_attended_features(
        self, predictions: list[dict[str, torch.Tensor]], attended_features: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Combine predictions using attended features."""
        # This is a simplified implementation
        # In practice, you'd use the attended features to weight the predictions

        # For now, just use the first model's predictions with attention weights
        base_pred = predictions[0]

        # Apply attention weights (simplified)
        attention_weights = attended_features.mean(dim=1)[:, 0]  # [N]

        return {
            "boxes": base_pred["boxes"],
            "scores": base_pred["scores"] * attention_weights,
            "labels": base_pred["labels"],
        }

    def set_combination_method(self, method: str):
        """Change the combination method."""
        valid_methods = ["soft_nms", "weighted_avg", "attention"]
        if method not in valid_methods:
            raise ValueError(f"Method must be one of {valid_methods}")
        self.combination_method = method

    def get_model_weights(self) -> torch.Tensor:
        """Get current model weights."""
        return self.normalized_weights.detach()

    def train(self, mode: bool = True):
        """Set training mode."""
        super().train(mode)
        # Keep constituent models in appropriate mode
        for model in self.models:
            model.train(mode)
        return self


class AdaptiveDetectionEnsemble(nn.Module):
    """
    Adaptive ensemble that learns to route inputs to different models.

    This module learns which model is best suited for each input, providing
    dynamic model selection based on image characteristics.
    """

    def __init__(
        self,
        models: list[nn.Module],
        feature_dim: int = 512,
        num_classes: int = 91,  # COCO classes
        routing_temperature: float = 1.0,
    ):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.num_models = len(models)

        # Router network
        self.router = nn.Sequential(
            nn.AdaptiveAvgPool2d((7, 7)),
            nn.Flatten(),
            nn.Linear(7 * 7 * 3, feature_dim),  # Assuming RGB input
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Linear(feature_dim // 2, self.num_models),
        )

        self.routing_temperature = routing_temperature

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        """Forward pass with adaptive routing."""
        # Stack images for routing
        stacked_images = torch.stack(images, dim=0)

        # Get routing weights
        routing_logits = self.router(stacked_images)
        routing_weights = F.softmax(routing_logits / self.routing_temperature, dim=1)

        if self.training and targets is not None:
            return self._forward_train_adaptive(images, targets, routing_weights)
        else:
            return self._forward_eval_adaptive(images, routing_weights)

    def _forward_train_adaptive(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]], routing_weights: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Training forward with adaptive routing."""
        all_losses = []

        for i, model in enumerate(self.models):
            model_losses = model(images, targets)

            # Weight losses by routing weights
            weighted_losses = {}
            for loss_name, loss_value in model_losses.items():
                # Average routing weights across batch
                avg_weight = routing_weights[:, i].mean()
                weighted_losses[loss_name] = loss_value * avg_weight

            all_losses.append(weighted_losses)

        # Combine losses
        combined_losses = {}
        for loss_name in all_losses[0].keys():
            combined_losses[loss_name] = sum(losses[loss_name] for losses in all_losses)

        return combined_losses

    def _forward_eval_adaptive(
        self, images: list[torch.Tensor], routing_weights: torch.Tensor
    ) -> list[dict[str, torch.Tensor]]:
        """Evaluation forward with adaptive routing."""
        batch_size = len(images)
        all_predictions = []

        # Get predictions from all models
        for model in self.models:
            predictions = model(images)
            all_predictions.append(predictions)

        # Combine predictions using routing weights
        combined_predictions = []

        for img_idx in range(batch_size):
            img_weights = routing_weights[img_idx]
            img_predictions = [pred[img_idx] for pred in all_predictions]

            # Weighted combination
            if img_predictions and len(img_predictions[0]["boxes"]) > 0:
                combined_boxes = []
                combined_scores = []
                combined_labels = []

                for pred, weight in zip(img_predictions, img_weights, strict=False):
                    combined_boxes.append(pred["boxes"])
                    combined_scores.append(pred["scores"] * weight)
                    combined_labels.append(pred["labels"])

                if combined_boxes:
                    all_boxes = torch.cat(combined_boxes, dim=0)
                    all_scores = torch.cat(combined_scores, dim=0)
                    all_labels = torch.cat(combined_labels, dim=0)

                    # Apply NMS
                    keep_indices = batched_nms(all_boxes, all_scores, all_labels, 0.5)

                    combined_predictions.append(
                        {
                            "boxes": all_boxes[keep_indices],
                            "scores": all_scores[keep_indices],
                            "labels": all_labels[keep_indices],
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
            else:
                combined_predictions.append(
                    {
                        "boxes": torch.empty((0, 4), device=images[0].device),
                        "scores": torch.empty((0,), device=images[0].device),
                        "labels": torch.empty((0,), dtype=torch.long, device=images[0].device),
                    }
                )

        return combined_predictions

    def get_routing_weights(self, images: list[torch.Tensor]) -> torch.Tensor:
        """Get routing weights for given images."""
        stacked_images = torch.stack(images, dim=0)
        routing_logits = self.router(stacked_images)
        return F.softmax(routing_logits / self.routing_temperature, dim=1)


class HierarchicalDetectionEnsemble(nn.Module):
    """
    Hierarchical ensemble that combines models at different scales/resolutions.

    This module is designed to combine models that specialize in different
    object scales, providing better coverage across all object sizes.
    """

    def __init__(
        self, models: list[nn.Module], scale_factors: list[float] | None = None, fusion_method: str = "scale_aware"
    ):
        super().__init__()

        self.models = nn.ModuleList(models)
        self.scale_factors = scale_factors or [1.0] * len(models)
        self.fusion_method = fusion_method

        # Scale-aware fusion network
        if fusion_method == "scale_aware":
            self.scale_fusion = nn.Sequential(
                nn.Linear(len(models) + 1, 64),  # +1 for scale information
                nn.ReLU(),
                nn.Linear(64, len(models)),
                nn.Softmax(dim=-1),
            )

    def forward(
        self, images: list[torch.Tensor], targets: list[dict[str, torch.Tensor]] | None = None
    ) -> list[dict[str, torch.Tensor]] | dict[str, torch.Tensor]:
        """Forward pass with hierarchical combination."""
        # Process images at different scales
        all_predictions = []

        for _i, (model, scale_factor) in enumerate(zip(self.models, self.scale_factors, strict=False)):
            if scale_factor != 1.0:
                # Resize images
                scaled_images = []
                for img in images:
                    h, w = img.shape[-2:]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    scaled_img = F.interpolate(
                        img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False
                    ).squeeze(0)
                    scaled_images.append(scaled_img)

                predictions = model(scaled_images, targets)

                # Scale boxes back to original resolution
                if not self.training:
                    for pred in predictions:
                        if len(pred["boxes"]) > 0:
                            pred["boxes"] = pred["boxes"] / scale_factor

                all_predictions.append(predictions)
            else:
                predictions = model(images, targets)
                all_predictions.append(predictions)

        if self.training and targets is not None:
            return self._combine_losses(all_predictions)
        else:
            return self._combine_hierarchical_predictions(all_predictions)

    def _combine_losses(self, all_predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Combine losses from different scales."""
        combined_losses = {}

        for loss_name in all_predictions[0].keys():
            combined_losses[loss_name] = sum(pred[loss_name] for pred in all_predictions) / len(all_predictions)

        return combined_losses

    def _combine_hierarchical_predictions(
        self, all_predictions: list[list[dict[str, torch.Tensor]]]
    ) -> list[dict[str, torch.Tensor]]:
        """Combine predictions from different scales."""
        num_images = len(all_predictions[0])
        combined_predictions = []

        for img_idx in range(num_images):
            img_predictions = [pred[img_idx] for pred in all_predictions]

            if self.fusion_method == "scale_aware":
                combined_pred = self._scale_aware_fusion(img_predictions)
            else:
                combined_pred = self._simple_fusion(img_predictions)

            combined_predictions.append(combined_pred)

        return combined_predictions

    def _scale_aware_fusion(self, predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Combine predictions using scale-aware fusion."""
        # Extract box sizes to determine scale weights
        all_boxes = []
        all_scores = []
        all_labels = []
        box_scales = []

        for pred in predictions:
            if len(pred["boxes"]) > 0:
                boxes = pred["boxes"]
                scores = pred["scores"]
                labels = pred["labels"]

                # Calculate box scales (area)
                box_areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
                scales = torch.sqrt(box_areas).unsqueeze(1)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)
                box_scales.append(scales)

        if not all_boxes:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

        # Concatenate all predictions
        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_scores = torch.cat(all_scores, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)
        torch.cat(box_scales, dim=0)

        # Apply NMS
        keep_indices = batched_nms(combined_boxes, combined_scores, combined_labels, 0.5)

        return {
            "boxes": combined_boxes[keep_indices],
            "scores": combined_scores[keep_indices],
            "labels": combined_labels[keep_indices],
        }

    def _simple_fusion(self, predictions: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Simple concatenation and NMS fusion."""
        all_boxes = []
        all_scores = []
        all_labels = []

        for pred in predictions:
            if len(pred["boxes"]) > 0:
                all_boxes.append(pred["boxes"])
                all_scores.append(pred["scores"])
                all_labels.append(pred["labels"])

        if not all_boxes:
            return {
                "boxes": torch.empty((0, 4)),
                "scores": torch.empty((0,)),
                "labels": torch.empty((0,), dtype=torch.long),
            }

        combined_boxes = torch.cat(all_boxes, dim=0)
        combined_scores = torch.cat(all_scores, dim=0)
        combined_labels = torch.cat(all_labels, dim=0)

        # Apply NMS
        keep_indices = batched_nms(combined_boxes, combined_scores, combined_labels, 0.5)

        return {
            "boxes": combined_boxes[keep_indices],
            "scores": combined_scores[keep_indices],
            "labels": combined_labels[keep_indices],
        }
