"""
Training script for detection ensemble models with ObjectDetectionPatch using PyTorch Lightning.

This script demonstrates how to train differentiable detection ensembles
in combination with adversarial patch training using the ObjectDetectionPatch module.
"""

import torch
import torchvision.models.detection as detection
from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
    RetinaNet_ResNet50_FPN_Weights,
)

from shok.data.datasets.coco import CocoDataModule
from shok.patch.module import ObjectDetectionPatch
from shok.utils.callbacks.wandb import LogPerformanceCallback, WandbObjectDetectionCallback
from shok.utils.models.detection_combo import (
    AdaptiveDetectionEnsemble,
    DifferentiableDetectionEnsemble,
    HierarchicalDetectionEnsemble,
)

# Set global torch settings
torch.autograd.set_detect_anomaly(True)
torch.set_float32_matmul_precision("medium")


def create_ensemble_model(ensemble_type="differentiable", combination_method="soft_nms", num_models=2):
    """
    Create different types of detection ensemble models.

    Args:
        ensemble_type: Type of ensemble ("differentiable", "adaptive", "hierarchical")
        combination_method: Combination method for differentiable ensemble
        num_models: Number of models to use in the ensemble

    Returns:
        Ensemble model ready for training

    """
    # Create diverse set of detection models
    base_models = [
        detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
        detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT),
    ]

    # Select subset of models
    models = base_models[:num_models]

    if ensemble_type == "differentiable":
        ensemble = DifferentiableDetectionEnsemble(
            models=models,
            combination_method=combination_method,
            learnable_weights=True,
            temperature=2.0,
            nms_threshold=0.5,
            score_threshold=0.05,
            max_detections=100,
            freeze_models=True,  # Only train ensemble weights for faster training
        )
    elif ensemble_type == "adaptive":
        ensemble = AdaptiveDetectionEnsemble(models=models, feature_dim=256, routing_temperature=1.0)
        # Freeze base models for faster training
        for model in ensemble.models:
            for param in model.parameters():
                param.requires_grad = False
    elif ensemble_type == "hierarchical":
        # Use different scales for different models
        scale_factors = [0.8, 1.0, 1.2][:num_models]
        ensemble = HierarchicalDetectionEnsemble(
            models=models, scale_factors=scale_factors, fusion_method="scale_aware"
        )
        # Freeze base models for faster training
        for model in ensemble.models:
            for param in model.parameters():
                param.requires_grad = False
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return ensemble


def lazy_create_ensemble_model(ensemble_type="differentiable", combination_method="soft_nms", num_models=2):
    """
    Create different types of detection ensemble models.

    Args:
        ensemble_type: Type of ensemble ("differentiable", "adaptive", "hierarchical")
        combination_method: Combination method for differentiable ensemble
        num_models: Number of models to use in the ensemble

    Returns:
        Ensemble model ready for training

    """
    # Create diverse set of detection models
    base_models = [
        detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
        detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT),
    ]

    # Select subset of models
    models = base_models[:num_models]

    if ensemble_type == "differentiable":
        ensemble_cls = DifferentiableDetectionEnsemble
        ensemble_kwargs = {
            "models": models,
            "combination_method": combination_method,
            "learnable_weights": True,
            "temperature": 2.0,
            "nms_threshold": 0.5,
            "score_threshold": 0.05,
            "max_detections": 100,
            "freeze_models": True,  # Only train ensemble weights for faster training
        }
    elif ensemble_type == "adaptive":
        ensemble_cls = AdaptiveDetectionEnsemble
        ensemble_kwargs = {"models": models, "feature_dim": 256, "routing_temperature": 1.0}
        # Freeze base models for faster training
        # for model in ensemble.models:
        #     for param in model.parameters():
        #         param.requires_grad = False
    elif ensemble_type == "hierarchical":
        # Use different scales for different models
        scale_factors = [0.8, 1.0, 1.2][:num_models]
        ensemble_cls = HierarchicalDetectionEnsemble
        ensemble_kwargs = {"models": models, "scale_factors": scale_factors, "fusion_method": "scale_aware"}
        # Freeze base models for faster training
        # for model in ensemble.models:
        #     for param in model.parameters():
        #         param.requires_grad = False
    else:
        raise ValueError(f"Unknown ensemble type: {ensemble_type}")

    return {"class_path": ensemble_cls.__module__ + "." + ensemble_cls.__name__, "init_args": ensemble_kwargs}
    return lazy_instance(ensemble_cls, **ensemble_kwargs)


# Default ensemble configurations
default_differentiable_ensemble = {
    "class_path": "shok.utils.models.detection_combo.DifferentiableDetectionEnsemble",
    "init_args": {
        "models": [
            detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
            detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        ],
        "combination_method": "soft_nms",
        "learnable_weights": True,
        "temperature": 2.0,
        "freeze_models": True,
    },
}

default_adaptive_ensemble = {
    "class_path": "shok.utils.models.detection_combo.AdaptiveDetectionEnsemble",
    "init_args": {
        "models": [
            detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
            detection.retinanet_resnet50_fpn(weights=RetinaNet_ResNet50_FPN_Weights.DEFAULT),
        ],
        "feature_dim": 256,
        "routing_temperature": 1.0,
    },
}

default_hierarchical_ensemble = {
    "class_path": "shok.utils.models.detection_combo.HierarchicalDetectionEnsemble",
    "init_args": {
        "models": [
            detection.fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT),
            detection.fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT),
        ],
        "scale_factors": [0.8, 1.2],
        "fusion_method": "scale_aware",
    },
}


class EnsembleTrainingCLI(LightningCLI):
    """
    Custom CLI class for training detection ensemble models with ObjectDetectionPatch.

    This CLI provides default configurations for different ensemble types and
    integrates with the ObjectDetectionPatch module for adversarial training.
    """

    def add_arguments_to_parser(self, parser):
        """Add ensemble-specific arguments to the parser."""
        parser.add_argument(
            "--ensemble-type",
            type=str,
            default="differentiable",
            choices=["differentiable", "adaptive", "hierarchical"],
            help="Type of ensemble to use",
        )
        parser.add_argument(
            "--combination-method",
            type=str,
            default="soft_nms",
            choices=["soft_nms", "weighted_avg", "attention"],
            help="Combination method for differentiable ensemble",
        )
        parser.add_argument("--num-models", type=int, default=2, help="Number of models in the ensemble")
        parser.add_argument(
            "--freeze-base-models",
            action="store_true",
            help="Freeze base model parameters (train only ensemble weights)",
        )

        # Set default model based on ensemble type
        # if hasattr(self, 'config') and self.config.get('ensemble_type') == 'adaptive':
        #     parser.set_defaults({"model.model": default_adaptive_ensemble})
        # elif hasattr(self, 'config') and self.config.get('ensemble_type') == 'hierarchical':
        #     parser.set_defaults({"model.model": default_hierarchical_ensemble})
        # else:
        #     parser.set_defaults({"model.model": default_differentiable_ensemble})

    def before_instantiate_classes(self):
        """Create ensemble model based on CLI arguments."""
        # Get ensemble configuration from args
        ensemble_type = getattr(self.config, "ensemble_type", "differentiable")
        combination_method = getattr(self.config, "combination_method", "soft_nms")
        num_models = getattr(self.config, "num_models", 2)

        # Create ensemble model
        ensemble_model = lazy_create_ensemble_model(
            ensemble_type=ensemble_type, combination_method=combination_method, num_models=num_models
        )
        # Update model configuration
        # Update lightning config with ensemble model
        self.config[self.config.subcommand]["model"]["model"] = ensemble_model
        # ic(self.config)
        # ic(list(self.config.keys()))
        # ic(self.config.config)
        # ic(self.config)
        # ic(type(self.config))
        # ic(self.config.as_dict().keys())
        # ic(self.config.subcommand)
        # ic(self.config.config)
        # ic(self.config.fit)
        # self.model.init_args['model'] = ensemble_model
        # self.config['model']['init_args']['model'] = ensemble_model
        # self.config_init['model']['init_args']['model'] = ensemble_model
        self.config[self.config.subcommand]["model"]["model"] = ensemble_model
        # self.config[self.config.subcommand]['model']['init_args']['model'] = ensemble_model
        # self.config[self.config.subcommand]['model']['model'] = lazy_instance(
        #     create_ensemble_model,
        #     ensemble_type=ensemble_type,
        #     combination_method=combination_method,
        #     num_models=num_models
        # )


def cli_main():
    """
    Main entry point for training detection ensemble models with ObjectDetectionPatch.

    This function configures and launches the training process with:
    - Support for multiple ensemble types (differentiable, adaptive, hierarchical)
    - Integration with ObjectDetectionPatch for adversarial training
    - Customizable ensemble configurations
    - Advanced logging and monitoring
    - Efficient training with frozen base models

    Usage Examples:
    ---------------
    # Train differentiable ensemble with soft NMS
    python train_detection_ensemble.py --ensemble-type differentiable --combination-method soft_nms

    # Train adaptive ensemble with 3 models
    python train_detection_ensemble.py --ensemble-type adaptive --num-models 3

    # Train hierarchical ensemble with frozen base models
    python train_detection_ensemble.py --ensemble-type hierarchical --freeze-base-models
    """
    EnsembleTrainingCLI(
        ObjectDetectionPatch,
        CocoDataModule,
        seed_everything_default=42,
        trainer_defaults={
            "max_epochs": 1000000,
            "accelerator": "auto",
            "devices": 1,
            "callbacks": [
                lazy_instance(WandbObjectDetectionCallback, train_log_frequency=4, val_log_frequency=16),
                lazy_instance(LogPerformanceCallback),
                # Note: Using class_path/init_args format for callbacks that need string references
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                    "project": "adversarial-patch-ensemble",
                    "tags": ["ensemble", "adversarial", "detection"],
                },
            },
            "sync_batchnorm": False,
            "enable_progress_bar": True,
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 8,
            "limit_train_batches": 0.1,  # For faster experimentation
            "limit_val_batches": 0.1,
            # "gradient_clip_val": 1.0,  # Prevent gradient explosion
            # "precision": "16-mixed",  # Mixed precision for faster training
            # "accumulate_grad_batches": 4,  # Accumulate gradients for larger effective batch size
        },
        save_config_callback=None,
    )


def train_differentiable_ensemble():
    """Train a differentiable ensemble with specific configuration."""
    import sys

    # Set CLI arguments programmatically
    sys.argv = [
        "train_detection_ensemble.py",
        "--ensemble-type",
        "differentiable",
        "--combination-method",
        "soft_nms",
        "--num-models",
        "2",
        "--freeze-base-models",
        "--model.learning_rate",
        "0.001",
        "--model.patch_shape",
        "[3, 1024, 1024]",
        "--model.eot_samples",
        "2",
        "--trainer.max_epochs",
        "50",
        "--trainer.limit_train_batches",
        "0.05",
        "--trainer.limit_val_batches",
        "0.05",
    ]

    cli_main()


def train_adaptive_ensemble():
    """Train an adaptive ensemble with specific configuration."""
    import sys

    # Set CLI arguments programmatically
    sys.argv = [
        "train_detection_ensemble.py",
        "--ensemble-type",
        "adaptive",
        "--num-models",
        "3",
        "--model.learning_rate",
        "0.0005",
        "--model.patch_shape",
        "[3, 1024, 1024]",
        "--model.eot_samples",
        "3",
        "--trainer.max_epochs",
        "100",
        "--trainer.limit_train_batches",
        "0.1",
        "--trainer.limit_val_batches",
        "0.1",
    ]

    cli_main()


def train_hierarchical_ensemble():
    """Train a hierarchical ensemble with specific configuration."""
    import sys

    # Set CLI arguments programmatically
    sys.argv = [
        "train_detection_ensemble.py",
        "--ensemble-type",
        "hierarchical",
        "--num-models",
        "3",
        "--model.learning_rate",
        "0.001",
        "--model.patch_shape",
        "[3, 1024, 1024]",
        "--model.eot_samples",
        "2",
        "--trainer.max_epochs",
        "75",
        "--trainer.limit_train_batches",
        "0.1",
        "--trainer.limit_val_batches",
        "0.1",
    ]

    cli_main()


if __name__ == "__main__":
    cli_main()
