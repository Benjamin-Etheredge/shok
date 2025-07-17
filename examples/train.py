from jsonargparse import lazy_instance
from lightning.pytorch.cli import LightningCLI
from torchvision.models.detection import (
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from shok.data.datasets.coco import CocoDataModule
from shok.patch.module import ObjectDetectionPatch
from shok.utils.callbacks import LogPerformanceCallback, WandbObjectDetectionCallback

default_model = {
    "class_path": "torchvision.models.detection.fasterrcnn_resnet50_fpn_v2",
    "init_args": {
        "weights": FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
    },
}


class CustomLightningCLI(LightningCLI):
    """
    Custom CLI class for PyTorch Lightning that allows setting default arguments for the model.

    Methods
    -------
    add_arguments_to_parser(parser):
        Adds custom default arguments to the parser, specifically setting the default model configuration.

    """

    def add_arguments_to_parser(self, parser):
        """
        Adds default arguments to the given argument parser.

        This method sets the default value for the "model.model" argument in the parser
        to the value specified by `default_model`.

        Args:
            parser: An argument parser object to which the default arguments will be added.

        """
        parser.set_defaults({"model.model": default_model})


def cli_main():
    """
    Main entry point for training an object detection patch model using a custom Lightning CLI.

    This function configures and launches the training process with the following features:
    - Uses `CustomLightningCLI` to handle configuration and training.
    - Sets a fixed random seed for reproducibility.
    - Specifies model and data module classes (`ObjectDetectionPatch`, `CocoDataModule`).
    - Configures trainer defaults, including:
        - Maximum epochs, accelerator, device count.
        - Custom callbacks for logging and performance monitoring.
        - Wandb logger integration for experiment tracking.
        - Batch normalization and progress bar settings.
        - Logging and validation frequency.
        - Limits on train and validation batches for faster experimentation.
    - Allows for further customization via TODOs (e.g., automatic batch size finding, gradient accumulation).
    - Optionally disables saving config and automatic training loop execution.

    Note:
        - Callback instantiation uses `lazy_instance` for compatibility.
        - Some advanced features are commented out for future implementation.

    """
    # TODO add automatic batch_size finding
    CustomLightningCLI(
        ObjectDetectionPatch,
        CocoDataModule,
        seed_everything_default=42,
        # TODO switch to accumulating gradients from batches with scheduler to up it slowly
        # TODO use subclass_mode_model
        trainer_defaults={
            "max_epochs": 1000000,
            "accelerator": "auto",
            "devices": 1,
            "callbacks": [
                # NOTE: For callbacks giving class_path and init_args will not work because it isn't jsonargparse instantiating them. This is kind of unfortunate, but it is like that currently. For more consistency you could use lazy_instance as explained in the link I sent before. This would work for all trainer defaults, i.e.
                lazy_instance(WandbObjectDetectionCallback, train_log_frequency=4, val_log_frequency=16),
                lazy_instance(LogPerformanceCallback),
                # lazy_instance(
                #     "lightning.pytorch.callbacks.ModelCheckpoint",
                #     monitor="val/loss",
                #     save_top_k=1,
                #     mode="min",
                #     )
            ],
            "logger": {
                "class_path": "lightning.pytorch.loggers.WandbLogger",
                "init_args": {
                    "project": "adversarial-patch",
                },
            },
            "sync_batchnorm": False,  # disable since batchnorm is locked
            "enable_progress_bar": True,
            "log_every_n_steps": 1,
            "check_val_every_n_epoch": 16,
            "limit_train_batches": 0.1,
            "limit_val_batches": 0.1,
            # "accumulate_grad_batches": 99999999999999999,
            # "profiler": "pytorch",  # Enable profiler for debugging
        },
        save_config_callback=None,
        # run=False,  # Don't run the training loop automatically
    )


if __name__ == "__main__":
    cli_main()
