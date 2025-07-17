import time

import lightning as L
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.utilities import rank_zero_only

import wandb


def translate_pred_to_wandb_boxes(preds, patch_image):
    """
    Converts prediction results into a format compatible with Weights & Biases (wandb) for bounding box visualization.

    Args:
        preds (dict): A dictionary containing prediction results with keys "boxes", "scores", and "labels".
            - "boxes" (list or tensor): Bounding box coordinates.
            - "scores" (list or tensor, optional): Confidence scores for each box. If missing, defaults to 100.0.
            - "labels" (list or tensor): Class labels for each box.
        patch_image: The image associated with the predictions, used for box translation.

    Returns:
        dict: A dictionary with a "predictions" key containing "box_data", which is a list of bounding boxes
        formatted for wandb visualization.

    """
    box_data = []
    if "scores" not in preds:
        # could be ground truth
        preds["scores"] = [torch.tensor(100.0)] * len(preds["boxes"])
    for box, score, label in zip(preds["boxes"], preds["scores"], preds["labels"]):
        box_data.append(translate_pytorch_boxes_to_wandb(box, score, label, patch_image))
    return {
        "predictions": {
            "box_data": box_data,
            # "class_labels": mapping
        }
    }


def translate_pytorch_boxes_to_wandb(boxes, score, label, patch_image):
    """
    Converts PyTorch bounding box coordinates to the format expected by Weights & Biases (wandb).

    Args:
        boxes (torch.Tensor): A tensor containing bounding box coordinates in the format [minX, minY, maxX, maxY].
        score (torch.Tensor or None): A tensor containing the confidence score for the bounding box, or None.
        label (torch.Tensor): A tensor containing the class label for the bounding box.
        patch_image (torch.Tensor): The image tensor corresponding to the bounding box, used to normalize coordinates.

    Returns:
        dict: A dictionary containing the normalized bounding box position, class ID, and score, formatted for wandb.

    """
    return {
        "position": {
            "minX": boxes[0].item() / patch_image.shape[2],
            "minY": boxes[1].item() / patch_image.shape[1],
            "maxX": boxes[2].item() / patch_image.shape[2],
            "maxY": boxes[3].item() / patch_image.shape[1],
        },
        "class_id": int(label.item()),
        # "box_caption": f"{mapping.get(int(label.item()), str(label.item()))}: {score.item():.2f}",
        "score": float(score.item()) if score is not None else None,
    }


class WandbObjectDetectionCallback(Callback):
    """
    WandbObjectDetectionCallback.

    A PyTorch Lightning callback for logging object detection images and bounding boxes to Weights & Biases (WandB) during training and validation.

    This callback supports logging images with bounding boxes at configurable frequencies for both training and validation phases. It automatically
    extracts class information from the provided datamodule and formats it for WandB visualization.

        patch_log_frequency (int): Frequency (in epochs) to log patch images at the end of training epochs.
        train_log_frequency (int): Frequency (in epochs) to log training images and bounding boxes.
        val_log_frequency (int): Frequency (in epochs) to log validation images and bounding boxes.

    Raises:
        ValueError: If WandbLogger is not set up or if the datamodule is not configured.

    Methods:
        setup(trainer, pl_module, stage=None): Initializes WandB experiment and class mapping from the datamodule.
        on_train_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0): Logs training images and bounding boxes at the specified frequency.
        on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0): Logs validation images and bounding boxes at the
        specified frequency.
        on_train_epoch_end(trainer, pl_module): Logs the patch image at the end of the training epoch if available.

    Note:
        - The callback expects the datamodule to have an `idx_to_class` attribute for mapping class indices to names.
        - Images and bounding boxes are logged using WandB's image and box utilities for visualization.

    """

    # TODO add support for logging patch images at the end of training epochs
    def __init__(
        self,
        patch_log_frequency: int = 1,
        train_log_frequency: int = 1,
        val_log_frequency: int = 1,
    ):
        """
        Constructor for WandbObjectDetectionCallback.

        Initializes the callback for logging images with bounding boxes to WandB during training and validation.

            patch_log_frequency (int, optional): Frequency of logging patch images (in epochs). Defaults to 1.
            train_log_frequency (int, optional): Frequency of logging during training (in epochs). Defaults to 1.
            val_log_frequency (int, optional): Frequency of logging during validation (in epochs). Defaults to 1.
        """
        super().__init__()
        self._wandb = None
        self.patch_log_frequency = patch_log_frequency
        self.train_log_frequency = train_log_frequency
        self.val_log_frequency = val_log_frequency

    def setup(self, trainer, pl_module, stage=None):
        """
        Sets up the callback by initializing the Wandb logger and extracting class information from the datamodule.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module (model) being trained.
            stage (optional): The current stage of training (e.g., 'fit', 'test').

        Raises:
            ValueError: If the Wandb logger is not set up or if the datamodule is not configured.

        Side Effects:
            - Initializes `self._wandb` with the Wandb experiment object.
            - Sets `self.idx_to_class` from the datamodule if available.
            - Creates `self.wandb_classes` as a `wandb.Classes` object if class mapping is provided.

        """
        if hasattr(pl_module, "logger") and isinstance(pl_module.logger, WandbLogger):
            self._wandb = pl_module.logger.experiment
        else:
            raise ValueError("Wandb logger is not set up. Please use WandbLogger.")

        datamodule = trainer.datamodule
        if datamodule is None:
            raise ValueError("Datamodule is not set up. Please ensure you have a datamodule configured.")

        self.idx_to_class = getattr(datamodule, "idx_to_class", None)
        if hasattr(datamodule, "idx_to_class"):
            self.wandb_classes = wandb.Classes(
                [
                    {
                        "name": datamodule.idx_to_class[idx],
                        "id": idx,
                        # "color": wandb.utils.generate_color(idx),
                    }
                    for idx in datamodule.idx_to_class.keys()
                ]
            )
        else:
            self.wandb_classes = None

        # self.trainer_dataloader_count = len(trainer.train_dataloader) if trainer.train_dataloader else 0
        # self.val_dataloader_count = len(trainer.val_dataloaders[0]) if trainer.val_dataloaders else 0
        # if self.trainer_dataloader_count == 0 or self.val_dataloader_count == 0:
        #     raise ValueError("Dataloaders are empty. Please ensure you have data loaded in your datamodule.")

    # def on_fit_start(self, trainer, pl_module):
    #     return super().on_fit_start(trainer, pl_module)

    def _wandb_log_image(self, prefix, x, y):
        """
        Logs a batch of images to Weights & Biases (wandb) with associated bounding boxes and class labels.

        Args:
            prefix (str): Prefix to use for the logged image keys.
            x (Tensor): Batch of images to log.
            y (Tensor): Corresponding predictions or annotations for each image in the batch.

        Notes:
            - Each image is logged with its bounding boxes and class labels.
            - The function assumes `translate_pred_to_wandb_boxes` converts predictions to wandb-compatible box format.
            - The `wandb_classes` attribute should contain the class labels for visualization.

        """
        self._wandb.log(
            {
                f"{prefix}_{idx}": wandb.Image(
                    x[idx].detach().cpu(),
                    # TODO remove passing in idx to class?
                    boxes=translate_pred_to_wandb_boxes(y[idx], x[idx]),
                    classes=self.wandb_classes,
                )
                for idx in range(len(x))
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called at the end of each training batch during model training.

        Logs input images and their corresponding labels to Weights & Biases (WandB) at specified intervals,
        determined by `train_log_frequency`. Only logs when `batch_idx` is 0 and the current epoch is a multiple
        of `train_log_frequency`.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The LightningModule being trained.
            outputs: The outputs from the training step.
            batch: The current batch of data, typically a tuple (x, y) of inputs and labels.
            batch_idx (int): Index of the current batch within the epoch.
            dataloader_idx (int, optional): Index of the dataloader, defaults to 0.

        Notes:
            - The method can be extended to mutate input images using model attributes if available.
            - Images and labels are logged using the `_wandb_log_image` helper method.

        """
        # TODO model could mutate input images. maybe do that too? or have train return it?
        if batch_idx == 0 and pl_module.current_epoch % self.train_log_frequency == 0:
            x, y = batch
            # if hasattr(pl_module, 'patch_image_mutator'):
            #     x = pl_module.patch_image_mutator(x)
            # if hasattr(pl_module, 'patched_image_mutator'):
            #     x = pl_module.patched_image_mutator(x)
            # if hasattr(pl_module, 'combiner'):
            #     x = pl_module.combiner(x)
            # if hasattr(pl_module, 'model'):
            #     y = pl_module.model(x)
            self._wandb_log_image(f"train_{dataloader_idx}/patch_image", x, y)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """
        Called at the end of each validation batch during model training.

        Logs validation patch images to Weights & Biases (WandB) at the start of each epoch,
        based on the specified validation log frequency.

        Args:
            trainer: The PyTorch Lightning Trainer instance.
            pl_module: The LightningModule being trained.
            outputs: The outputs from the validation step.
            batch: The current batch of data, typically a tuple (x, y).
            batch_idx (int): Index of the current batch within the epoch.
            dataloader_idx (int, optional): Index of the dataloader. Defaults to 0.

        """
        if batch_idx == 0 and pl_module.current_epoch % self.val_log_frequency == 0:
            x, y = batch
            self._wandb_log_image("val/patch_image", x, y)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Logs the patch image to Weights & Biases (wandb) at intervals defined by `self.train_log_frequency`.
        If the `pl_module` has a `patch` attribute, the patch image is logged using `wandb.Image`.
        If the `patch` attribute is not present, no image is logged.

        Args:
            trainer: The PyTorch Lightning trainer instance.
            pl_module: The PyTorch Lightning module being trained.

        Note:
            Logging occurs only if the current epoch is divisible by `self.train_log_frequency`.

        """
        if pl_module.current_epoch % self.train_log_frequency == 0:
            # Log the patch image at the end of the training epoch
            if hasattr(pl_module, "patch"):
                self._wandb.log(
                    {
                        "patch/image": wandb.Image(
                            # TODO is clone needed?
                            pl_module.patch.clone().detach().cpu(),
                        ),
                    }
                )
            else:
                pass
                # TODO Log a warning if the patch is not available


# TODO copy Timer callback way of doing this
class LogPerformanceCallback(Callback):
    """
    Callback to log performance metrics during model training using PyTorch Lightning.

    This callback tracks and logs the following timing metrics:
    - Total fit time (from start to stop of the fit process)
    - Total training time (from start to stop of training)
    - Training epoch time (duration of each training epoch)

    Metrics are logged to the LightningModule using the `log` method, with the following keys:
    - "performance/fit_time"
    - "performance/train_time"
    - "performance/train_epoch_time"

    All logging is performed only on rank zero in distributed settings.

    Methods:
        on_fit_start(trainer, pl_module): Records the start time of the fit process.
        on_fit_stop(trainer, pl_module): Logs the total fit time.
        on_train_start(trainer, pl_module): Records the start time of training.
        on_train_stop(trainer, pl_module): Logs the total training time.
        on_train_epoch_start(trainer, pl_module): Records the start time of the training epoch.
        on_train_epoch_end(trainer, pl_module): Logs the duration of the training epoch.

    """

    def __init__(self):
        """
        Constructor for LogPerformanceCallback.

        Initializes timing attributes for tracking the start times of various phases:
        - start_time: Overall start time.
        - forward_start_time: Start time of the forward pass.
        - backward_start_time: Start time of the backward pass.
        - train_epoch_start_time: Start time of the training epoch.
        """
        super().__init__()
        self.start_time = 0.0
        self.forward_start_time = 0.0
        self.backward_start_time = 0.0
        self.train_epoch_start_time = 0.0

    @rank_zero_only
    def on_fit_start(self, trainer: Trainer, pl_module: L.LightningModule):
        """
        Called at the beginning of the model fitting process.

        Args:
            trainer (Trainer): The Trainer instance managing the training loop.
            pl_module (L.LightningModule): The LightningModule being trained.

        Side Effects:
            Initializes the fit_time attribute with the current time to track the duration of the fitting process.

        """
        self.fit_time = time.time()

    @rank_zero_only
    def on_fit_stop(self, trainer: Trainer, pl_module: L.LightningModule):
        """
        Called when the fitting process stops.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance managing the training loop.
            pl_module (L.LightningModule): The LightningModule being trained.

        Logs the total fitting time under the metric name "performance/fit_time".

        """
        fit_time = time.time() - self.fit_time
        pl_module.log("performance/fit_time", fit_time, on_step=False, on_epoch=True)

    @rank_zero_only
    def on_train_start(self, trainer: Trainer, pl_module: L.LightningModule):
        """
        Called at the beginning of the training process.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance managing the training loop.
            pl_module (L.LightningModule): The LightningModule being trained.

        Side Effects:
            Initializes and stores the start time of training in `self.start_time`.

        """
        self.start_time = time.time()

    @rank_zero_only
    def on_train_stop(self, trainer: Trainer, pl_module: L.LightningModule):
        """
        Called when the training process stops.

        Args:
            trainer (Trainer): The PyTorch Lightning Trainer instance managing the training loop.
            pl_module (L.LightningModule): The LightningModule being trained.

        Logs the total training time under the key "performance/train_time" at the end of the epoch.

        """
        train_time = time.time() - self.start_time
        pl_module.log("performance/train_time", train_time, on_step=False, on_epoch=True)

    @rank_zero_only
    def on_train_epoch_start(self, trainer, pl_module):
        """
        Called at the start of each training epoch.

        Args:
            trainer: The PyTorch Lightning Trainer instance managing the training process.
            pl_module: The LightningModule being trained.

        Side Effects:
            Records the current time as the start time of the training epoch in `self.train_epoch_start_time`.

        """
        self.train_epoch_start_time = time.time()

    @rank_zero_only
    def on_train_epoch_end(self, trainer, pl_module):
        """
        Called at the end of each training epoch.

        Args:
            trainer: The PyTorch Lightning Trainer instance managing the training loop.
            pl_module: The LightningModule being trained.

        Calculates the duration of the training epoch and logs it under the key
        "performance/train_epoch_time" for monitoring purposes.

        """
        train_epoch_time = time.time() - self.train_epoch_start_time
        pl_module.log("performance/train_epoch_time", train_epoch_time, on_step=False, on_epoch=True)
