import torch
import torchmetrics
from lightning.pytorch import LightningModule
from torchvision.transforms import v2

from shok.utils.transforms import (
    ConvertToTVTensorBBoxes,
    PassRound,
    ScaleApplyPatch,
    ScaleGradTransform,
    ScaleImageValues,
    SoftRound,
)
from shok.utils.transforms.apply_patch import ApplyPatch

# TODO log average patch size, location, and scale
# TODO maybe log samples from ds to see how patch is being applied


class ObjectDetectionPatch(LightningModule):
    """
    ObjectDetectionPatch is a PyTorch Lightning module designed for adversarial patch training in object detection models.

    A PyTorch Lightning module for adversarial patch training in object detection models.
    This module manages the creation, optimization, and application of an adversarial patch to input images,
    with the goal of affecting object detection performance. It supports expectation over transformation (EOT)
    sampling, custom image mutators, gradient accumulation, and evaluation using mean average precision (mAP).
        model: The object detection model to attack. Its parameters are frozen during patch training.
        patch_shape (tuple[int, int, int], optional): Shape of the adversarial patch (channels, height, width).
            Defaults to (3, 2048, 2048).
        learning_rate (float, optional): Learning rate for patch optimization. Defaults to 20.0.
        targeted (bool, optional): Whether the attack is targeted. Defaults to False.
        verbose (bool, optional): If True, enables verbose logging. Defaults to True.
        clip_values (tuple[float, float] | None, optional): Min and max values to clip the patch. Defaults to (0, 255).
        use_y_hat (bool, optional): If True, uses model predictions as targets. Defaults to False.
        gamma (float, optional): Learning rate decay factor for the scheduler. Defaults to 0.995.
        base_image_mutator (v2.Compose | bool | None, optional): Transform for base images. Defaults to None.
        patched_image_mutator (v2.Compose | bool, optional): Transform for patched images. Defaults to False.
        eot_samples (int, optional): Number of EOT samples per batch. Defaults to 1.

    Attributes:
        patch (torch.nn.Parameter): The adversarial patch being optimized.
        patched_image_mutator: Transform applied to patched images.
        combiner: Callable for applying the patch to images.
        other_transforms: Compose of transforms applied after patching.
        eval_combiner: Compose of transforms for evaluation.
        eval_maps (torch.nn.ModuleList): List of mAP metrics for evaluation.
        automatic_optimization (bool): Disables automatic optimization for manual control.
        grad_accumulation_steps (int): Number of gradient accumulation steps.
        record_count (int): Counter for processed records.
        eot_samples (int): Number of EOT samples per batch.

    Methods:
        setup(stage: str): Sets up the module for training or evaluation.
        train(mode=True): Sets training/evaluation mode for the model and normalization layers.
        configure_optimizers(): Configures the optimizer and learning rate scheduler for patch optimization.
        on_validation_epoch_start(): Switches the model to evaluation mode at the start of validation.
        validation_step(batch, batch_idx, dataloader_idx): Performs a validation step and logs metrics.
        on_train_epoch_start(): Switches the model to training mode at the start of each epoch.
        eot_sample_batches(batch): Yields EOT-sampled batches for robust patch training.
        _should_update(batch_idx: int) -> bool: Determines if an optimizer update should occur.
        on_train_batch_start(batch, batch_idx): Updates record count at the start of each batch.
        on_train_epoch_end(): Increments EOT samples and logs at the end of each epoch.
        training_step(batch, batch_idx): Performs a training step with EOT sampling and manual backward.
        _update(): Applies optimizer step, clamps patch values, logs patch statistics, and resets gradients.

    Notes:
        - The module is designed for adversarial patch training and evaluation in object detection tasks.
        - Manual optimization is used for fine-grained control over patch updates.
        - Logging is performed for patch statistics, gradients, and evaluation metrics.
        - The patch is clipped to specified value ranges after each update.

    """

    def __init__(
        self,
        model,
        patch_shape: tuple[int, int, int] = (3, 2048, 2048),
        learning_rate: float = 20.0,
        targeted: bool = False,
        # summary_writer: str | bool | SummaryWriter = False,
        verbose: bool = True,
        clip_values: tuple[float, float] | None = (0, 255),
        use_y_hat: bool = False,
        gamma: float = 0.995,
        patch_combiner: torch.nn.Module | None = None,
        val_patch_combiner: torch.nn.Module | None = None,
        # base_image_mutator: torch.nn.Module | bool | None = None,  # this should probably be in data module
        patched_image_transforms: torch.nn.Module | None = None,
        val_patched_image_transforms: torch.nn.Module | None = None,
        # TODO make scheduler
        eot_samples: int = 1,
        eot_rate: int = 32,
        # TODO should this just get pulled/set from the trainer?
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "model",
                "summary_writer",
                "patch_combiner",
                "val_patch_combiner",
                "patched_image_transforms",
                "val_patched_image_transforms",
            ]
        )
        self.model = model

        # TODO test speed with and without this
        for param in self.model.parameters():
            param.requires_grad = False

        patch = (
            torch.randint(0, 255, size=patch_shape, requires_grad=True, dtype=torch.float32)
            / 255
            * (clip_values[1] - clip_values[0])
            + clip_values[0]
        )
        self.patch = torch.nn.Parameter(patch, requires_grad=True)

        # TODO use "erase" transforms for comparing patching vs no patching

        # TODO adjust all transforms to work with multiple inputs
        # TODO adjust combiner to work with multiple inputs
        if patch_combiner is None:
            patch_combiner = ApplyPatch()
        self.patch_combiner = patch_combiner

        if patched_image_transforms is None:
            patched_image_transforms = v2.Identity()

        # TODO assert transforms don't go outside clip values
        # These other transforms are needed for the generation
        self.patched_image_transforms = v2.Compose(
            [
                SoftRound(),
                patched_image_transforms,
                # SoftRound(),
                ScaleGradTransform(),
                ScaleImageValues(min=clip_values[0], max=clip_values[1]),
                ConvertToTVTensorBBoxes(),
                v2.SanitizeBoundingBoxes(),
            ]
        )

        if val_patch_combiner is None:
            val_patch_combiner = ScaleApplyPatch(0.25)
        self.val_patch_combiner = val_patch_combiner

        if val_patched_image_transforms is None:
            val_patched_image_transforms = v2.Compose(
                [
                    PassRound(),
                    ScaleImageValues(min=clip_values[0], max=clip_values[1]),
                    ConvertToTVTensorBBoxes(),
                    v2.SanitizeBoundingBoxes(),
                ]
            )
        self.val_patched_image_transforms = val_patched_image_transforms

        self.automatic_optimization = False

        self.eot_samples = eot_samples

        # TODO pull out or move to setup
        # TODO move to callback
        self.eval_maps = torch.nn.ModuleList(
            [
                torchmetrics.detection.MeanAveragePrecision(
                    iou_type="bbox",
                    backend="faster_coco_eval",
                ),
                torchmetrics.detection.MeanAveragePrecision(
                    iou_type="bbox",
                    backend="faster_coco_eval",
                ),
            ]
        )

    def train(self, mode=True):
        """
        Sets the module and its underlying model to training or evaluation mode.

        Args:
            mode (bool, optional): If True, sets the module to training mode;
                if False, sets to evaluation mode. Defaults to True.

        Returns:
            self: Returns the module instance.

        Notes:
            - Calls the superclass's `train` method.
            - Sets the underlying model to train or eval mode accordingly.
            - For certain normalization and attention layers, forces evaluation mode and disables gradient updates for their weights and biases when
            in training mode.
            - Contains commented-out code and TODOs for future cleanup.

        """
        # self.training = mode
        # return self
        super().train(mode)
        # TODO clean up now that bug was found
        if mode:
            self.model.train()
        else:
            self.model.eval()
        # self.model.train(mode)
        if mode:
            for param in self.model.parameters():
                if isinstance(
                    param,
                    (
                        torch.nn.BatchNorm2d,
                        torch.nn.GroupNorm,
                        torch.nn.InstanceNorm2d,
                        torch.nn.LayerNorm,
                        torch.nn.LocalResponseNorm,
                        torch.nn.SyncBatchNorm,
                        torch.nn.modules.MultiheadAttention,
                        # torch.nn.modules.dropout._Dropout,
                        torch.nn.modules.dropout._DropoutNd,
                        torch.nn.modules.batchnorm._BatchNorm,
                        # torch.nn.modules.normalization._GroupNorm,
                        # torch.nn.modules.normalization._InstanceNorm,
                        # torch.nn.modules.normalization._LayerNorm,
                        # torch.nn.modules.normalization._LocalResponseNorm,
                        # torch.nn.modules.normalization._SyncBatchNorm,
                    ),
                ):
                    param.eval()
                    param.weight.requires_grad = False
                    param.bias.requires_grad = False
        return self

        if mode:
            self.model.train()
        else:
            self.model.eval()

    # TODO explore overriding configure gradient clipping for scalling
    def configure_optimizers(self):
        """Configure the optimizer for the patch."""
        opt = torch.optim.Adam(
            [self.patch],
            lr=self.hparams.learning_rate,
            # TODO what will happen to patch if we minimize?
            # TODO could we grow the patch over time to make it take on params of image with minimize?
            maximize=True,
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=1,  # this only steps every update
            gamma=self.hparams.gamma,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": lr_scheduler,
        }

    def on_validation_epoch_start(self):
        """
        Called at the start of the validation epoch.

        Sets the model to evaluation mode by disabling training-specific behaviors
        such as dropout and batch normalization updates.
        """
        self.train(False)

    @staticmethod
    def _apply_patch(x, patch, y, combiner):
        """
        Applies the adversarial patch to the input images.

        TODO make combiner work on multiple inputs and targets with one call instead of a loop

        Args:
            x (List[torch.Tensor]): Input images.
            patch (torch.Tensor): Adversarial patch.
            y (List[torch.Tensor]): Corresponding targets.
            combiner (callable): Function to combine images and patch.

        Returns:
            tuple: Transformed images and targets after applying the patch.

        """
        x, y = zip(*[combiner(image, patch, target) for image, target in zip(x, y)])
        return x, y

    def validation_step(self, batch, batch_idx, dataloader_idx):
        """Validation step for the patch."""
        # TODO implement a validation step for the patch
        x, y = batch
        x, y = self._apply_patch(x, self.patch, y, self.val_patch_combiner)
        x, y = zip(*[self.val_patched_image_transforms(image, target) for image, target in zip(x, y)])

        detections = self.model(x, y)

        metrics = self.eval_maps[dataloader_idx](preds=detections, target=y)
        # TODO figure out how to log
        metrics["classes"]
        del metrics["classes"]  # Remove classes from metrics to avoid logging it
        metrics = {f"val_{dataloader_idx}_map/{k}": v for k, v in metrics.items()}
        # self.log_dict(self.eval_maps[dataloader_idx], on_step=False, on_epoch=True, prog_bar=False, add_dataloader_idx=True)
        self.log_dict(
            metrics,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=len(x),
            add_dataloader_idx=False,
        )

    # TODO include train model and eval model maybe?
    # TODO explore using truth labels vs model predictions

    def on_train_epoch_start(self):
        """
        Called at the start of each training epoch.

        Ensures the module is set to training mode by calling `self.train()`.
        Note: This method currently requires manual invocation to set the mode.
        """
        # TODO: why am I having to manually call this?
        self.train()

    def eot_sample_batches(self, batch):
        """
        Generates multiple transformed batches using EOT (Expectation Over Transformation) sampling.

        Args:
            batch (tuple): A tuple containing two lists:
                - batch[0]: List of input images.
                - batch[1]: List of corresponding targets.

        Yields:
            tuple: A tuple of two lists:
                - The first list contains transformed images.
                - The second list contains transformed targets.

        Each yielded batch is created by applying `self.combiner` and `self.other_transforms`
        to each image-target pair in the input batch, repeated `self.eot_samples` times.

        """
        for _ in range(self.eot_samples):
            eot_batch = ([], [])
            # new_x, new_y = self._apply_patch(image, self.patch, target, self.patch_combiner)
            for image, target in zip(batch[0], batch[1]):
                new_x, new_y = self.patch_combiner(image, self.patch, target)
                new_x, new_y = self.patched_image_transforms(new_x, new_y)
                eot_batch[0].append(new_x)
                eot_batch[1].append(new_y)
            yield eot_batch

    # TODO set function at start so no ifs
    # TODO use self.trainer.estimated_stepping_batches somewhere
    def _should_update(self, batch_idx: int) -> bool:
        """
        Determines whether an update should occur for the current batch.

        Args:
            batch_idx (int): The index of the current batch.

        Returns:
            bool: True if the current batch is the last batch and an update should occur, False otherwise.

        Note:
            The update is currently set to occur only on the last batch.
            Consider revisiting this logic if gradient accumulation steps are set to the number of batches.

        """
        return self.trainer.is_last_batch
        # TODO should this always update on last batch?
        # TODO if I set grad_accumulation_steps to be num of batches, no logic needed

    def on_train_epoch_end(self):
        """
        Called at the end of each training epoch.

        Increments the `eot_samples` counter every 16 epochs and logs its value.
        The logging is performed at the end of each epoch and displayed in the progress bar.

        Side Effects:
            - Updates `self.eot_samples` if the current epoch is a multiple of 16.
            - Logs the value of `eot_samples` with the key "eot_samples".
        """
        # TODO could just int divide by eot_rate, but this lets us start at any int
        if (self.current_epoch + 1) % self.hparams.eot_rate == 0:
            self.eot_samples += 1
        self.log("eot_samples", self.eot_samples, on_step=False, on_epoch=True, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """
        Performs a training step over a batch of data, applying multiple EOT (Expectation Over Transformation) samples per batch.

        For each EOT sample generated from the input batch, computes the model losses, logs individual loss components,
        aggregates the total loss, and applies manual backward propagation with loss scaling.
        Loss is scaled by the number of EOT samples and the total number of training batches to account for gradient accumulation.
        Triggers model update if the batch index meets update criteria.

        Args:
            batch: The input batch of data.
            batch_idx (int): The index of the current batch.

        Returns:
            None

        """
        # NOTE doing this here insures that the exact same images are used for each eot sample
        # TODO maybe pull this out and augement dataset and trainer instead
        for eot_batch in self.eot_sample_batches(batch):
            x, y = eot_batch
            losses = self.model(x, y)
            self.log_dict(
                losses,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
            )
            loss = sum(loss_value for loss_value in losses.values())
            self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, add_dataloader_idx=False)
            # TODO is this needed since we rescale gradients?
            loss /= self.eot_samples
            # loss /= self.grad_accumulation_steps
            loss /= self.trainer.num_training_batches
            self.manual_backward(loss)

        # TODO is the loss already scaled? should I manually compute the values?
        # TODO should just sum all loses and filter at model level

        if self._should_update(batch_idx):
            self._update()

    def _update(self):
        self.record_count = 0
        # NOTE decaying learning rate simulates sample_size increasing if we only use whole values
        opt = self.optimizers()
        lr_scheduler = self.lr_schedulers()

        self.log("lr", opt.param_groups[0]["lr"], on_step=False, on_epoch=True, prog_bar=False)

        num_pixel_updates = (torch.abs(torch.sign(self.patch.grad))).sum().item()
        self.log("patch_delta/num_pixel_updates", num_pixel_updates, on_step=False, on_epoch=True, prog_bar=True)

        old_patch = self.patch.clone().detach()

        opt.step()
        lr_scheduler.step()

        # NOTE: messing with gradients in the optimizer is not the best practice
        self.log("patch_grad/grad_norm", torch.norm(self.patch.grad), on_step=False, on_epoch=True)
        self.log("patch_grad/grad_mean", self.patch.grad.mean(), on_step=False, on_epoch=True)
        self.log("patch_grad/grad_abs_mean", torch.abs(self.patch.grad).mean(), on_step=False, on_epoch=True)
        self.log("patch_grad/grad_max", self.patch.grad.max(), on_step=False, on_epoch=True)
        self.log("patch_grad/grad_min", self.patch.grad.min(), on_step=False, on_epoch=True)

        opt.zero_grad()
        if self.hparams.clip_values is not None:
            with torch.no_grad():
                self.patch.data.clamp_(min=self.hparams.clip_values[0], max=self.hparams.clip_values[1])

        with torch.no_grad():
            patch_delta = torch.abs(self.patch - old_patch)

        patch_delta_mean = patch_delta.mean()
        self.log("patch_delta/delta_mean", patch_delta_mean, on_step=False, on_epoch=True)
        self.log("patch_delta/delta_max", patch_delta.max(), on_step=False, on_epoch=True)
        self.log("patch_delta/delta_min", patch_delta.min(), on_step=False, on_epoch=True)
        self.log("patch_delta/delta_std", patch_delta.std(), on_step=False, on_epoch=True)
        self.log("patch_delta/delta_sum", patch_delta.sum(), on_step=False, on_epoch=True)

        self.log("patch/avg_patch_value", self.patch.mean(), on_step=False, on_epoch=True)

        # TODO cut these since we clamp?
        if torch.sum(self.patch < 0) != 0:
            raise ValueError("Patch values should be non-negative")
        if torch.sum(self.patch > 255) != 0:
            raise ValueError("Patch values should be good")
