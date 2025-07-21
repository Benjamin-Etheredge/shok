"""
PyTorch Lightning callback for computing mean Average Precision (mAP) using torchmetrics.

This module provides a callback for object detection models that computes and logs
mAP metrics during training and validation using torchmetrics.detection.MeanAveragePrecision.
"""

from typing import Any

import lightning as L
import torch
from lightning.pytorch.callbacks import Callback
from torchmetrics.detection import MeanAveragePrecision


class MeanAveragePrecisionCallback(Callback):
    """
    A PyTorch Lightning callback for computing mean Average Precision (mAP) metrics for object detection tasks.

    This callback uses torchmetrics.detection.MeanAveragePrecision to compute various mAP metrics
    including mAP@0.5, mAP@0.75, and mAP@0.5:0.95 for both training and validation phases.

    Supports multiple validation dataloaders by automatically creating separate metrics for each dataloader
    and logging them with appropriate suffixes (e.g., val_dl0_map, val_dl1_map).

    Args:
        iou_thresholds (Optional[List[float]]): IoU thresholds to compute mAP at.
            If None, defaults to [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95].
        rec_thresholds (Optional[List[float]]): Recall thresholds to compute mAP at.
            If None, uses 101 linearly spaced values between 0 and 1.
        max_detection_thresholds (Optional[List[int]]): Maximum detection thresholds.
            If None, defaults to [1, 10, 100].
        class_metrics (bool): Whether to compute per-class metrics. Default: False.
        compute_on_cpu (bool): Whether to compute metrics on CPU to save GPU memory. Default: True.
        log_on_step (bool): Whether to log metrics on each step. Default: False.
        log_on_epoch (bool): Whether to log metrics at the end of each epoch. Default: True.
        sync_dist (bool): Whether to synchronize metrics across distributed processes. Default: True.
        prefix (str): Prefix to add to logged metric names. Default: "".

    Example:
        ```python
        from shok.utils.callbacks.map import MeanAveragePrecisionCallback

        # Basic usage
        map_callback = MeanAveragePrecisionCallback()

        # Custom IoU thresholds
        map_callback = MeanAveragePrecisionCallback(
            iou_thresholds=[0.5, 0.75],
            class_metrics=True,
            prefix="detection_"
        )

        # Add to trainer - automatically handles multiple validation dataloaders
        trainer = L.Trainer(callbacks=[map_callback])
        ```

    Note:
        - Expects model outputs to be in the format: List[Dict[str, torch.Tensor]]
          where each dict contains 'boxes', 'scores', 'labels' keys
        - Expects targets to be in the format: List[Dict[str, torch.Tensor]]
          where each dict contains 'boxes', 'labels' keys
        - Boxes should be in xyxy format (x1, y1, x2, y2)
        - Multiple validation dataloaders are automatically detected and handled with separate metrics

    """

    def __init__(
        self,
        iou_thresholds: list[float] | None = None,
        rec_thresholds: list[float] | None = None,
        max_detection_thresholds: list[int] | None = None,
        class_metrics: bool = False,
        compute_on_cpu: bool = True,
        log_on_step: bool = False,
        log_on_epoch: bool = True,
        sync_dist: bool = True,
        prefix: str = "",
    ):
        super().__init__()

        self.iou_thresholds = iou_thresholds
        self.rec_thresholds = rec_thresholds
        self.max_detection_thresholds = max_detection_thresholds
        self.class_metrics = class_metrics
        self.compute_on_cpu = compute_on_cpu
        self.log_on_step = log_on_step
        self.log_on_epoch = log_on_epoch
        self.sync_dist = sync_dist
        self.prefix = prefix

        # Metrics will be initialized in setup()
        self.train_map: MeanAveragePrecision | None = None
        self.val_map: MeanAveragePrecision | torch.nn.ModuleList | None = None

    @property
    def _using_multiple_val_dataloaders(self) -> bool:
        """Check if we're using multiple validation dataloaders."""
        return isinstance(self.val_map, torch.nn.ModuleList)

    def setup(self, trainer: L.Trainer, pl_module: L.LightningModule, stage: str | None = None) -> None:
        """Initialize the mAP metrics."""
        device = "cpu" if self.compute_on_cpu else pl_module.device

        metric_kwargs = {
            "iou_thresholds": self.iou_thresholds,
            "rec_thresholds": self.rec_thresholds,
            "max_detection_thresholds": self.max_detection_thresholds,
            "class_metrics": self.class_metrics,
            "sync_on_compute": self.sync_dist,
            "backend": "faster_coco_eval",  # Use faster-coco-eval backend
        }

        # Initialize training metrics
        self.train_map = MeanAveragePrecision(**metric_kwargs).to(device)

        # Initialize validation metrics - support multiple dataloaders
        # Get number of validation dataloaders
        num_val_dataloaders = 1
        if hasattr(trainer, "val_dataloaders") and trainer.val_dataloaders is not None:
            if isinstance(trainer.val_dataloaders, list | tuple):
                num_val_dataloaders = len(trainer.val_dataloaders)

        # Create metrics for each validation dataloader
        if num_val_dataloaders > 1:
            self.val_map = torch.nn.ModuleList(
                [MeanAveragePrecision(**metric_kwargs).to(device) for _ in range(num_val_dataloaders)]
            )
        else:
            self.val_map = MeanAveragePrecision(**metric_kwargs).to(device)

    # def on_train_batch_end(
    #     self,
    #     trainer: L.Trainer,
    #     pl_module: L.LightningModule,
    #     outputs: Any,
    #     batch: Any,
    #     batch_idx: int
    # ) -> None:
    #     """Update training mAP metrics at the end of each training batch."""
    #     if self.train_map is None:
    #         return

    #     # Extract predictions and targets from batch
    #     preds, targets = self._extract_preds_and_targets(outputs, batch, pl_module)

    #     if preds is not None and targets is not None:
    #         # Move to appropriate device if needed
    #         if self.compute_on_cpu:
    #             preds = self._move_to_cpu(preds)
    #             targets = self._move_to_cpu(targets)

    #         self.train_map.update(preds, targets)

    #     # Log on step if requested
    #     if self.log_on_step:
    #         metrics = self.train_map.compute()
    #         self._log_metrics(pl_module, metrics, "train", on_step=True, on_epoch=False)

    def on_validation_batch_end(
        self,
        trainer: L.Trainer,
        pl_module: L.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Update validation mAP metrics at the end of each validation batch."""
        if self.val_map is None:
            return

        # Extract predictions and targets from batch
        preds, targets = self._extract_preds_and_targets(outputs, batch, pl_module)

        if preds is not None and targets is not None:
            # Move to appropriate device if needed
            if self.compute_on_cpu:
                preds = self._move_to_cpu(preds)
                targets = self._move_to_cpu(targets)

            # Handle multiple dataloaders
            if self._using_multiple_val_dataloaders:
                val_map_list = self.val_map
                if not isinstance(val_map_list, torch.nn.ModuleList):
                    msg = "Expected val_map to be ModuleList when using multiple dataloaders"
                    raise TypeError(msg)
                if dataloader_idx < len(val_map_list):
                    val_map_list[dataloader_idx].update(preds, targets)
            else:
                val_map_single = self.val_map
                if not isinstance(val_map_single, MeanAveragePrecision):
                    msg = "Expected val_map to be MeanAveragePrecision when using single dataloader"
                    raise TypeError(msg)
                val_map_single.update(preds, targets)

    # def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
    #     """Compute and log training mAP metrics at the end of each training epoch."""
    #     if self.train_map is None or not self.log_on_epoch:
    #         return

    #     metrics = self.train_map.compute()
    #     self._log_metrics(pl_module, metrics, "train", on_step=False, on_epoch=True)
    #     self.train_map.reset()

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule) -> None:
        """Compute and log validation mAP metrics at the end of each validation epoch."""
        if self.val_map is None or not self.log_on_epoch:
            return

        # Handle multiple dataloaders
        if self._using_multiple_val_dataloaders:
            val_map_list = self.val_map
            if not isinstance(val_map_list, torch.nn.ModuleList):
                msg = "Expected val_map to be ModuleList when using multiple dataloaders"
                raise TypeError(msg)
            # Log metrics for each dataloader
            for dataloader_idx in range(len(val_map_list)):
                val_metric = val_map_list[dataloader_idx]
                if not isinstance(val_metric, MeanAveragePrecision):
                    msg = f"Expected val_metric at index {dataloader_idx} to be MeanAveragePrecision"
                    raise TypeError(msg)
                metrics = val_metric.compute()
                # Add dataloader suffix to metric names if multiple dataloaders
                dataloader_suffix = f"_dl{dataloader_idx}" if len(val_map_list) > 1 else ""
                self._log_metrics(pl_module, metrics, f"val{dataloader_suffix}", on_step=False, on_epoch=True)
                val_metric.reset()
        else:
            val_map_single = self.val_map
            if not isinstance(val_map_single, MeanAveragePrecision):
                msg = "Expected val_map to be MeanAveragePrecision when using single dataloader"
                raise TypeError(msg)
            metrics = val_map_single.compute()
            self._log_metrics(pl_module, metrics, "val", on_step=False, on_epoch=True)
            val_map_single.reset()

    def _extract_preds_and_targets(
        self, outputs: Any, batch: Any, pl_module: L.LightningModule
    ) -> tuple[list[dict[str, torch.Tensor]] | None, list[dict[str, torch.Tensor]] | None]:
        """
        Extract predictions and targets from model outputs and batch.

        This method tries to handle different output formats that might be returned
        by object detection models during training and validation.
        """
        try:
            # Case 1: During validation, outputs might be the predictions directly
            if isinstance(outputs, list) and len(outputs) > 0 and isinstance(outputs[0], dict):
                if "boxes" in outputs[0] and "scores" in outputs[0] and "labels" in outputs[0]:
                    preds = outputs
                    # Get targets from batch
                    if isinstance(batch, list | tuple) and len(batch) >= 2:
                        targets = batch[1]  # Assuming (images, targets) format
                    else:
                        targets = None
                    return preds, targets

            # Case 2: During training, outputs might contain loss dict and predictions
            if isinstance(outputs, dict):
                # Look for predictions in outputs
                if "predictions" in outputs:
                    preds = outputs["predictions"]
                elif "preds" in outputs:
                    preds = outputs["preds"]
                else:
                    # Try to get predictions from the model's forward pass
                    if isinstance(batch, list | tuple) and len(batch) >= 2:
                        images, targets = batch[0], batch[1]
                        # Set model to eval mode temporarily to get predictions
                        was_training = pl_module.training
                        pl_module.eval()
                        with torch.no_grad():
                            preds = pl_module(images)
                        if was_training:
                            pl_module.train()
                    else:
                        preds = None

                # Get targets from outputs or batch
                if "targets" in outputs:
                    targets = outputs["targets"]
                elif isinstance(batch, list | tuple) and len(batch) >= 2:
                    targets = batch[1]
                else:
                    targets = None

                return preds, targets

            # Case 3: Fallback - try to run model inference
            if isinstance(batch, list | tuple) and len(batch) >= 2:
                images, targets = batch[0], batch[1]
                was_training = pl_module.training
                pl_module.eval()
                with torch.no_grad():
                    preds = pl_module(images)
                if was_training:
                    pl_module.train()
                return preds, targets

        except Exception as e:
            pl_module.log("map_callback_error", 1.0)
            raise e
            print(f"Warning: Could not extract predictions and targets for mAP computation: {e}")

        return None, None

    def _move_to_cpu(self, data: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
        """Move tensor data to CPU."""
        cpu_data = []
        for item in data:
            cpu_item = {}
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    cpu_item[key] = value.cpu()
                else:
                    cpu_item[key] = value
            cpu_data.append(cpu_item)
        return cpu_data

    def _log_metrics(
        self,
        pl_module: L.LightningModule,
        metrics: dict[str, torch.Tensor],
        phase: str,
        on_step: bool = False,
        on_epoch: bool = True,
    ) -> None:
        """Log mAP metrics to the logger."""
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, torch.Tensor) and metric_value.numel() == 1:
                # Create full metric name with prefix and phase
                full_name = f"{self.prefix}{phase}_{metric_name}"

                pl_module.log(
                    full_name,
                    metric_value,
                    on_step=on_step,
                    on_epoch=on_epoch,
                    prog_bar=metric_name == "map",  # Show overall mAP in progress bar
                    logger=True,
                    sync_dist=self.sync_dist,
                )
