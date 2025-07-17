from unittest.mock import MagicMock, patch

import pytest
from lightning.pytorch.loggers import WandbLogger

from shok.utils.callbacks import WandbObjectDetectionCallback


@pytest.fixture
def mock_wandb_logger():
    mock_logger = MagicMock(spec=WandbLogger)
    mock_logger.experiment = MagicMock()
    return mock_logger


@pytest.fixture
def mock_datamodule():
    dm = MagicMock()
    dm.idx_to_class = {0: "cat", 1: "dog"}
    return dm


@pytest.fixture
def mock_trainer(mock_datamodule):
    trainer = MagicMock()
    trainer.datamodule = mock_datamodule
    return trainer


@pytest.fixture
def mock_pl_module(mock_wandb_logger):
    pl_module = MagicMock()
    pl_module.logger = mock_wandb_logger
    return pl_module


def test_setup_success(mock_trainer, mock_pl_module):
    callback = WandbObjectDetectionCallback()
    with patch("shok.utils.callbacks.wandb.Classes") as mock_classes:
        callback.setup(mock_trainer, mock_pl_module)
        assert callback._wandb is mock_pl_module.logger.experiment
        assert callback.idx_to_class == mock_trainer.datamodule.idx_to_class
        mock_classes.assert_called_once()
        assert callback.wandb_classes == mock_classes.return_value


def test_setup_no_logger(mock_trainer):
    pl_module = MagicMock()
    pl_module.logger = None
    callback = WandbObjectDetectionCallback()
    with pytest.raises(ValueError, match="Wandb logger is not set up"):
        callback.setup(mock_trainer, pl_module)


def test_setup_no_datamodule(mock_pl_module):
    trainer = MagicMock()
    trainer.datamodule = None
    callback = WandbObjectDetectionCallback()
    with pytest.raises(ValueError, match="Datamodule is not set up"):
        callback.setup(trainer, mock_pl_module)


def test_setup_no_idx_to_class(mock_trainer, mock_pl_module):
    mock_trainer.datamodule.idx_to_class = None
    callback = WandbObjectDetectionCallback()
    with patch("shok.utils.callbacks.wandb.Classes"):
        callback.setup(mock_trainer, mock_pl_module)
        assert callback.idx_to_class == {}
        assert callback.wandb_classes is None
