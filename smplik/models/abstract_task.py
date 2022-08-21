import base64
import errno
import logging
import os
from typing import Dict, Any
import json
import torch
import matplotlib.figure as fig
import matplotlib.pyplot as plt
from omegaconf import DictConfig
import pytorch_lightning as pl

from smplik.utils.onnx_export import export_named_model_to_onnx
from smplik.utils.options import BaseOptions

try:
    import neural_graph
    USE_NEURAL_GRAPH = True
except:
    USE_NEURAL_GRAPH = False

try:
    import wandb
    USE_WANDB = True
except:
    USE_WANDB = False


# Note: as the base task currently does not have any additional option
# we define it as an alias as a dataclass cannot be empty
AbstractTaskOptions = BaseOptions


class AbstractTask(pl.LightningModule):
    @staticmethod
    def get_metrics() -> Dict[str, Any]:
        return {}

    def __init__(self, opts: AbstractTaskOptions):
        super().__init__()

        assert isinstance(opts, DictConfig), f"opt constructor argument must be of type DictConfig but got {type(opts)} instead."

        self.save_hyperparameters(opts)

    def on_train_start(self) -> None:
        super().on_train_start()
        self.log_ng_metadata()

    def on_train_epoch_start(self) -> None:
        try:
            dataset = self.trainer.train_dataloader.dataset
            dataset.set_epoch(self.current_epoch)
        except Exception:
            pass
        return super().on_train_epoch_start()

    def log_train_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("train/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def log_validation_losses(self, losses: Dict[str, Any], prefix: str = ""):
        for k, v in losses.items():
            if v is not None:
                self.log("validation/" + prefix + k, v, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

    def log_ng_tensor(self, tag: str, tensor: torch.Tensor):
        tensors = {}
        tensors[tag] = tensor
        self.log_ng_tensors(tensors=tensors)

    def log_ng_tensors(self, tensors: Dict[str, torch.Tensor]):
        if USE_NEURAL_GRAPH:
            # Avoid infinite loop: when model.trainer.fast_dev_run=True, experiment iterates on itself
            experiments = self.logger.experiment if isinstance(self.logger.experiment, list) else [self.logger.experiment]
            for experiment in experiments:
                if isinstance(experiment, neural_graph.logging.unity_summary_writer.UnitySummaryWriter):
                    for tag in tensors:
                        experiment.add_tensor(tag, tensors[tag].double(), self.global_step)

    def log_ng_metadata(self):
        metadata = self.get_metadata()
        metadata_text = json.dumps(metadata)
        self.log_ng_text("Metadata", metadata_text, 0.0)

    def log_ng_text(self, tag: str, text: str, time: float):
        if USE_NEURAL_GRAPH:
            # Avoid infinite loop: when model.trainer.fast_dev_run=True, experiment iterates on itself
            experiments = self.logger.experiment if isinstance(self.logger.experiment, list) else [self.logger.experiment]
            for experiment in experiments:
                if isinstance(experiment, neural_graph.logging.unity_summary_writer.UnitySummaryWriter):
                    experiment.add_text(tag, text, time)

    def log_histogram(self, tag: str, tensor: torch.Tensor):
        tensors = {}
        tensors[tag] = tensor
        self.log_histograms(tensors=tensors)

    def log_figures(self, tensors: Dict[str, fig.Figure]):
        # Avoid infinite loop: when model.trainer.fast_dev_run=True, experiment iterates on itself
        experiments = self.logger.experiment if isinstance(self.logger.experiment, list) else [self.logger.experiment]
        for experiment in experiments:
            if isinstance(experiment, torch.utils.tensorboard.writer.SummaryWriter):
                for tag, figure in tensors.items():
                    experiment.add_figure(tag, figure, self.global_step, close=True)

            elif USE_WANDB and isinstance(experiment, wandb.sdk.wandb_run.Run):
                for tag, figure in tensors.items():
                    experiment.log({tag: figure})

            elif USE_NEURAL_GRAPH and isinstance(experiment, neural_graph.logging.unity_summary_writer.UnitySummaryWriter):
                pass

        # Close those things!
        for k, v in tensors.items():
            plt.close(v)

    def get_dummy_input(self):
        return {}

    def get_dummy_output(self):
        return {}

    def get_dynamic_axes(self):
        return {}

    def get_metadata(self):
        return {
            "model": type(self).__name__
        }

    def export(self, filepath: str, **kwargs):
        dirpath = os.path.dirname(filepath)
        if not os.path.exists(dirpath):
            try:
                os.makedirs(dirpath)
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        dummy_input = self.get_dummy_input()
        dynamic_axes = self.get_dynamic_axes()
        metadata = self.get_metadata()
        metadata_json = {"json": json.dumps(metadata)}
        export_named_model_to_onnx(self, dummy_input, filepath, metadata=metadata_json, dynamic_axes=dynamic_axes, verbose=True, **kwargs)
