import argparse
import yaml
import os
import random
import logging

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import OmegaConf

from protores.utils.onnx_export import ModelExport
from protores.utils.tensorboard import TensorBoardLoggerWithMetrics
from protores.utils.model_factory import ModelFactory
from protores.utils.options import BaseOptions
from protores.utils.versioning import get_git_diff

from hydra.experimental import compose, initialize
from sklearn.model_selection import ParameterGrid
from protores.utils.checkpointing import set_latest_checkpoint

# register models
import protores.models


def run(cfg: BaseOptions):
    # resolve variable interpolation
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)
    OmegaConf.set_struct(cfg, True)

    print(OmegaConf.to_yaml(cfg))

    if not torch.cuda.is_available():
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! CUDA is NOT available !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # Create data module
    # Note: we need to manually process the data module in order to get a valid skeleton
    # We don't call setup() as this breaks ddp_spawn training for some reason
    # Calling setup() is not required here as we only need access to the skeleton data, achieved with get_skeleton()
    dm = instantiate(cfg.dataset)
    dm.prepare_data()

    # create model
    model = ModelFactory.instantiate(cfg, skeleton=dm.get_skeleton())

    # setup logging
    tb_logger = TensorBoardLoggerWithMetrics(save_dir=cfg.logging.path,
                                             name=cfg.logging.name,
                                             version=cfg.logging.version)
    print("Logging saved to: " + tb_logger.log_dir)
    all_loggers = [tb_logger]

    # setup callbacks
    callbacks = []
    callbacks.append(ModelCheckpoint(dirpath=tb_logger.log_dir + '/checkpoints', save_top_k=None, monitor=None, mode="min"))
    if cfg.logging.export_period > 0:
        callbacks.append(ModelExport(dirpath=tb_logger.log_dir + '/exports', filename=cfg.logging.export_name, period=cfg.logging.export_period))

    # Set random seem to guarantee proper working of distributed training
    # Note: we do it just before instantiating the trainer to guarantee nothing else will break it
    rnd_seed = cfg.seed
    random.seed(rnd_seed)
    np.random.seed(rnd_seed)
    torch.manual_seed(rnd_seed)
    print("Random Seed: ", rnd_seed)
    
    # Save Git diff for reproducibility
    current_log_path = os.path.normpath(os.getcwd() + "/./" + tb_logger.log_dir)
    logging.info("Logging saved to: %s" % current_log_path)
    if not os.path.exists(current_log_path):
        os.makedirs(current_log_path)
    git_diff = get_git_diff()
    if git_diff != "":
        with open(current_log_path + "/git_diff.patch", 'w') as f:
            f.write(git_diff)

    # training
    trainer = pl.Trainer(logger=all_loggers, callbacks=callbacks, **cfg.trainer)
    trainer.fit(model, datamodule=dm)

    # export
    model.export(tb_logger.log_dir + '/model.onnx')

    # test
    trainer.test(datamodule=dm, model=model)


def main(filepath: str, overrides: list = []):
    with open(filepath) as f:
        experiment_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    config_path = "protores/configs"
    initialize(config_path=config_path)

    base_config = experiment_cfg["base_config"]
    experiment_params = experiment_cfg["parameters"]
    for k in experiment_params:
        if not isinstance(experiment_params[k], list):
            experiment_params[k] = [experiment_params[k]]

    param_grid = ParameterGrid(experiment_params)
    for param_set in param_grid:
        param_overrides = []

        for k in param_set:
            param_overrides.append(k + "=" + str(param_set[k]))

        # add global overrides last
        param_overrides += overrides

        cfg = compose(base_config + ".yaml", overrides=param_overrides)

        set_latest_checkpoint(cfg)

        run(cfg.model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False, description="Experiment")
    parser.add_argument('--config', type=str, help='Path to the experiment configuration file', required=True)
    parser.add_argument("overrides", nargs="*",
                        help="Any key=value arguments to override config values (use dots for.nested=overrides)", )
    args = parser.parse_args()

    main(filepath=args.config, overrides=args.overrides)
