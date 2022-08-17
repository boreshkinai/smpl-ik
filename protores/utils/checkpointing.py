from glob import glob
import os
from protores.utils.options import BaseOptions
from omegaconf import OmegaConf
import yaml


def get_latest_checkpoint(checkpoint_dir: str = None) -> str:
    if checkpoint_dir is None:
        return None
    checkpoints = glob(checkpoint_dir)

    checkpoint_epochs = {c.split("=")[-1].split(".")[0]: c for c in checkpoints}
    checkpoint_epochs = {int(c): dir for c, dir in checkpoint_epochs.items() if c.isdigit()}
    if len(checkpoint_epochs) == 0:
        return None

    max_epoch = max(checkpoint_epochs.keys())
    if os.path.isfile(checkpoint_epochs[max_epoch]):
        return checkpoint_epochs[max_epoch]
    else:
        return None


def set_latest_checkpoint(cfg: BaseOptions) -> None:
    if cfg.model.trainer.resume_from_checkpoint is None:

        # If there is no previous config file, we cannot restart
        job_directory = os.path.join(cfg.model.logging.path, cfg.model.logging.name)
        if not os.path.isfile(os.path.join(job_directory, "hparams.yaml")):
            return

        # Validate that the model configuration is the same before restarting, if not abort
        with open(os.path.join(job_directory, "hparams.yaml")) as f:
            job_cfg = yaml.load(f, Loader=yaml.SafeLoader)

        cfg_model = OmegaConf.to_container(cfg.model, resolve=True)
        cfg_model = OmegaConf.create(cfg_model)
        OmegaConf.set_struct(cfg_model, True)

        if cfg_model != job_cfg:
            print(cfg_model)
            print(job_cfg)
            print("Configuration differences:")

            all_keys = list(set(list(cfg_model.keys()) + list(job_cfg.keys())))
            for k in all_keys:
                v_old = job_cfg.get(k, None)
                v_new = cfg_model.get(k, None)
                if v_old != v_new:
                    print(f"Key: {k}")
                    print(f"new value: {v_new}")
                    print(f"old value: {v_old}")

            message = f"You attempt automatically restarting from existing job {job_directory}. However, the configuration saved with the job 'hparams.yaml' is different from your current configuration. Aborting."
            assert False, message

        # Find checkpoints and reload
        checkpoint_dir = os.path.join(job_directory, "checkpoints/*.ckpt")
        cfg.model.trainer.resume_from_checkpoint = get_latest_checkpoint(checkpoint_dir=checkpoint_dir)

