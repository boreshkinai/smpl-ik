from dataclasses import dataclass
from typing import Optional, Union, List, Any

from omegaconf import MISSING

from protores.utils.versioning import get_git_commit_id


@dataclass
class VersioningInfo:
    commit_id: str = get_git_commit_id()


# refer to https://pytorch-lightning.readthedocs.io/en/latest/trainer.html# for more information
@dataclass
class TrainerOptions:
    checkpoint_callback: bool = True
    default_root_dir: Optional[str] = None
    gradient_clip_val: float = 0
    process_position: int = 0
    num_nodes: int = 1
    num_processes: int = 1
    gpus: Optional[Any] = 1
    auto_select_gpus: bool = False
    tpu_cores: Optional[Any] = None
    log_gpu_memory: Optional[str] = None
    progress_bar_refresh_rate: int = 1
    overfit_batches: Any = 0.0
    track_grad_norm: Any = -1
    check_val_every_n_epoch: int = 1
    fast_dev_run: bool = False
    accumulate_grad_batches: Any = 1
    max_epochs: int = 1000
    min_epochs: int = 1
    max_steps: Optional[int] = None
    min_steps: Optional[int] = None
    limit_train_batches: Any = 1.0
    limit_val_batches: Any = 1.0
    limit_test_batches: Any = 1.0
    val_check_interval: Any = 1.0
    flush_logs_every_n_steps: int = 100
    log_every_n_steps: int = 50
    accelerator: Optional[Any] = None
    sync_batchnorm: bool = False
    precision: int = 32
    weights_summary: Optional[str] = 'top'
    weights_save_path: Optional[str] = None
    num_sanity_val_steps: int = 2
    truncated_bptt_steps: Optional[int] = None
    resume_from_checkpoint: Optional[str] = None
    profiler: Optional[Any] = None
    benchmark: bool = False
    deterministic: bool = False
    reload_dataloaders_every_epoch: bool = False
    auto_lr_find: Any = False
    replace_sampler_ddp: bool = True
    terminate_on_nan: bool = False
    auto_scale_batch_size: Any = False
    prepare_data_per_node: bool = True
    amp_backend: str = 'native'
    amp_level: str = 'O2'
    distributed_backend: Optional[str] = None
    automatic_optimization: bool = True


@dataclass
class LoggingOptions:
    name: str = ""
    path: str = ""
    version: str = "./"
    export_period: int = 0  # model is exported to ONNX each period epochs
    export_name: Optional[str] = None  # filename for ONNX export during training. Use {0} to indicate epoch count
    

@dataclass
class BaseOptions:
    _target_: str = MISSING  # the model class identifier, ie the full reference to the model class
    logging: LoggingOptions = LoggingOptions()
    version: VersioningInfo = VersioningInfo()
    trainer: TrainerOptions = TrainerOptions()
    seed: int = 0  # the seed being used to intialized random generators. Required to ensure distributed training works properly
    dataset: Any = None

