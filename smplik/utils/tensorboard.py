from argparse import Namespace
from typing import Optional, Union, List, Dict, Any

from pytorch_lightning.loggers import TensorBoardLogger


class TensorBoardLoggerWithMetrics(TensorBoardLogger):
    def __init__(
            self,
            save_dir: str,
            name: Optional[str] = "default",
            version: Optional[Union[int, str]] = None,
            log_graph: bool = False,
            metrics: Dict[str, Any] = {},
            **kwargs
    ):
        super().__init__(save_dir=save_dir, name=name, version=version, log_graph=log_graph, default_hp_metric=False, **kwargs)
        self.metrics = metrics

    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        super().log_hyperparams(params, self.metrics)
