from typing import Callable, Optional

import pytorch_lightning as pl
from hydra.core.config_store import ConfigStore
from protores.utils.options import BaseOptions
from protores.utils.python import get_full_class_reference


class ModelFactory:
    registry = {}

    @classmethod
    def register(cls, opts_class: BaseOptions, schema_name: Optional[str] = None) -> Callable:
        def inner_wrapper(wrapped_class: pl.LightningModule) -> Callable:
            fully_qualified_name = get_full_class_reference(wrapped_class)
            if schema_name is None:
                name = fully_qualified_name
            else:
                name = schema_name

            opts_class._target_ = name
            assert name not in cls.registry, ('Model %s is already registered (class: %s)' % (name, fully_qualified_name))

            cls.registry[name] = wrapped_class

            cs = ConfigStore.instance()
            cs.store(group="schema/model", name=name, node=opts_class, package="model")

            return wrapped_class

        return inner_wrapper

    @classmethod
    def instantiate(cls, opts: BaseOptions, **kwargs) -> pl.LightningModule:
        name = opts._target_
        assert name in cls.registry, ('Model %s is not registered' % name)

        model_class = cls.registry[name]
        model = model_class(opts=opts, **kwargs)

        return model
