from typing import Optional

import torch
from pytorch_lightning import Callback


class OnnxWrapper(torch.nn.Module):
    def __init__(self, module, dummy_input):
        super(OnnxWrapper, self).__init__()
        self.module = module

        for k in list(dummy_input.keys()):
            dummy_input[k] = dummy_input[k].to(module.device)

        self.dummy_output = module(dummy_input)
        self.input_names = list(dummy_input.keys())
        self.dummy_inputs = [dummy_input[k] for k in self.input_names]
        self.output_names = []
        for k, v in self.dummy_output.items():
            if v is not None:
                self.output_names.append(k)

    def forward(self, inputs):
        named_input = {}
        for i in range(len(inputs)):
            named_input[self.input_names[i]] = inputs[i].to(self.module.device)

        named_output = self.module(named_input)
        outputs = [named_output[k] for k in self.output_names]
        return outputs

    def export(self, filepath, *args, **kwargs):
        torch.onnx.export(self, self.dummy_inputs, filepath, input_names=self.input_names,
                          output_names=self.output_names, *args, **kwargs)


def export_named_model_to_onnx(module, dummy_input, filepath, *args, **kwargs):
    wrapped_module = OnnxWrapper(module, dummy_input)
    wrapped_module.export(filepath, *args, **kwargs)


class ModelExport(Callback):
    def __init__(self, dirpath: str = "", filename: Optional[str]=None, period: int=1):
        super().__init__()
        self.period = period
        self.dirpath = dirpath
        self.filename = filename
        if self.filename is None:
            self.filename = "model_{0}.onnx"

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        if trainer.current_epoch % self.period == 0:
            filepath = self.dirpath + "/" + self.filename.format(trainer.current_epoch)
            pl_module.export(filepath)