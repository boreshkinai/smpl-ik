from typing import Dict

import torch
import onnx


class OnnxWrapper(torch.nn.Module):
    def __init__(self, module, dummy_input):
        super(OnnxWrapper, self).__init__()
        self.module = module

        for k in list(dummy_input.keys()):
            if isinstance(dummy_input[k], torch.Tensor):
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
            named_input[self.input_names[i]] = inputs[i].to(self.module.device) if isinstance(inputs[i], torch.Tensor) else inputs[i]

        named_output = self.module(named_input)
        outputs = [named_output[k] for k in self.output_names]
        return outputs

    def export(self, filepath, *args, **kwargs):
        torch.onnx.export(self, self.dummy_inputs, filepath, input_names=self.input_names,
                          output_names=self.output_names, *args, **kwargs)


def export_named_model_to_onnx(module: torch.nn.Module, dummy_input: torch.Tensor, filepath: str, metadata: Dict[str, str] = None, *args, **kwargs):

    wrapped_module = OnnxWrapper(module, dummy_input)
    wrapped_module.export(filepath, *args, **kwargs)

    if metadata is not None:
        add_metadata(onnx_filepath=filepath, metadata=metadata)


# Adds meta data to an existing model
# This will load and re-export the original model
# onnx_filepath: the input ONNX file
# output_filepath: the output ONNX file. If None, input file will be replaced
# args, kwards: any other argument used to save the modified model. See onnx.save_model for a list of possibilities
def add_metadata(onnx_filepath: str, metadata: Dict[str, str], output_filepath:str = None, *args, **kwargs):
    if output_filepath is None:
        output_filepath = onnx_filepath

    # load model
    model = onnx.load(onnx_filepath)

    # add meta-data
    for key in metadata:
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = metadata[key]

    # save modified model
    onnx.save(model, output_filepath, *args, **kwargs)
