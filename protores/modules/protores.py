from typing import Union, List
import omegaconf
import torch

from protores.modules.custom_layers import FCBlock, FCBlockNorm


class ProtoEncoder(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, num_blocks: int, num_layers: int, layer_width: int, dropout: float):
        super(ProtoEncoder, self).__init__()

        self.blocks = [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout, size_in=size_in,
                               size_out=size_out)]
        self.blocks += [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout, size_in=size_out,
                                size_out=size_out) for _ in range(num_blocks - 1)]
        self.blocks = torch.nn.ModuleList(self.blocks)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backcast = x
        encoding = 0.0
        for i, block in enumerate(self.blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            if i < len(self.blocks) - 1:
                prototype = encoding.mean(dim=-2, keepdim=True)
                backcast = backcast - prototype / (i + 1.0)
                backcast = torch.relu(backcast)

        embedding = encoding.mean(dim=-2, keepdim=False)
        return embedding


class WeightedProtoEncoder(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, num_blocks: int, num_layers: int, layer_width: int, dropout: float,
                 eps: float = 1e-6):
        super(WeightedProtoEncoder, self).__init__()

        self.eps = eps
        self.blocks = [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout, size_in=size_in,
                               size_out=size_out)]
        self.blocks += [FCBlock(num_layers=num_layers, layer_width=layer_width, dropout=dropout, size_in=size_out,
                                size_out=size_out) for _ in range(num_blocks - 1)]
        self.blocks = torch.nn.ModuleList(self.blocks)

    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weights = weights.unsqueeze(2)
        weights_sum = torch.clamp(weights.sum(dim=1, keepdim=True), min=self.eps)

        backcast = x
        encoding = 0.0
        for i, block in enumerate(self.blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            if i < len(self.blocks) - 1:
                prototype = (encoding * weights).sum(dim=-2, keepdim=True) / weights_sum
                backcast = backcast - prototype / (i + 1.0)
                backcast = torch.relu(backcast)

        embedding = (encoding * weights).sum(dim=-2, keepdim=True) / weights_sum
        embedding = embedding.squeeze(-2)

        return embedding


class ProtoDecoder(torch.nn.Module):
    def __init__(self, size_in: int, size_out: int, num_blocks: int, num_layers: int, layer_width: Union[int, List[int]], dropout: float,
                 layer_norm: bool = False, eps: float = 1e-6):
        super(ProtoDecoder, self).__init__()

        if isinstance(layer_width, omegaconf.listconfig.ListConfig) or isinstance(layer_width, list):
            assert len(layer_width) == num_blocks, "The list of layer widths provided must be equal to the number of blocks"
            layer_widths = layer_width
        else:
            layer_widths = [layer_width for _ in range(num_blocks)]

        if layer_norm:
            self.blocks = [FCBlockNorm(num_layers=num_layers, layer_width=layer_widths[0], dropout=dropout, size_in=size_in,
                                       size_out=size_out, eps=eps)]
            self.blocks += [FCBlockNorm(num_layers=num_layers, layer_width=layer_widths[i], dropout=dropout, size_in=layer_widths[i-1],
                                        size_out=size_out, eps=eps) for i in range(1, num_blocks)]
        else:
            self.blocks = [FCBlock(num_layers=num_layers, layer_width=layer_widths[0], dropout=dropout, size_in=size_in,
                                   size_out=size_out)]
            self.blocks += [FCBlock(num_layers=num_layers, layer_width=layer_width[i], dropout=dropout, size_in=layer_width[i-1],
                                    size_out=size_out) for i in range(1, num_blocks)]
        self.blocks = torch.nn.ModuleList(self.blocks)
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backcast = x
        forecast = 0.0
        for i, block in enumerate(self.blocks):
            backcast, f = block(backcast)
            forecast = forecast + f
        return forecast
