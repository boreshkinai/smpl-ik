import torch as t
import torch.nn.functional as f
from typing import Tuple


class FCBlock(t.nn.Module):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int):
        super(FCBlock, self).__init__()
        self.num_layers = num_layers
        self.layer_width = layer_width

        self.fc_layers = [t.nn.Linear(size_in, layer_width)]
        self.relu_layers = [t.nn.LeakyReLU(inplace=True)]
        if dropout > 0.0:
            self.fc_layers.append(t.nn.Dropout(p=dropout))
            self.relu_layers.append(t.nn.Identity())
        self.fc_layers += [t.nn.Linear(layer_width, layer_width) for _ in range(num_layers - 1)]
        self.relu_layers += [t.nn.LeakyReLU(inplace=True) for _ in range(num_layers - 1)]

        self.forward_projection = t.nn.Linear(layer_width, size_out)
        self.backward_projection = t.nn.Linear(size_in, layer_width)
        self.fc_layers = t.nn.ModuleList(self.fc_layers)
        self.relu_layers = t.nn.ModuleList(self.relu_layers)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        h = x
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = t.relu(h + self.backward_projection(x))
        return b, f


class LayerNorm(t.nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, num_features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = t.nn.Parameter(t.ones(num_features))
        self.b_2 = t.nn.Parameter(t.zeros(num_features))
        self.eps = eps

    def forward(self, x: t.Tensor) -> t.Tensor:
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class FCBlockNorm(FCBlock):
    """Fully connected residual block"""

    def __init__(self, num_layers: int, layer_width: int, dropout: float, size_in: int, size_out: int, eps: float = 1e-6):
        super(FCBlockNorm, self).__init__(num_layers=num_layers, layer_width=layer_width,
                                          dropout=dropout, size_in=size_in, size_out=size_out)
        self.norm = LayerNorm(num_features=size_in, eps=eps)

    def forward(self, x: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        h = self.norm(x)
        for layer, relu in zip(self.fc_layers, self.relu_layers):
            h = relu(layer(h))
        f = self.forward_projection(h)
        b = t.relu(h + self.backward_projection(x))
        return b, f


class Embedding(t.nn.Module):
    """Implementation of embedding using one hot encoded input and fully connected layer"""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super(Embedding, self).__init__()
        self.projection = t.nn.Linear(num_embeddings, embedding_dim)
        self.num_embeddings = num_embeddings

    def forward(self, e: t.Tensor) -> t.Tensor:
        e_ohe = f.one_hot(e, num_classes=self.num_embeddings).float()
        return self.projection(e_ohe)
