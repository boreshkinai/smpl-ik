import torch as t
from protores.modules.custom_layers import Embedding, FCBlock
from protores.modules.custom_layers import FCBlockNorm
from typing import Tuple


class WeightedProtoResSingleStage(t.nn.Module):
    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage: int, num_layers_stage: int, layer_width_stage: int,
                 dropout: float, size_in: int, size_out: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6):
        super().__init__()

        self.layer_width_enc = layer_width_enc

        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]

        # IMPORTANT NOTE: We don't use LayerNorm in these blocks as this would inject inputs without taking into account their weights
        self.encoder_blocks = [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                             size_in=size_in + embedding_dim * embedding_num, size_out=layer_width_enc)]
        self.encoder_blocks += [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                              size_in=layer_width_enc, size_out=layer_width_enc) for _ in range(num_blocks_enc - 1)]

        if layer_norm:
            self.stage_blocks = [FCBlockNorm(num_layers=num_layers_stage, layer_width=layer_width_stage, dropout=dropout,
                                                size_in=layer_width_enc, size_out=size_out, eps=eps)] + \
                                 [FCBlockNorm(num_layers=num_layers_stage, layer_width=layer_width_stage, dropout=dropout,
                                                size_in=layer_width_stage, size_out=size_out, eps=eps) for _ in range(num_blocks_stage - 1)]
        else:
            self.stage_blocks = [FCBlock(num_layers=num_layers_stage, layer_width=layer_width_stage,
                                              dropout=dropout, size_in=layer_width_enc, size_out=size_out)] + \
                                 [FCBlock(num_layers=num_layers_stage, layer_width=layer_width_stage,
                                              dropout=dropout, size_in=layer_width_stage, size_out=size_out) for _ in range(num_blocks_stage - 1)]

        self.model = t.nn.ModuleList(
            self.encoder_blocks + self.stage_blocks + self.embeddings)

    def encode(self, x: t.Tensor, weights: t.Tensor, *args) -> t.Tensor:
        """
                x the continuous input : BxNxF
                weights the weight of each input BxN
                e the categorical inputs BxNxC
                """

        weights = weights.unsqueeze(2)
        weights_sum = weights.sum(dim=1, keepdim=True)

        ee = [x]
        for i, v in enumerate(args):
            ee.append(self.embeddings[i](v))
        backcast = t.cat(ee, dim=-1)

        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e

            # weighted average
            prototype = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum

            backcast = backcast - prototype / (i + 1.0)
            backcast = t.relu(backcast)

        pose_embedding = (encoding * weights).sum(dim=1, keepdim=True) / weights_sum
        pose_embedding = pose_embedding.squeeze(1)
        return pose_embedding

    def decode(self, pose_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = pose_embedding
        stage_forecast = 0.0
        for block in self.stage_blocks:
            backcast, f = block(backcast)
            stage_forecast = stage_forecast + f

        return stage_forecast

    def forward(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, *args)
        return self.decode(pose_embedding)