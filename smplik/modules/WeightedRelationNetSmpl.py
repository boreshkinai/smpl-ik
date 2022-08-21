import torch as t
from smplik.modules.custom_layers import Embedding
from typing import Tuple
from smplik.modules.protores import ProtoDecoder, WeightedProtoEncoder


class WeightedRelationNetSmpl(t.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage1: int, num_layers_stage1: int, layer_width_stage1: int,
                 num_blocks_stage2: int, num_layers_stage2: int, layer_width_stage2: int,
                 dropout: float,
                 size_in: int, size_out: int, size_out_stage1: int, shape_size: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6):
        super().__init__()

        self.layer_width_enc = layer_width_enc

        self.embeddings = t.nn.ModuleList([Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)])

        self.encoder = WeightedProtoEncoder(size_in=size_in + embedding_dim * embedding_num + shape_size,  size_out=layer_width_enc,
                                            num_blocks=num_blocks_enc, num_layers=num_layers_enc,
                                            layer_width=layer_width_enc, dropout=dropout, eps=eps)

        self.decoder_stage1 = ProtoDecoder(size_in=layer_width_enc + shape_size, size_out=size_out_stage1,
                                           num_blocks=num_blocks_stage1, num_layers=num_layers_stage1,
                                           layer_width=layer_width_stage1, dropout=dropout, layer_norm=layer_norm,
                                           eps=eps)

        self.decoder_stage2 = ProtoDecoder(size_in=size_out_stage1 + layer_width_enc + shape_size, size_out=size_out,
                                           num_blocks=num_blocks_stage2, num_layers=num_layers_stage2,
                                           layer_width=layer_width_stage2, dropout=dropout, layer_norm=layer_norm,
                                           eps=eps)

    def encode(self, x: t.Tensor, weights: t.Tensor, shapes: t.Tensor, *args) -> t.Tensor:
        """
                x the continuous input : BxNxF
                weights the weight of each input BxN
                shapes the SMPL beta parameters of each input BxS
                e the categorical inputs BxNxC
                """
        ee = [x, shapes.unsqueeze(1).repeat(1, x.shape[1], 1)]
        for i, v in enumerate(args):
            ee.append(self.embeddings[i](v))
        backcast = t.cat(ee, dim=-1)
        pose_embedding = self.encoder(backcast, weights)
        return pose_embedding

    def decode(self, pose_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        stage1_forecast = self.decoder_stage1(pose_embedding)

        stage1_forecast_no_hips = stage1_forecast - stage1_forecast[:, 0:3].repeat(1, stage1_forecast.shape[1] // 3)

        backcast = t.cat([stage1_forecast_no_hips, pose_embedding], dim=-1)
        stage2_forecast = self.decoder_stage2(backcast)

        return stage1_forecast, stage2_forecast


    def forward(self, x: t.Tensor, weights: t.Tensor, shapes: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        shapes the SMPL beta parameters of each input BxS
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, shapes, *args)
        pose_embedding = t.cat([pose_embedding, shapes], dim=1)

        return self.decode(pose_embedding)
