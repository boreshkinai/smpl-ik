import torch as t
from protores.modules.WeightedProtoRes import WeightedProtoRes
from typing import Tuple


class MaskedFcr(WeightedProtoRes):

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage1: int, num_layers_stage1: int, layer_width_stage1: int,
                 num_blocks_stage2: int, num_layers_stage2: int, layer_width_stage2: int,
                 dropout: float,
                 size_in: int, size_out: int, size_out_stage1: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6):

        super().__init__(num_blocks_enc=num_blocks_enc, num_layers_enc=num_layers_enc, layer_width_enc=layer_width_enc,
                         num_blocks_stage1=num_blocks_stage1, num_layers_stage1=num_layers_stage1, layer_width_stage1=layer_width_stage1,
                         num_blocks_stage2=num_blocks_stage2, num_layers_stage2=num_layers_stage2, layer_width_stage2=layer_width_stage2,
                         dropout=dropout, size_in=size_in, size_out=size_out, size_out_stage1=size_out_stage1,
                         embedding_dim=embedding_dim, embedding_size=embedding_size, embedding_num=0,
                         layer_norm=layer_norm, eps=eps)

        self.default_effector_inputs_pos = t.nn.Parameter(t.rand((64, 7)), requires_grad=True)
        self.default_effector_inputs_rot = t.nn.Parameter(t.rand((64, 7)), requires_grad=True)
        self.default_effector_inputs_lookat = t.nn.Parameter(t.rand((64, 7)), requires_grad=True)

    def encode(self, x: t.Tensor, weights: t.Tensor, x_ids: t.Tensor, x_types: t.Tensor) -> t.Tensor:
        """
        x the continuous input : BxNxF
        weights the weight of each input : BxN
        x_ids the id of each x :  BxN
        """
        batch_size = x.shape[0]

        # Positions
        positions_indices = t.nonzero((x_types[0] == 0)).squeeze(1)
        backcast_pos = self.default_effector_inputs_pos.unsqueeze(0).repeat(batch_size, 1, 1)
        pos_ids = x_ids[:, positions_indices]
        pos_x = x[:, positions_indices, :]
        backcast_pos[t.arange(backcast_pos.shape[0]).unsqueeze(1), pos_ids] = pos_x
        backcast_pos = backcast_pos.view(batch_size, -1)

        # Rotations
        rotations_indices = t.nonzero((x_types[0] - 1) == 0).squeeze(1)
        backcast_rot = self.default_effector_inputs_rot.unsqueeze(0).repeat(batch_size, 1, 1)
        if rotations_indices.shape[0] > 0:  # We might have no rotation input
            rot_ids = x_ids[:, rotations_indices]
            rot_x = x[:, rotations_indices, :]
            backcast_rot[t.arange(backcast_rot.shape[0]).unsqueeze(1), rot_ids] = rot_x
        backcast_rot = backcast_rot.view(batch_size, -1)

        # Look At
        lookat_indices = t.nonzero((x_types[0] - 2) == 0).squeeze(1)
        backcast_lookat = self.default_effector_inputs_lookat.unsqueeze(0).repeat(batch_size, 1, 1)
        if lookat_indices.shape[0] > 0:  # We might have no rotation input
            lookat_ids = x_ids[:, lookat_indices]
            lookat_x = x[:, lookat_indices, :]
            backcast_lookat[t.arange(backcast_lookat.shape[0]).unsqueeze(1), lookat_ids] = lookat_x
        backcast_lookat = backcast_lookat.view(batch_size, -1)

        # Combine
        backcast = t.cat([backcast_pos, backcast_rot, backcast_lookat], dim=1)

        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            encoding = encoding + e
        return encoding

    def forward(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, args[0], args[1])
        return self.decode(pose_embedding)

