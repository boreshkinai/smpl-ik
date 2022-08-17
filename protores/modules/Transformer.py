import torch as t
from protores.modules.custom_layers import Embedding, FCBlockNorm


class Transformer2Stage(t.nn.Module):
    """
    Transformer architechture wrapper
    """
    def __init__(self, num_joints, d_model: int, nhead:int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, activation: str,
                 size_in: int, size_out: int, size_out_stage1: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int,
                 fcr_num_layers: int, fcr_num_blocks: int, fcr_width: int):

        super(Transformer2Stage, self).__init__()

        self.size_in = size_in
        self.size_out = size_out // num_joints
        self.num_joints = num_joints
        self.size_out_stage1 = size_out_stage1 // num_joints

        self.register_buffer('joint_ids', t.arange(0, num_joints, dtype=t.int64))

        self.src_projection = t.nn.Linear(size_in+embedding_dim*embedding_num, d_model)
        self.tgt_projection = t.nn.Linear(embedding_dim, d_model)
        self.transformer = t.nn.Transformer(d_model=d_model, nhead=nhead,
                                            num_encoder_layers=num_encoder_layers,
                                            num_decoder_layers=num_decoder_layers,
                                            dim_feedforward=dim_feedforward,
                                            dropout=dropout, activation=activation,
                                            )
        self.stage1_projection = [FCBlockNorm(num_layers=fcr_num_layers, layer_width=fcr_width,  dropout=dropout,
                                  size_in=d_model, size_out=self.size_out+self.size_out_stage1)] +\
                                 [FCBlockNorm(num_layers=fcr_num_layers, layer_width=fcr_width, dropout=dropout,
                                  size_in=fcr_width, size_out=self.size_out+self.size_out_stage1) for _ in range(fcr_num_blocks - 1)]

        self.embeddings_input = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) \
                                    for _ in range(embedding_num)]

        self.stage1_projection = t.nn.ModuleList(self.stage1_projection)
        self.embeddings_input = t.nn.ModuleList(self.embeddings_input)

    def forward(self, x: t.Tensor, *args) -> t.Tensor:
        """
        It is assumed here that the node ids is position 0 in args
        """
        ee = [x]
        for i, v in enumerate(args):
            ee.append(self.embeddings_input[i](v))
        src = t.cat(ee, dim=-1)
        src = self.src_projection(src)

        joint_ids = self.joint_ids.repeat(x.shape[0], 1)
        joint_embeddings = self.embeddings_input[0](joint_ids)
        tgt = self.tgt_projection(joint_embeddings)  # this becomes B x J x d_model

        # We need to align the batch first for the input with the batch second with transformer
        out = self.transformer(src=t.transpose(src, 0, 1), # E x B x d_model
                               tgt=t.transpose(tgt, 0, 1)) # J x B x d_model

        stage_1 = 0.0
        backcast = t.transpose(out, 0, 1)
        for block in self.stage1_projection:
            backcast, f = block(backcast)
            stage_1 = stage_1 + f

        output = stage_1.view(-1, self.num_joints*(self.size_out_stage1 + self.size_out))
        stage_1 = output[..., :self.num_joints*self.size_out_stage1].contiguous()
        stage_2 = output[..., self.num_joints*self.size_out_stage1:].contiguous()
        return stage_1, stage_2


class WeightedTransformer(Transformer2Stage):
    def __init__(self, num_joints, d_model: int, nhead:int,
                 num_encoder_layers: int, num_decoder_layers: int, dim_feedforward: int,
                 dropout: float, activation: str,
                 size_in: int, size_out: int, size_out_stage1: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int,
                 fcr_num_layers: int, fcr_num_blocks: int, fcr_width: int):
        super().__init__(num_joints, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward,
                         dropout, activation, size_in, size_out, size_out_stage1, embedding_dim, embedding_size,
                         embedding_num, fcr_num_layers, fcr_num_blocks, fcr_width)

    def forward(self, effectors_in, effector_weights, effector_ids, effector_types):
        return super().forward(effectors_in, effector_ids, effector_types)  # ignore blending weights
