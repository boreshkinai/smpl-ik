import torch as t
from protores.modules.custom_layers import Embedding, FCBlock
from protores.modules.custom_layers import FCBlockNorm
from typing import Tuple


def maxpool(x, dim=-1, keepdim=False):
    ''' Performs a maxpooling operation.
    Args:
        x (tensor): input
        dim (int): dimension of pooling
        keepdim (bool): whether to keep dimensions
    '''
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class PointResNet(t.nn.Module):
    """
    Fully-connected residual architechture with many categorical inputs wrapped in embeddings
    """

    def __init__(self,
                 num_blocks_enc: int, num_layers_enc: int, layer_width_enc,
                 num_blocks_stage1: int, num_layers_stage1: int, layer_width_stage1: int,
                 num_blocks_stage2: int, num_layers_stage2: int, layer_width_stage2: int,
                 dropout: float,
                 size_in: int, size_out: int, size_out_stage1: int,
                 embedding_dim: int, embedding_size: int, embedding_num: int, layer_norm: bool = True, eps: float = 1e-6):
        super().__init__()
        
        self.pool = maxpool

        self.layer_width_enc = layer_width_enc
        self.num_blocks_stage1 = num_blocks_stage1
        self.num_blocks_stage2 = num_blocks_stage2

        self.embeddings = [Embedding(num_embeddings=embedding_size, embedding_dim=embedding_dim) for _ in
                           range(embedding_num)]

        # IMPORTANT NOTE: We don't use LayerNorm in these blocks as this would inject inputs without taking into account their weights
        self.encoder_blocks = [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                       size_in=size_in + embedding_dim * embedding_num, size_out=layer_width_enc)]
        self.encoder_blocks += [FCBlock(num_layers=num_layers_enc, layer_width=layer_width_enc, dropout=dropout,
                                        size_in=layer_width_enc * 2, size_out=layer_width_enc) for _ in range(num_blocks_enc - 1)]
        
        if layer_norm:
            fc_block_primitive = FCBlockNorm
        else:
            fc_block_primitive = FCBlock
            
        if num_blocks_stage1 == 0:
            self.stage1_blocks = [t.nn.Linear(layer_width_enc, size_out_stage1)]
        else:
            self.stage1_blocks = [fc_block_primitive(num_layers=num_layers_stage1, layer_width=layer_width_stage1, 
                                                     dropout=dropout, size_in=layer_width_enc, size_out=size_out_stage1)] + \
                                 [fc_block_primitive(num_layers=num_layers_stage1, layer_width=layer_width_stage1, 
                                                     dropout=dropout, size_in=layer_width_stage1, 
                                                     size_out=size_out_stage1) for _ in range(num_blocks_stage1 - 1)]
            
        if num_blocks_stage2 == 0:
            self.stage2_blocks = [t.nn.Linear(size_out_stage1 + layer_width_enc, size_out)]
        else:
            self.stage2_blocks = [fc_block_primitive(num_layers=num_layers_stage2, layer_width=layer_width_stage2, dropout=dropout,
                                                size_in=size_out_stage1 + layer_width_enc, size_out=size_out)] + \
                                 [fc_block_primitive(num_layers=num_layers_stage2, layer_width=layer_width_stage2, 
                                                     dropout=dropout, size_in=layer_width_stage2, 
                                                     size_out=size_out) for _ in range(num_blocks_stage2 - 1)]

        self.model = t.nn.ModuleList(
            self.encoder_blocks + self.stage1_blocks + self.stage2_blocks + self.embeddings)

    def encode(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
                x the continuous input : BxNxF
                weights the weight of each input BxN
                e the categorical inputs BxNxC
                """
        
        batch_size, N, F = x.shape

        weights = weights.unsqueeze(2)
        weights_sum = weights.sum(dim=1, keepdim=True)

        ee = [x]
        for i, v in enumerate(args):
            ee.append(self.embeddings[i](v))
        backcast = t.cat(ee, dim=-1)

        encoding = 0.0
        for i, block in enumerate(self.encoder_blocks):
            backcast, e = block(backcast)
            pooled = self.pool(backcast, dim=1, keepdim=True).expand(-1, N, -1)
            backcast = t.cat([backcast, pooled], dim=2)
            
        pose_embedding = self.pool(pooled, dim=1)
        return pose_embedding
    
    def decode(self, pose_embedding: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        backcast = pose_embedding
        stage1_forecast = 0.0
        for block in self.stage1_blocks:
            if self.num_blocks_stage1 > 0:
                backcast, f = block(backcast)
                stage1_forecast = stage1_forecast + f
            else:
                stage1_forecast = block(backcast)

        stage1_forecast_no_hips = stage1_forecast - stage1_forecast[:, 0:3].repeat(1, stage1_forecast.shape[1] // 3)
        backcast = t.cat([stage1_forecast_no_hips, pose_embedding], dim=-1)
        stage2_forecast = 0.0
        for block in self.stage2_blocks:
            if self.num_blocks_stage2 > 0:
                backcast, f = block(backcast)
                stage2_forecast = stage2_forecast + f
            else:
                stage2_forecast = block(backcast)
                
        return stage1_forecast, stage2_forecast

    def forward(self, x: t.Tensor, weights: t.Tensor, *args) -> Tuple[t.Tensor, t.Tensor]:
        """
        x the continuous input : BxNxF
        weights the weight of each input BxN
        e the categorical inputs BxNxC
        """

        pose_embedding = self.encode(x, weights, *args)
        return self.decode(pose_embedding)
