"""
Modified from https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/modules/tokengt_graph_encoder_layer.py
"""
# tokengt_graph_encoder_layer.py
from typing import Callable, Optional, Tuple
import torch
import torch.nn as nn
from torch.nn import Dropout, LayerNorm
from feedforward import FeedForward
from multihead_attention import MultiheadAttention
from multihead_performer_attention import MultiheadPerformerAttention
from droppath import DropPath


class TokenGTGraphEncoderLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        drop_path: float = 0.0,
        performer: bool = False,
        performer_nb_features: Optional[int] = None,
        performer_generalized_attention: bool = False,
        activation_fn: str = "relu",
        layernorm_style: str = "prenorm",
        return_attention: bool = False,
    ) -> None:
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_attention_heads = num_attention_heads
        self.layernorm_style = layernorm_style
        self.return_attention = return_attention

        self.dropout_module = Dropout(p=dropout)

        self.self_attn = self.build_self_attention(
            embed_dim=self.embedding_dim,
            num_attention_heads=num_attention_heads,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            attention_dropout=attention_dropout,
            dropout=dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.feedforward = self.build_FFN(
            embedding_dim=self.embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
        )
        self.final_layer_norm = LayerNorm(self.embedding_dim)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def build_FFN(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        activation_fn: str,
        activation_dropout: float,
        dropout: float,
    ) -> FeedForward:
        return FeedForward(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            activation_fn=activation_fn,
            activation_dropout=activation_dropout,
            dropout=dropout,
        )

    def build_self_attention(
        self,
        embed_dim: int,
        num_attention_heads: int,
        performer: bool,
        performer_nb_features: Optional[int],
        performer_generalized_attention: bool,
        attention_dropout: float,
        dropout: float,
    ) -> nn.Module:
        if performer:
            return MultiheadPerformerAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                performer_nb_features=performer_nb_features,
                performer_generalized_attention=performer_generalized_attention,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )
        else:
            return MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_attention_heads,
                attention_dropout=attention_dropout,
                dropout=dropout,
                self_attention=True,
            )

    def performer_finetune_setup(self, performer_nb_features: int, performer_generalized_attention: bool):
        if hasattr(self.self_attn, 'performer_finetune_setup'):
            self.self_attn.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer implementation.
        """
        # x: T x B x C
        attn = None
        
        if self.layernorm_style == "prenorm":
            # 1. Pre-LN Self-Attention
            residual = x
            x_norm = self.self_attn_layer_norm(x)
            x, attn = self.self_attn(
                query=x_norm,
                key=x_norm,
                value=x_norm,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                attn_mask=self_attn_mask,
                attn_bias=self_attn_bias,
            )
            x = self.dropout_module(x)
            x = self.drop_path1(x)
            x = residual + x

            # 2. Pre-LN Feed-Forward
            residual = x
            x_norm = self.final_layer_norm(x)
            x = self.feedforward(x_norm)
            # x = self.dropout_module(x) 
            x = self.drop_path2(x)
            x = residual + x

        elif self.layernorm_style == "postnorm":
            # 1. Post-LN Self-Attention
            residual = x
            x, attn = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=self.return_attention,
                attn_mask=self_attn_mask,
                attn_bias=self_attn_bias,
            )
            x = self.dropout_module(x)
            x = residual + x
            x = self.self_attn_layer_norm(x)

            # 2. Post-LN Feed-Forward
            residual = x
            x = self.feedforward(x)
            x = residual + x
            x = self.final_layer_norm(x)

        else:
            raise NotImplementedError(f"layernorm_style '{self.layernorm_style}' is not supported.")
            
        return x, attn