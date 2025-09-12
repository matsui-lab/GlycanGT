"""
Modified from https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/modules/tokengt_graph_encoder.py
"""

# tokengt_graph_encoder.py
from typing import Optional, Tuple

import torch
import torch.nn as nn

from torch.nn import LayerNorm
from multihead_attention import MultiheadAttention
from tokengt_graph_encoder_layer import TokenGTGraphEncoderLayer
from performer_pytorch import ProjectionUpdater
from tokenizer import GraphFeatureTokenizer


def init_graphormer_params(module: nn.Module):
    """
    Initialize the weights specific to the Graphormer Model.
    This function remains compatible as it targets standard nn.Module attributes.
    """
    def normal_(data: torch.Tensor):
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        if module.weight is not None:
            normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        if hasattr(module, 'q_proj') and module.q_proj.weight is not None:
            normal_(module.q_proj.weight.data)
        if hasattr(module, 'k_proj') and module.k_proj.weight is not None:
            normal_(module.k_proj.weight.data)
        if hasattr(module, 'v_proj') and module.v_proj.weight is not None:
            normal_(module.v_proj.weight.data)


class TokenGTGraphEncoder(nn.Module):
    def __init__(
        self,
        # Tokenizer args
        num_atoms: int,
        num_edges: int,
        rand_node_id: bool = False,
        rand_node_id_dim: int = 64,
        orf_node_id: bool = False,
        orf_node_id_dim: int = 64,
        lap_node_id: bool = False,
        lap_node_id_k: int = 8,
        lap_node_id_sign_flip: bool = False,
        lap_node_id_eig_dropout: float = 0.0,
        type_id: bool = False,
        # Model args
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        stochastic_depth: bool = False,
        encoder_normalize_before: bool = False,
        layernorm_style: str = "prenorm",
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        # Performer args
        performer: bool = False,
        performer_finetune: bool = False,
        performer_nb_features: int = None,
        performer_feature_redraw_interval: int = 1000,
        performer_generalized_attention: bool = False,
        performer_auto_check_redraw: bool = True,
        # Other args
        n_trans_layers_to_freeze: int = 0,
        traceable: bool = False,
        return_attention: bool = False
    ) -> None:
        super().__init__()
        self.dropout_module = nn.Dropout(p=dropout)
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable
        self.performer = performer
        self.performer_finetune = performer_finetune

        self.graph_feature = GraphFeatureTokenizer(
            num_atoms=num_atoms,
            num_edges=num_edges,
            rand_node_id=rand_node_id,
            rand_node_id_dim=rand_node_id_dim,
            orf_node_id=orf_node_id,
            orf_node_id_dim=orf_node_id_dim,
            lap_node_id=lap_node_id,
            lap_node_id_k=lap_node_id_k,
            lap_node_id_sign_flip=lap_node_id_sign_flip,
            lap_node_id_eig_dropout=lap_node_id_eig_dropout,
            type_id=type_id,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers
        )

        self.emb_layer_norm = LayerNorm(self.embedding_dim) if encoder_normalize_before else None
        self.final_layer_norm = LayerNorm(self.embedding_dim) if layernorm_style == "prenorm" else None

        self.layers = nn.ModuleList([])

        if stochastic_depth:
            assert layernorm_style == 'prenorm', "Stochastic depth is only supported with pre-layernorm style"

        self.cached_performer_options = None
        if self.performer_finetune:
            assert self.performer
            self.cached_performer_options = (
                performer_nb_features,
                performer_generalized_attention,
                performer_auto_check_redraw,
                performer_feature_redraw_interval
            )
            self.performer = False
            performer = False
            performer_nb_features = None
            performer_generalized_attention = False
            performer_auto_check_redraw = False

        self.layers.extend(
            [
                self.build_tokengt_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    drop_path=(0.1 * (layer_idx + 1) / num_encoder_layers) if stochastic_depth else 0,
                    performer=performer,
                    performer_nb_features=performer_nb_features,
                    performer_generalized_attention=performer_generalized_attention,
                    activation_fn=activation_fn,
                    layernorm_style=layernorm_style,
                    return_attention=return_attention,
                )
                for layer_idx in range(num_encoder_layers)
            ]
        )

        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        if n_trans_layers_to_freeze > 0:
            for layer in self.layers[:n_trans_layers_to_freeze]:
                for p in layer.parameters():
                    p.requires_grad = False

        self.performer_proj_updater = None
        if performer:
            self.performer_auto_check_redraw = performer_auto_check_redraw
            self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def performer_finetune_setup(self):
        assert self.performer_finetune
        assert self.cached_performer_options is not None
        (
            performer_nb_features,
            performer_generalized_attention,
            performer_auto_check_redraw,
            performer_feature_redraw_interval
        ) = self.cached_performer_options

        for layer in self.layers:
            if hasattr(layer, 'performer_finetune_setup'):
                layer.performer_finetune_setup(performer_nb_features, performer_generalized_attention)

        self.performer = True
        self.performer_auto_check_redraw = performer_auto_check_redraw
        self.performer_proj_updater = ProjectionUpdater(self.layers, performer_feature_redraw_interval)

    def build_tokengt_graph_encoder_layer(
        self,
        embedding_dim: int,
        ffn_embedding_dim: int,
        num_attention_heads: int,
        dropout: float,
        attention_dropout: float,
        activation_dropout: float,
        drop_path: float,
        performer: bool,
        performer_nb_features: Optional[int],
        performer_generalized_attention: bool,
        activation_fn: str,
        layernorm_style: str,
        return_attention: bool,
    ) -> TokenGTGraphEncoderLayer:
        return TokenGTGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            drop_path=drop_path,
            performer=performer,
            performer_nb_features=performer_nb_features,
            performer_generalized_attention=performer_generalized_attention,
            activation_fn=activation_fn,
            layernorm_style=layernorm_style,
            return_attention=return_attention
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
    ) -> Tuple[list, torch.Tensor, dict]:
        if self.performer and self.performer_proj_updater is not None:
            if self.performer_auto_check_redraw:
                self.performer_proj_updater.redraw_projections()

        # 1. Tokenize graph features
        x, padding_mask, padded_index = self.graph_feature(batched_data, perturb)

        # 2. Apply embedding layer norm and dropout
        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)
        x = self.dropout_module(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn_dict = {'maps': {}, 'padded_index': padded_index}
        
        # 3. Forward through encoder layers
        for i, layer in enumerate(self.layers):
            # LayerDrop logic: stochastically drop a layer during training
            if self.training and self.layerdrop > 0.0 and torch.rand(1).item() < self.layerdrop:
                attn_dict['maps'][i] = None # Record that the layer was dropped
                continue

            x, attn = layer(x, self_attn_padding_mask=padding_mask)
            
            if not last_state_only:
                inner_states.append(x)
            attn_dict['maps'][i] = attn

        # The first token ([CLS] or equivalent) is used as the graph representation
        graph_rep = x[0, :, :]

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep, attn_dict
        
        return inner_states, graph_rep, attn_dict