"""
Modified from https://github.com/jw9730/tokengt/blob/main/large-scale-regression/tokengt/modules/multihead_attention.py
"""
# multihead_attention.py
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from einops import rearrange

class FastAttention(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        print("Warning: Using a dummy implementation of FastAttention.")
    def forward(self, q, k, v, key_padding_mask=None):

        attn_weights = torch.einsum('b h n d, b h m d -> b h n m', q, k)
        if key_padding_mask is not None:
             attn_weights.masked_fill_(key_padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_weights, dim=-1)
        output = torch.einsum('b h n m, b h m d -> b h n d', attn_weights, v)
        return output

class MultiheadAttention(nn.Module):
    """
    Multi-headed attention, with an option to switch to Performer's
    FastAttention.
    """
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        kdim: Optional[int] = None,
        vdim: Optional[int] = None,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        bias: bool = True,
        self_attention: bool = False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.num_heads = num_heads
        self.self_attention = self_attention
        self.qkv_same_dim = self.kdim == embed_dim and self.vdim == embed_dim
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5

        assert self.self_attention, "This implementation only supports self-attention."
        assert not self.self_attention or self.qkv_same_dim, "Self-attention requires Q, K, V to be of the same dimension."
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads."

        self.attention_dropout_module = nn.Dropout(p=attention_dropout)
        self.dropout_module = nn.Dropout(p=dropout)

        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.fast_attention: Optional[nn.Module] = None
        self.reset_parameters()

    def performer_finetune_setup(self, performer_nb_features: int, performer_generalized_attention: bool):
        """
        Sets up the module to use Performer's FastAttention and switches the
        forward pass to the Performer implementation.
        """
        self.fast_attention = FastAttention(
            self.head_dim,
            performer_nb_features,
            causal=False,
            generalized_attention=performer_generalized_attention,
            kernel_fn=nn.ReLU(),
            no_projection=False
        )
        self.forward = self.forward_performer

    def reset_parameters(self):
        """Initializes the weights of the linear layers."""
        if self.qkv_same_dim:
            # Scaled initialization for better convergence
            nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
            nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        else:
            nn.init.xavier_uniform_(self.k_proj.weight)
            nn.init.xavier_uniform_(self.v_proj.weight)
            nn.init.xavier_uniform_(self.q_proj.weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Standard dot-product attention forward pass.

        Input shape: Time x Batch x Channel
        """
        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        assert embed_dim == self.embed_dim, f"query's embed_dim ({embed_dim}) does not match module's embed_dim ({self.embed_dim})"
        # self-attention is asserted in __init__, so we use query for k,v
        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q *= self.scaling

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))

        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

        if attn_bias is not None:
            attn_weights += attn_bias.view(bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights += attn_mask.unsqueeze(0)

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2).to(torch.bool),
                float("-inf"),
            )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights_float = F.softmax(attn_weights, dim=-1)
        attn_probs = self.attention_dropout_module(attn_weights_float)

        attn = torch.bmm(attn_probs, v)
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        avg_attn_weights: Optional[Tensor] = None
        if need_weights:
            avg_attn_weights = attn_weights_float.view(
                bsz, self.num_heads, tgt_len, src_len
            ).mean(dim=1)

        return attn, avg_attn_weights

    def forward_performer(
        self,
        query: Tensor,
        key: Optional[Tensor],
        value: Optional[Tensor],
        attn_bias: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        need_weights: bool = True,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Performer (FastAttention) forward pass.

        Input shape: Time x Batch x Channel
        """
        if attn_bias is not None:
            raise NotImplementedError("attn_bias is not supported in Performer mode.")
        if self.fast_attention is None:
            raise RuntimeError("FastAttention is not initialized. Call `performer_finetune_setup` first.")

        tgt_len, bsz, embed_dim = query.size()
        src_len = tgt_len

        q = self.q_proj(query)
        k = self.k_proj(query)
        v = self.v_proj(query)

        q, k, v = map(lambda t: rearrange(t, 'n b (h d) -> b h n d', h=self.num_heads), (q, k, v))
        
        mask = None
        if key_padding_mask is not None:
            mask = key_padding_mask.to(torch.bool)[:, None, :, None]

        attn = self.fast_attention(q, k, v, key_padding_mask=mask)
        attn = rearrange(attn, 'b h n d -> n b (h d)')

        attn = self.out_proj(attn)
        attn = self.dropout_module(attn)

        if need_weights:
            pass

        return attn, None

    def upgrade_state_dict_named(self, state_dict, name):
        """
        A utility to convert a state dict from a fairseq model that used
        a single in_proj_weight to separate q, k, v projections.
        """
        prefix = name + "." if name != "" else ""
        items_to_add = {}
        keys_to_remove = []
        for k in state_dict.keys():
            if k.endswith(prefix + "in_proj_weight"):
                dim = int(state_dict[k].shape[0] / 3)
                items_to_add[prefix + "q_proj.weight"] = state_dict[k][:dim]
                items_to_add[prefix + "k_proj.weight"] = state_dict[k][dim: 2 * dim]
                items_to_add[prefix + "v_proj.weight"] = state_dict[k][2 * dim:]
                keys_to_remove.append(k)

                k_bias = prefix + "in_proj_bias"
                if k_bias in state_dict.keys():
                    items_to_add[prefix + "q_proj.bias"] = state_dict[k_bias][:dim]
                    items_to_add[prefix + "k_proj.bias"] = state_dict[k_bias][dim: 2 * dim]
                    items_to_add[prefix + "v_proj.bias"] = state_dict[k_bias][2 * dim:]
                    keys_to_remove.append(k_bias)

        for k in keys_to_remove:
            del state_dict[k]
        for key, value in items_to_add.items():
            state_dict[key] = value
            
# [T,B,C] --(Linear)--> q,k,v: [T,B,C]
#        --(split heads)--> [B*H, T, D_h]
#        --(q·k^T)--> [B*H, T, T]
#        --(+bias/masks + softmax + dropout)--> [B*H, T, T]
#        --(×v)--> [B*H, T, D_h]
#        --(merge heads)--> [T, B, C]
#        --(out_proj + dropout)--> [T, B, C]