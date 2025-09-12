# glycan_tokengt_multitask.py
"""
Wrapper model that combines TokenGT encoder with a LM head for NTP / SMTP.
"""
from __future__ import annotations
from typing import Dict, Literal, Tuple, Optional
import torch
import torch.nn as nn
from torch import Tensor
from config_tokengt import get_config
from tokengt_graph_encoder import TokenGTGraphEncoder  

TaskType = Literal["smtp", "ntp"]


class GlycanTokenGTMultiTask(nn.Module):
    """TokenGT specialised for glycans.

    * `forward(..., task)` returns `(logits, extra)` where:
        - logits: `[T, B, V_atom]`  (V_atom = |monomer vocab|)
        - extra : dict, may include graph_rep for downstream tasks.
    """

    def __init__(self, cfg: Dict | str):
        super().__init__()

        if isinstance(cfg, str): 
            cfg = get_config(cfg)
        self.cfg = cfg

        self.encoder = TokenGTGraphEncoder(
            num_atoms=cfg["num_atoms"],
            num_edges=cfg["num_edges"],
            rand_node_id=cfg["rand_node_id"],
            rand_node_id_dim=cfg["rand_node_id_dim"],
            orf_node_id=cfg["orf_node_id"],
            orf_node_id_dim=cfg["orf_node_id_dim"],
            lap_node_id=cfg["lap_node_id"],
            lap_node_id_k=cfg["lap_node_id_k"],
            lap_node_id_sign_flip=cfg["lap_node_id_sign_flip"],
            lap_node_id_eig_dropout=cfg["lap_node_id_eig_dropout"],
            type_id=cfg["type_id"],
            # Model args
            num_encoder_layers=cfg["num_encoder_layers"],
            embedding_dim=cfg["embedding_dim"],
            ffn_embedding_dim=cfg["ffn_embedding_dim"],
            num_attention_heads=cfg["num_attention_heads"],
            dropout=cfg["dropout"],
            attention_dropout=cfg["attention_dropout"],
            activation_dropout=cfg["activation_dropout"],
            layerdrop=cfg.get("layerdrop", 0.0),
            stochastic_depth=cfg.get("stochastic_depth", False),
            encoder_normalize_before=cfg.get("encoder_normalize_before", False),
            layernorm_style=cfg["layernorm_style"],
            apply_graphormer_init=cfg.get("apply_graphormer_init", True),
            activation_fn=cfg["activation_fn"],
            # Performer args
            performer=cfg.get("performer", False),
            performer_finetune=cfg.get("performer_finetune", False),
            performer_nb_features=cfg.get("performer_nb_features"),
            performer_generalized_attention=cfg.get("performer_generalized_attention", False),
            performer_auto_check_redraw=cfg.get("performer_auto_check_redraw", True),
            # Other args
            n_trans_layers_to_freeze=cfg.get("n_trans_layers_to_freeze", 0),
            traceable=cfg.get("traceable", False),
            return_attention=cfg.get("return_attention", False),
            )
        
        # head for node prediction
        self.node_vocab_size = cfg["num_atoms"]
        self.node_lm_head = nn.Linear(cfg["embedding_dim"], self.node_vocab_size)

        # head for edge prediction
        self.edge_vocab_size = cfg["num_edges"]
        self.edge_lm_head = nn.Linear(cfg["embedding_dim"], self.edge_vocab_size)


    def _shift_targets_ntp(self, token_ids: Tensor) -> Tensor:
        """shift right: predict next token; token_ids shape `[T] or [B, T]`."""
        if token_ids.dim() == 2:  # [B, T]
            return token_ids[:, 1:]
        return token_ids[1:]

    def forward(
        self,
        batch: Dict[str, Tensor],
        task: TaskType = "smtp",
    ) -> Tuple[Tensor, Dict]:
        """Return logits compatible with ntp / smtp loss.

        * SMTP expects batch keys: `mask_label` (T_mask,B)
        * NTP  expects `input_ids` to include BOS, we predict next
        """
        inner_states, graph_rep, attn_dict = self.encoder(
            batched_data=batch,
            perturb=None,
            last_state_only=True
        )

        padded_index = attn_dict['padded_index']   # [B, T, 2]
        h = inner_states[-1]                       # [2+T, B, C]
        
        B = h.size(1)
        C = h.size(2)
        h_body = h[2:, :, :]                       # [T, B, C]
        T = h_body.size(0)
        h_flat = h_body.reshape(T * B, C)         
        
        node_num_list = batch["node_num"]          # List[int], 長さB
        edge_num_list = batch["edge_num"]          # List[int], 長さB
        
        node_idx_list = []
        edge_idx_list = []
        for b, (n, e) in enumerate(zip(node_num_list, edge_num_list)):
            for k in range(n):
                node_idx_list.append(k * B + b)
            for k in range(e):
                t = n + k
                edge_idx_list.append(t * B + b)
                
        device = h_flat.device
        node_idx = torch.tensor(node_idx_list, dtype=torch.long, device=device)
        edge_idx = torch.tensor(edge_idx_list, dtype=torch.long, device=device)
        
        node_features = h_flat.index_select(0, node_idx)    # [sum_nodes, C]
        edge_features = h_flat.index_select(0, edge_idx)    # [sum_edges, C]
        
        node_logits = self.node_lm_head(node_features)      # [sum_nodes, V_atom]
        edge_logits = self.edge_lm_head(edge_features)      # [sum_edges, V_edge]
        
        logits = {"node": node_logits, "edge": edge_logits}
        extra = {"graph_rep": graph_rep} 
        
        if task == "ntp":
            pass
        
        return logits, extra