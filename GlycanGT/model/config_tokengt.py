# config_tokengt.py
"""Model hyper‑parameter presets for Glycan TokenGT

Each entry returns a plain dict so that hydra / argparse / yaml can load and
update it easily.
"""
from __future__ import annotations

from copy import deepcopy
from typing import Dict

# ---------------------------------------------------------------------------
# Baseline sizes -------------------------------------------------------------
# ---------------------------------------------------------------------------
_base_super_small: Dict = {
    # === vocabulary sizes (to be overwritten after tokenizer build) ===
    "num_atoms": 512,          # len(monomer_vocab)
    "num_edges": 128,          # len(linkage_vocab)

    # === node‐id settings ===
    "rand_node_id": False,
    "rand_node_id_dim": 0,
    "orf_node_id": True,
    "orf_node_id_dim": 64,
    "lap_node_id": False,
    "lap_node_id_k": 0,
    "lap_node_id_sign_flip": False,
    "lap_node_id_eig_dropout": 0.0,
    "type_id": True,          # distinguish node / edge tokens

    # === architecture ===
    "num_encoder_layers": 4,   # 6 -> 4
    "embedding_dim": 128,      # 256 -> 128
    "ffn_embedding_dim": 512,  # 768 -> 512
    "num_attention_heads": 4,  # 8 -> 4

    # === dropout & regularisation ===
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "stochastic_depth": False,

    # Performer (optional) --------------------------------------------------
    "performer": False,
    "performer_nb_features": 64,
    "performer_generalized_attention": False,

    # misc ------------------------------------------------------------------
    "layernorm_style": "postnorm",  # or "prenorm"
    "activation_fn": "gelu",

    # Train‑time options (handled outside the encoder) ----------------------
    "embed_scale": None,             # √d by default inside tokenizer
    "freeze_embeddings": False,
    "n_trans_layers_to_freeze": 0,
    "return_attention": False,
}

_base_small: Dict = {
    # === vocabulary sizes (to be overwritten after tokenizer build) ===
    "num_atoms": 512,          # len(monomer_vocab)
    "num_edges": 128,          # len(linkage_vocab)

    # === node‐id settings ===
    "rand_node_id": False,
    "rand_node_id_dim": 0,
    "orf_node_id": True,
    "orf_node_id_dim": 64,
    "lap_node_id": False,
    "lap_node_id_k": 0,
    "lap_node_id_sign_flip": False,
    "lap_node_id_eig_dropout": 0.0,
    "type_id": True,          # distinguish node / edge tokens

    # === architecture ===
    "num_encoder_layers": 6,
    "embedding_dim": 256,
    "ffn_embedding_dim": 768,
    "num_attention_heads": 8,

    # === dropout & regularisation ===
    "dropout": 0.1,
    "attention_dropout": 0.1,
    "activation_dropout": 0.0,
    "stochastic_depth": False,

    # Performer (optional) --------------------------------------------------
    "performer": False,
    "performer_nb_features": 64,
    "performer_generalized_attention": False,

    # misc ------------------------------------------------------------------
    "layernorm_style": "postnorm",  # or "prenorm"
    "activation_fn": "gelu",

    # Train‑time options (handled outside the encoder) ----------------------
    "embed_scale": None,         
    "freeze_embeddings": False,
    "n_trans_layers_to_freeze": 0,
    "return_attention": False,
}

base_medium = deepcopy(_base_small)
base_medium.update(
    {
        "embedding_dim": 512,
        "ffn_embedding_dim": 2048,
        "num_encoder_layers": 8,
        "num_attention_heads": 16,
    }
)

base_large = deepcopy(_base_small)
base_large.update(
    {
        "embedding_dim": 768,
        "ffn_embedding_dim": 3072,
        "num_encoder_layers": 12,
        "num_attention_heads": 32,
    }
)

# ---------------------------------------------------------------------------
# Helper --------------------------------------------------------------------
# ---------------------------------------------------------------------------

def get_config(name: str) -> Dict:
    """Return a deep‑copied config dict so callers can mutate safely."""
    if name == "ss":
        return deepcopy(_base_super_small)
    if name == "small":
        return deepcopy(_base_small)
    if name == "medium":
        return deepcopy(base_medium)
    if name == "large":
        return deepcopy(base_large)
    raise ValueError(f"Unknown config preset: {name}")
