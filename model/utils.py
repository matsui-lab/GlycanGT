# utils.py

import torch
import torch.nn as nn
import json
import os
from torch.utils.data import Dataset

# --- Dataset ---
class GlycanGraphDataset(Dataset):
    """
    Class for take graph json file path lists and use is as pytorch Dataset.
    Loading and Parsing the Graph.
    """
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            with open(file_path, 'r') as f:
                graph_edges = json.load(f)

            if not graph_edges: return None
            
            nodes = {}
            for edge in graph_edges:
                nodes[edge['in_node_id']] = edge['in_node_vocab_id']
                nodes[edge['out_node_id']] = edge['out_node_vocab_id']
                
            if not nodes: return None
                
            sorted_node_ids = sorted(nodes.keys())
            node_id_to_idx = {node_id: i for i, node_id in enumerate(sorted_node_ids)}
            node_data = [nodes[node_id] for node_id in sorted_node_ids]
            
            edge_index_src, edge_index_dst, edge_data = [], [], []
            for edge in graph_edges:
                src_idx = node_id_to_idx[edge['out_node_id']]
                dst_idx = node_id_to_idx[edge['in_node_id']]
                edge_index_src.append(src_idx)
                edge_index_dst.append(dst_idx)
                edge_data.append(edge['edge_vocab_id'])
                
            return {
                'node_data': torch.LongTensor(node_data),
                'edge_data': torch.LongTensor(edge_data),
                'edge_index': torch.LongTensor([edge_index_src, edge_index_dst]),
            }
        except (json.JSONDecodeError, KeyError, TypeError):
            return None

# --- batch function ---
def collate_fn_for_tokengt(batch_list):
    """
    collate function for TokenGT model.
    filtering None and add dummy features
    """
    batch_list = [g for g in batch_list if g is not None and g['node_data'].numel() > 0]
    if not batch_list: return None

    node_num = [g['node_data'].shape[0] for g in batch_list]
    edge_num = [g['edge_data'].shape[0] for g in batch_list]
    total_nodes = sum(node_num)
    
    collated_node_data = torch.cat([g['node_data'] for g in batch_list], dim=0)
    collated_edge_data = torch.cat([g['edge_data'] for g in batch_list], dim=0)
    
    edge_index_list = [g['edge_index'] for g in batch_list]
    collated_edge_index = torch.cat(edge_index_list, dim=1)
    
    lap_eigvec = torch.randn(total_nodes, 16)
    lap_eigval = torch.randn(total_nodes, 16)
    
    return {
        "node_data": collated_node_data, "edge_data": collated_edge_data,
        "edge_index": collated_edge_index, "node_num": node_num, "edge_num": edge_num,
        "lap_eigvec": lap_eigvec, "lap_eigval": lap_eigval,
    }

# --- helper function for models ---
def vocab_size(v):
    if isinstance(v, list):
        return len(v)
    return (max(v.values()) + 1) if len(v) > 0 else 0

def load_pretrained_tokengt_multitask(size: str, vocabs: tuple, pretrained_dir: str, device):
    """
    Load a *multi-task* pretrained GlycanTokenGT model.
    expects file name: pretrained_tokengt_{size}_multitask_final.pt
    """
    import sys, os, torch
    sys.path.append("/share3/kitani/glycoGT/model")
    from glycan_tokengt_multitask import GlycanTokenGTMultiTask
    from config_tokengt import get_config

    monomer_vocab, linkage_vocab = vocabs

    model_path = os.path.join(pretrained_dir, f"pretrained_tokengt_{size}_multitask_final.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")

    cfg = get_config(size)
    cfg.update({
    'num_atoms': vocab_size(monomer_vocab),
    'num_edges': vocab_size(linkage_vocab),
    'orf_node_id': True,
    'lap_node_id': False})

    model = GlycanTokenGTMultiTask(cfg=cfg)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def load_vocabs(data_dir: str):
    """
    General function for loading vocabulary data.
    Supports vocab as list (ID -> token) or dict (token -> ID).
    Ensures both monomer and linkage vocabs contain a [MASK] token.
    """
    import json, os

    vocab_dir = os.path.join(data_dir, 'vocab_expanded')
    with open(os.path.join(vocab_dir, 'monomer.json'), 'r') as f:
        monomer_vocab = json.load(f)
    with open(os.path.join(vocab_dir, 'linkage.json'), 'r') as f:
        linkage_vocab = json.load(f)

    MASK_TOKEN = "[MASK]"

    def ensure_mask(vocab):
        """Ensure [MASK] exists in vocab (list or dict). Return vocab (mutated)."""
        if isinstance(vocab, list):
            if MASK_TOKEN not in vocab:
                vocab.append(MASK_TOKEN)
        else:  # dict: token -> id
            if MASK_TOKEN not in vocab:
                # assign next available id (handles empty dict too)
                next_id = (max(vocab.values()) + 1) if len(vocab) > 0 else 0
                vocab[MASK_TOKEN] = next_id
        return vocab

    monomer_vocab  = ensure_mask(monomer_vocab)
    linkage_vocab  = ensure_mask(linkage_vocab)

    return monomer_vocab, linkage_vocab
