# export_multitask_attention.py
# This script loads a pre-trained multi-task Glycan-TokenGT model,
# inputs a single glycan from a .json file,
# and generates a CSV file of the attention matrix.

import torch
import torch.nn as nn
import json
import os
import sys
import pandas as pd
import numpy as np
import argparse
from collections import Counter

# --- path setting ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, "model"))

# --- import common functions and classes ---
from utils import collate_fn_for_tokengt, load_vocabs
from glycan_tokengt_multitask_attn import GlycanTokenGTMultiTask
from config_tokengt import get_config

# --- constants ---
GLYCOGT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'mask_node_70')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'analysis', '5_attention', 'node_only', 'attention_visualization')


def load_graph_from_json(json_path: str):
    """
    Loads a single preprocessed glycan graph from a .json file.
    Also returns metadata needed for creating descriptive labels.
    """
    try:
        with open(json_path, 'r') as f:
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
        
        edge_info_list = []
        edge_data = []
        edge_index_src = []
        edge_index_dst = []
        for edge in graph_edges:
            src_idx = node_id_to_idx[edge['out_node_id']]
            dst_idx = node_id_to_idx[edge['in_node_id']]
            edge_index_src.append(src_idx)
            edge_index_dst.append(dst_idx)
            edge_data.append(edge['edge_vocab_id'])
            edge_info_list.append({
                "src_id": edge['out_node_id'], "dst_id": edge['in_node_id'],
                "vocab_id": edge['edge_vocab_id']
            })

        graph_data = {
            'node_data': torch.LongTensor(node_data),
            'edge_data': torch.LongTensor(edge_data),
            'edge_index': torch.LongTensor([edge_index_src, edge_index_dst])
        }
        
        return {
            "graph_data": graph_data,
            "sorted_node_ids": sorted_node_ids,
            "edge_info_list": edge_info_list
        }
    except Exception as e:
        print(f"Error loading or processing {json_path}: {e}")
        return None

def get_descriptive_tokens(batch: dict, vocabs: tuple, metadata: dict):
    """
    Reconstructs a list of unique and descriptive token strings.
    e.g., "Man(ID:5)", "a1-3(5->8)"
    """
    monomer_vocab, linkage_vocab = vocabs
    id2atom = {i: token for i, token in enumerate(monomer_vocab)}
    id2edge = {i: token for i, token in enumerate(linkage_vocab)}
    
    sorted_node_ids = metadata["sorted_node_ids"]
    edge_info_list = metadata["edge_info_list"]
    
    tokens = ['[GRAPH_TOKEN]', '[NULL_TOKEN]']
    
    for i, node_vocab_id in enumerate(batch['node_data'].cpu().numpy()):
        node_name = id2atom.get(node_vocab_id, '?')
        original_id = sorted_node_ids[i]
        tokens.append(f"{node_name}(ID:{original_id})")

    for edge_info in edge_info_list:
        edge_name = id2edge.get(edge_info['vocab_id'], '?')
        tokens.append(f"{edge_name}({edge_info['src_id']}->{edge_info['dst_id']})")

    counts = Counter()
    unique_tokens = []
    for token in tokens:
        idx = counts[token]
        unique_tokens.append(f"{token}_{idx}" if idx > 0 else token)
        counts[token] += 1
    return unique_tokens

def main(model_size: str, json_path: str, layer: int):
    """
    Main function to load model, process glycan, and save attention CSV.
    """
    print("====== GlycanGT Multi-task Attention Export to CSV Start ======")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vocabs = load_vocabs(GLYCOGT_DATA_DIR)
    
    monomer_vocab, linkage_vocab = vocabs
    if "[MASK]" not in monomer_vocab: monomer_vocab.append("[MASK]")
    if "[MASK]" not in linkage_vocab: linkage_vocab.append("[MASK]")
    vocabs = (monomer_vocab, linkage_vocab)
    
    # --- 1. Load Model for Attention Extraction ---
    try:
        print(f"\n--- Loading pre-trained '{model_size}' multi-task model ---")
        
        model_path = os.path.join(PRETRAINED_DIR, f"pretrained_tokengt_{model_size}_multitask_node70_final.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file Not Found: {model_path}")
        
        config = get_config(model_size)
        config.update({
            'num_atoms': len(vocabs[0]), 'num_edges': len(vocabs[1]),
            'orf_node_id': True, 'lap_node_id': False, 'return_attention': True
        })
        
        model = GlycanTokenGTMultiTask(cfg=config)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
        print("Model loaded successfully.")

    except FileNotFoundError as e:
        print(f"FATAL: {e}. Ensure a pre-trained model for size '{model_size}' exists.")
        return

    # --- 2. Load and Process the Input Glycan from JSON ---
    print(f"\n--- Loading glycan from: {json_path} ---")
    loaded_data = load_graph_from_json(json_path)
    if loaded_data is None: return
    
    graph_data = loaded_data['graph_data']
    metadata = {k: v for k, v in loaded_data.items() if k != 'graph_data'}
        
    batch = collate_fn_for_tokengt([graph_data])
    if batch is None:
        print("Error: Failed to collate the graph data.")
        return
        
    for key in batch:
        if isinstance(batch[key], torch.Tensor):
            batch[key] = batch[key].to(device)

    # --- 3. Run Inference and Get Attention ---
    print("\n--- Running model inference to extract attention weights ---")
    with torch.no_grad():
        logits, extra = model(batch=batch)
    
    attn_last = extra.get('attn_last', None)
    if attn_last is None:
        attn_all = extra.get('attn_all', None)
        if attn_all:
            attn_last = attn_all[-1]

    if attn_last is None:
        print("\nERROR: The model did not return attention weights. Check 'return_attention' and model forward.\n")
        return
        
    attention_map = attn_last[0].cpu()
    if attention_map.dim() == 3: attention_map = attention_map.mean(dim=0)
    if attention_map.dim() != 2:
        print(f"\nERROR: Unexpected attention map dimension: {attention_map.dim()}\n")
        return
    
    attention_map_np = attention_map.numpy()
    
    display_tokens = get_descriptive_tokens(graph_data, vocabs, metadata)
    actual_seq_len = len(display_tokens)
    
    padded_len = attention_map_np.shape[0]
    if actual_seq_len > padded_len:
        display_tokens = display_tokens[:padded_len]
    
    attention_map_sliced = attention_map_np[:actual_seq_len, :actual_seq_len]

    print(f"\n--- Exporting attention dataframe for layer {layer + 1} ---")
    
    attention_df = pd.DataFrame(attention_map_sliced, index=display_tokens, columns=display_tokens)
    
    output_filename = os.path.join(OUTPUT_DIR, f"attention_df_{os.path.basename(json_path).replace('.json', '')}_{model_size}_multitask_layer_{layer+1}.csv")
    attention_df.to_csv(output_filename)
    
    print(f"\n✅ Attention DataFrame saved successfully to: {output_filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GlycanGT Multi-task Attention to a CSV file.")
    parser.add_argument("--model_size", type=str, default="large", choices=["ss", "small", "medium", "large"], help="Size of the pre-trained model to load.")
    parser.add_argument("--json_path", type=str, required=True, help="Full path to the preprocessed .json file of the glycan to analyze.")
    parser.add_argument("--layer", type=int, default=-1, help="The encoder layer to visualize. Use -1 for the last layer.")
    
    args = parser.parse_args()
    main(args.model_size, args.json_path, args.layer)