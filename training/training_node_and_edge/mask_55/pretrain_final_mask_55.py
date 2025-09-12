# [Final Version] Pre-trains models on the full dataset using the multi-task objective.

import torch
import torch.nn as nn
import json
import os
import sys
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import random
import numpy as np
import torch.nn.functional as F

# --- path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'model'))

# --- helper function---
from glycan_tokengt_multitask import GlycanTokenGTMultiTask 
from config_tokengt import get_config
from utils import GlycanGraphDataset, collate_fn_for_tokengt, load_vocabs 

# --- constant ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'mask_55')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- helper ---
def vocab_size(v):
    if isinstance(v, list):
        return len(v)
    return (max(v.values()) + 1) if len(v) > 0 else 0

def save_final_train_history(history: dict, filename: str):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'total_loss', 'node_loss', 'edge_loss'])
        for i in range(len(history['total_loss'])):
            writer.writerow([
                i + 1, 
                history['total_loss'][i],
                history['node_loss'][i],
                history['edge_loss'][i]
            ])
    print(f"✅ Saved final training loss to {filename}.")

def run_final_training():
    # --- 1. Setting ---
    model_sizes_to_run = ['ss', 'small', 'medium', 'large']
    
    best_epochs = {'ss': 187, 'small': 167, 'medium': 579, 'large': 450}
    
    hparams = {'learning_rate': 1e-6, 'batch_size': 32, 'edge_loss_weight': 0.5}
    
    print(f"--- TokenGT Final Pre-training (Mask 5 model, Node & Edge Prediction) ---")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # --- seed ---
    seed = 42
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    # --- 2. Vocab and MASK ID setting ---
    monomer_vocab, linkage_vocab = load_vocabs(DATA_DIR)
    
    MASK_TOKEN = "[MASK]"
    def get_mask_id(v):
        return v.index(MASK_TOKEN) if isinstance(v, list) else v[MASK_TOKEN]
    MASK_IDS = {"node": get_mask_id(monomer_vocab), "edge": get_mask_id(linkage_vocab)}
    assert MASK_IDS['node'] != 0 and MASK_IDS['edge'] != 0, "MASK id must not be 0"
    
    vocabs = (monomer_vocab, linkage_vocab)
    print(f"✅ Vocab and MASK IDs prepared. Monomers: {len(monomer_vocab)}, Linkages: {len(linkage_vocab)}")

    # --- 3. Dataloader ---
    processed_data_dir = os.path.join(DATA_DIR, 'processed_expanded_unique')
    all_file_paths = [os.path.join(processed_data_dir, f) for f in os.listdir(processed_data_dir) if f.endswith('.json')]
    
    for fp in all_file_paths:
        with open(fp, "r") as f:
            edges = json.load(f)
            node_ids = [e['in_node_vocab_id'] for e in edges] + [e['out_node_vocab_id'] for e in edges]
            edge_ids = [e['edge_vocab_id'] for e in edges]
            assert 0 not in node_ids, f"Found node vocab_id=0 in file {fp}"
            assert 0 not in edge_ids, f"Found edge vocab_id=0 in file {fp}"
    print("✅ No vocab_id=0 found in any node/edge data.")
    
    full_dataset = GlycanGraphDataset(all_file_paths)
    train_loader = DataLoader(full_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=collate_fn_for_tokengt, num_workers=4, pin_memory=True)
    print(f"✅ Using full dataset with {len(full_dataset)} samples for final training.")

    # --- 4. Loop for each model size ---
    for config_name in model_sizes_to_run:
        num_epochs  = best_epochs[config_name]
        print(f"\n===== Final Training {config_name} ({num_epochs} epochs, mask 55%) =====")
        
        config = get_config(config_name)
        config.update({
            'num_atoms': len(monomer_vocab),
            'num_edges': len(linkage_vocab),
            'orf_node_id': True,
            'lap_node_id': False
            })
        
        model = GlycanTokenGTMultiTask(cfg=config).to(device)
        optimizer = AdamW(model.parameters(), lr=hparams['learning_rate'], weight_decay=0.01)

        print("✅ DONE: preparing model, optimizer")
        
        # --- 5. training loop ---
        train_loss_history = {'total_loss': [], 'node_loss': [], 'edge_loss': []}
        model.train() 
        for epoch in range(num_epochs):
            print(f"\n--- Final Training Epoch {epoch + 1}/{num_epochs} ---")
            total_loss, total_node_loss, total_edge_loss = 0.0, 0.0, 0.0
            
            current_ratio = 0.55
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Training (mask ratio: {current_ratio:.2f})")
            
            for batch in progress_bar:
                if batch is None: continue
                
                original_node_data = batch['node_data']
                original_edge_data = batch['edge_data']
                node_mask = torch.rand_like(original_node_data.float()) < current_ratio
                edge_mask = torch.rand_like(original_edge_data.float()) < current_ratio
                if node_mask.sum() == 0 and edge_mask.sum() == 0: continue
                
                masked_batch = batch.copy()
                masked_batch['node_data'] = original_node_data.clone().masked_fill_(node_mask, MASK_IDS['node'])
                masked_batch['edge_data'] = original_edge_data.clone().masked_fill_(edge_mask, MASK_IDS['edge'])
                
                for key, value in masked_batch.items():
                    if isinstance(value, torch.Tensor):
                        masked_batch[key] = value.to(device)
                        
                optimizer.zero_grad()
                logits_dict, extra = model(masked_batch)
                node_logits, edge_logits = logits_dict['node'], logits_dict['edge']
                masked_node_logits = node_logits[node_mask.to(node_logits.device)]
                masked_edge_logits = edge_logits[edge_mask.to(edge_logits.device)]
                
                node_target = original_node_data[node_mask].to(device)
                node_loss = F.cross_entropy(masked_node_logits, node_target) if node_target.numel() > 0 else torch.tensor(0.0, device=device)
                edge_target = original_edge_data[edge_mask].to(device)
                edge_loss = F.cross_entropy(masked_edge_logits, edge_target) if edge_target.numel() > 0 else torch.tensor(0.0, device=device)
                
                loss = node_loss + hparams.get('edge_loss_weight', 0.5) * edge_loss
                if torch.isnan(loss): continue
                
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_node_loss += node_loss.item()
                total_edge_loss += edge_loss.item()
                progress_bar.set_postfix({'loss': loss.item()})

            avg_loss = total_loss / len(progress_bar) if len(progress_bar) > 0 else 0
            avg_node_loss = total_node_loss / len(progress_bar) if len(progress_bar) > 0 else 0
            avg_edge_loss = total_edge_loss / len(progress_bar) if len(progress_bar) > 0 else 0
            
            train_loss_history['total_loss'].append(avg_loss)
            train_loss_history['node_loss'].append(avg_node_loss)
            train_loss_history['edge_loss'].append(avg_edge_loss)
            
            print(f"Epoch {epoch + 1} completed - Mean Loss: {avg_loss:.4f} (Node: {avg_node_loss:.4f}, Edge: {avg_edge_loss:.4f}), LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # --- 6. saving models ---
        save_path = os.path.join(OUTPUT_DIR, f"pretrained_tokengt_{config_name}_multitask_mask_55_final.pt")
        torch.save(model.state_dict(), save_path)
        
        history_filename = os.path.join(OUTPUT_DIR, f"final_pretrain_loss_history_{config_name}_multitask_mask_55.csv")
        save_final_train_history(train_loss_history, history_filename)
        
        # Clean up memory for the next model
        del model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    run_final_training()