# [Final Version] Pre-training script for finding optimal epochs by predicting both nodes and edges.

import torch
import torch.nn as nn
import json
import os
import sys
import csv
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.model_selection import train_test_split 
import random
import numpy as np
import torch.nn.functional as F

# --- path setting ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Navigate up three levels from script's dir to reach the project root
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'model'))

# --- import models and helpers ---
from glycan_tokengt_multitask import GlycanTokenGTMultiTask
from config_tokengt import get_config
from utils import GlycanGraphDataset, collate_fn_for_tokengt, load_vocabs

# --- constants ---
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'outputs')

# --- helper functions ---
def vocab_size(v):
    if isinstance(v, list):
        return len(v)
    return (max(v.values()) + 1) if len(v) > 0 else 0

def count_parameters(model: nn.Module):
    """count model parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def save_loss_history(history: dict, filename: str):
    """Saves the loss history to a CSV file."""
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch', 'train_loss', 'val_loss', 'avg_train_node_loss', 'avg_train_edge_loss', 'avg_val_node_loss', 'avg_val_edge_loss'])
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1, 
                history['train_loss'][i], 
                history['val_loss'][i],
                history['avg_train_node_loss'][i],
                history['avg_train_edge_loss'][i],
                history['avg_val_node_loss'][i],
                history['avg_val_edge_loss'][i]
            ])
    print(f"✅ Loss history saved to {filename}")

def save_summary_results(results: dict):
    """Saves the summary of training results to a text file."""
    output_path = os.path.join(OUTPUT_DIR, 'training_summary_mask_35.txt')
    with open(output_path, 'w') as f:
        f.write("--- TokenGT Multi-Task Pre-training Summary ---\n\n")
        for size, data in results.items():
            f.write(f"Model Size: {size}\n")
            f.write(f"  - Total Parameters: {data['total_params']:,}\n")
            f.write(f"  - Best Epoch: {data['best_epoch']}\n")
            f.write(f"  - Best Validation Loss: {data['best_val_loss']:.4f}\n\n")
    print(f"\n✅ Experiment summary saved to {output_path}")

# --- main training function ---
def train_and_validate(config_name: str, vocabs: tuple, all_file_paths: list, hparams: dict, MASK_IDS: dict):
    print(f"\n{'='*20} Start training: model='{config_name}' (Node & Edge Prediction) {'='*20}")
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    monomer_vocab, linkage_vocab = vocabs
    
    config = get_config(config_name)
    config.update({
    'num_atoms': vocab_size(monomer_vocab),
    'num_edges': vocab_size(linkage_vocab),
    'orf_node_id': True,
    'lap_node_id': False})

    train_paths, val_paths = train_test_split(all_file_paths, test_size=0.1, random_state=42)
    train_dataset = GlycanGraphDataset(train_paths)
    val_dataset = GlycanGraphDataset(val_paths)
    train_loader = DataLoader(train_dataset, batch_size=hparams['batch_size'], shuffle=True, collate_fn=collate_fn_for_tokengt, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=hparams['batch_size'], shuffle=False, collate_fn=collate_fn_for_tokengt, num_workers=4, pin_memory=True)
    print(f"✅ Data splitting: Training data {len(train_dataset)}, Validation data {len(val_dataset)}")
    
    model = GlycanTokenGTMultiTask(cfg=config).to(device)
    optimizer = AdamW(model.parameters(), lr=hparams['learning_rate'], weight_decay=0.01)
    
    total_params, _ = count_parameters(model)
    print(f"✅ Model and optimizer ready. Total Parameters: {total_params:,}")

    best_val_loss, best_epoch, patience_counter = float('inf'), 0, 0
    patience = hparams['patience']
    history = {'train_loss': [], 'val_loss': [], 'avg_train_node_loss': [], 'avg_train_edge_loss': [], 'avg_val_node_loss': [], 'avg_val_edge_loss': []}

    for epoch in range(hparams['num_epochs']):
        print(f"\n--- Epoch {epoch + 1}/{hparams['num_epochs']} ---")
        
        model.train()
        total_train_loss, total_node_loss, total_edge_loss = 0.0, 0.0, 0.0
        
        current_ratio = hparams['masking_ratio']
        train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train (mask ratio: {current_ratio:.2f})")
        
        for batch in train_progress_bar:
            if batch is None: continue
            
            original_node_data, original_edge_data = batch['node_data'], batch['edge_data']
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
            node_target = original_node_data[node_mask].to(device)
            node_loss = F.cross_entropy(masked_node_logits, node_target) if node_target.numel() > 0 else torch.tensor(0.0, device=device)

            masked_edge_logits = edge_logits[edge_mask.to(edge_logits.device)]
            edge_target = original_edge_data[edge_mask].to(device)
            edge_loss = F.cross_entropy(masked_edge_logits, edge_target) if edge_target.numel() > 0 else torch.tensor(0.0, device=device)
            
            loss = node_loss + hparams.get('edge_loss_weight', 0.5) * edge_loss
            if torch.isnan(loss): continue

            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_node_loss += node_loss.item()
            total_edge_loss += edge_loss.item()
            train_progress_bar.set_postfix({'loss': loss.item(), 'node': node_loss.item(), 'edge': edge_loss.item()})

        avg_train_loss = total_train_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_node_loss = total_node_loss / len(train_loader) if len(train_loader) > 0 else 0
        avg_edge_loss = total_edge_loss / len(train_loader) if len(train_loader) > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['avg_train_node_loss'].append(avg_node_loss)
        history['avg_train_edge_loss'].append(avg_edge_loss)
        
        # --- Validation Loop ---
        model.eval()
        total_val_loss, total_val_node_loss, total_val_edge_loss = 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None: continue
                
                original_node_data, original_edge_data = batch['node_data'], batch['edge_data']
                node_mask = torch.rand_like(original_node_data.float()) < current_ratio
                edge_mask = torch.rand_like(original_edge_data.float()) < current_ratio
                if node_mask.sum() == 0 and edge_mask.sum() == 0: continue

                masked_batch = batch.copy()
                masked_batch['node_data'] = original_node_data.clone().masked_fill_(node_mask, MASK_IDS['node'])
                masked_batch['edge_data'] = original_edge_data.clone().masked_fill_(edge_mask, MASK_IDS['edge'])

                for key, value in masked_batch.items():
                    if isinstance(value, torch.Tensor):
                        masked_batch[key] = value.to(device)
                
                logits_dict, _ = model(masked_batch)
                
                node_logits, edge_logits = logits_dict['node'], logits_dict['edge']
                masked_node_logits = node_logits[node_mask.to(node_logits.device)]
                
                node_target = original_node_data[node_mask].to(device)
                node_loss = F.cross_entropy(masked_node_logits, node_target) if node_target.numel() > 0 else torch.tensor(0.0, device=device)
                
                masked_edge_logits = edge_logits[edge_mask.to(edge_logits.device)]
                edge_target = original_edge_data[edge_mask].to(device)
                edge_loss = F.cross_entropy(masked_edge_logits, edge_target) if edge_target.numel() > 0 else torch.tensor(0.0, device=device)
                
                loss = node_loss + hparams.get('edge_loss_weight', 0.5) * edge_loss
                if not torch.isnan(loss):
                    total_val_loss += loss.item()
                    total_val_node_loss += node_loss.item()
                    total_val_edge_loss += edge_loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader) if len(val_loader) > 0 else float('inf')
        avg_val_node_loss = total_val_node_loss / len(val_loader) if len(val_loader) > 0 else 0
        avg_val_edge_loss = total_val_edge_loss / len(val_loader) if len(val_loader) > 0 else 0
        
        print(f"Epoch {epoch + 1} Completed - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        history['val_loss'].append(avg_val_loss)
        history['avg_val_node_loss'].append(avg_val_node_loss)
        history['avg_val_edge_loss'].append(avg_val_edge_loss)

        if avg_val_loss < best_val_loss:
            print(f"  -> Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f})")
            best_val_loss, best_epoch, patience_counter = avg_val_loss, epoch + 1, 0
            # Optional: save the best model
            # torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, f"best_model_{config_name}.pth"))
        else:
            patience_counter += 1
            print(f"  -> No improvement in validation loss for {patience_counter} epochs.")
        
        if patience_counter >= patience:
            print(f"--- Early stopping triggered after {epoch + 1} epochs. ---")
            break

    print(f"\nTraining completed for model '{config_name}'. Best epoch: {best_epoch}, Best Val Loss: {best_val_loss:.4f}")
    return total_params, best_epoch, best_val_loss, history

if __name__ == "__main__":
    # Hyperparameters
    hparams = {'num_epochs': 1000, 'patience': 30, 'learning_rate': 1e-6, 'batch_size': 32, 'masking_ratio': 0.35, 'edge_loss_weight': 0.5}
    model_sizes_to_run = ['ss','small', 'medium', 'large']
    
    # --- Setup ---
    results = {}
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # --- Load Data and Vocabs ---
    monomer_vocab, linkage_vocab = load_vocabs(DATA_DIR)
    MASK_TOKEN = "[MASK]"
    def get_mask_id(vocab):
        return vocab.index(MASK_TOKEN) if isinstance(vocab, list) else vocab[MASK_TOKEN]
    MASK_IDS = {"node": get_mask_id(monomer_vocab), "edge": get_mask_id(linkage_vocab)}
    assert MASK_IDS['node'] != 0 and MASK_IDS['edge'] != 0, "MASK id must not be 0"
    vocabs = (monomer_vocab, linkage_vocab)
    
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
    print(f"Found {len(all_file_paths)} total data files.")

    # --- Run Training Loop for each model size ---
    for size in model_sizes_to_run:
        local_hp = hparams.copy()
        # Adjust patience based on model size
        if size == 'ss' or size == 'small':
            local_hp['patience'] = 15
        elif size == 'medium' or size == 'large':
            local_hp['patience'] = 30
        
        total_params, best_epoch, best_val_loss, loss_history = train_and_validate(
            size, vocabs, all_file_paths, local_hp, MASK_IDS
        )
        results[size] = {'total_params': total_params, 'best_epoch': best_epoch, 'best_val_loss': best_val_loss}
        
        history_filename = os.path.join(OUTPUT_DIR, f"loss_history_{size}_multitask_mask_35.csv")
        save_loss_history(loss_history, history_filename)
        save_summary_results(results) # Save summary after each model finishes