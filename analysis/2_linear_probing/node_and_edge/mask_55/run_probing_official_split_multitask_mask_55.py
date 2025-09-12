import torch
import torch.nn as nn
import json
import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

# --- scikit-learn & LightGBM ---
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform, randint, uniform
import lightgbm as lgb

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.getLogger("lightgbm").setLevel(logging.ERROR)

# --- path setting ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..', '..'))
sys.path.append("/path/to/glycoGT/model")

# --- import common functions from utils.py ---
from utils import GlycanGraphDataset, collate_fn_for_tokengt, load_vocabs
from glycan_tokengt_multitask import GlycanTokenGTMultiTask
from config_tokengt import get_config

# --- constants ---
GLYCOGT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
PREPROCESSED_DIR = os.path.join(GLYCOGT_DATA_DIR, 'processed_benchmark')
LABEL_DIR = os.path.join(GLYCOGT_DATA_DIR, 'benchmark_labels')
PRETRAINED_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'mask_55')
FEATURE_CACHE_DIR = os.path.join(PROJECT_ROOT, 'outputs', 'feature_cache_mask_55')

# --- Define all classification tasks to be evaluated ---
TASKS = {
    "taxonomy": {"label_file": "taxonomy_labels.csv", "target_cols": ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"], "graph_subdir": "taxonomy"},
    "immunogenicity": {"label_file": "immunogenicity_labels.csv", "target_cols": ["immunogenicity"], "graph_subdir": "immunogenicity", "type": "binary"},
    "glycosylation": {"label_file": "glycosylation_labels.csv", "target_cols": ["link"], "graph_subdir": "glycosylation"},
}

def load_pretrained_multitask_model(size, vocabs, pretrained_dir, device):
    """
    Loads a pre-trained GlycanTokenGTMultiTask model.
    """
    monomer_vocab, linkage_vocab = vocabs
    config = get_config(size)
    config.update({'num_atoms': len(monomer_vocab), 'num_edges': len(linkage_vocab), 'orf_node_id': True, 'lap_node_id': False})

    model = GlycanTokenGTMultiTask(cfg=config)
    model_path = os.path.join(pretrained_dir, f"pretrained_tokengt_{size}_multitask_mask_55_final.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained model not found at {model_path}")

    print(f"Loading pre-trained multi-task model from: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# --- helper functions ---
def extract_features(model, graph_dir, graph_ids, device):
    """extract graph representations using the model"""
    print(f"--- Extracting features for {len(graph_ids)} graphs ---")
    file_paths = [os.path.join(graph_dir, f"{gid}.json") for gid in graph_ids]
    dataset = GlycanGraphDataset(file_paths)
    data_loader = DataLoader(dataset, batch_size=256, shuffle=False, collate_fn=collate_fn_for_tokengt, num_workers=4, pin_memory=True)
    all_features = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc=f"Extracting Features"):
            if batch is None: continue
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)
            
            _, extra = model(batch)
            all_features.append(extra['graph_rep'].cpu())
            
    return torch.cat(all_features, dim=0) if all_features else torch.empty(0)

def search_and_evaluate(X_train, y_train, X_test, y_test, seed, classifier_type, is_binary):
    """Performs HPO on the training data and evaluates the best model on the test data."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    if classifier_type == 'svm':
        model = SVC(class_weight='balanced', probability=True)
        param_distributions = [
            {'kernel': ['rbf'], 'C': loguniform(1e-3, 1e2), 'gamma': loguniform(1e-4, 1e-1)},
            {'kernel': ['linear'], 'C': loguniform(1e-3, 1e2)}
        ]
        n_iter_search = 10
    elif classifier_type == 'lgbm':
        model = lgb.LGBMClassifier(class_weight='balanced', random_state=seed, n_jobs=1, verbosity=-1)
        param_distributions = {
            'n_estimators': randint(100, 1000),
            'learning_rate': loguniform(0.01, 0.3),
            'num_leaves': randint(20, 150),
            'max_depth': randint(5, 20),
            'reg_alpha': loguniform(1e-2, 1e1),
            'reg_lambda': loguniform(1e-2, 1e1),
            'colsample_bytree': uniform(0.6, 0.4),
            'subsample': uniform(0.6, 0.4),
        }
        n_iter_search = 15
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")

    random_search = RandomizedSearchCV(
        model, param_distributions, n_iter=n_iter_search, cv=3, random_state=seed, n_jobs=-1,
        scoring='f1_macro'
    )

    print(f"    -> Running RandomizedSearchCV for {classifier_type.upper()} (seed={seed})...")
    random_search.fit(X_train_scaled, y_train)

    best_clf = random_search.best_estimator_
    best_params = random_search.best_params_
    
    print(f"    -> Best HPO params for {classifier_type.upper()}: {random_search.best_params_}")
    y_pred = best_clf.predict(X_test_scaled)

    scores = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Macro-F1': f1_score(y_test, y_pred, average='macro', zero_division=0),
        'Macro-Precision': precision_score(y_test, y_pred, average='macro', zero_division=0),
        'Macro-Recall': recall_score(y_test, y_pred, average='macro', zero_division=0)
    }
    if is_binary:
        y_prob = best_clf.predict_proba(X_test_scaled)[:, 1]
        scores['AUPRC'] = average_precision_score(y_test, y_prob)

    return scores, best_params 

# --- main execution function ---
def main():
    print("====== Linear Probing with Official Splits for mask 55 Multi-task Model ======")
    os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    vocabs = load_vocabs(GLYCOGT_DATA_DIR)
    all_results = []
    all_params = []
    
    monomer_vocab, linkage_vocab = vocabs
    if "[MASK]" not in monomer_vocab:
        monomer_vocab.append("[MASK]")
    if "[MASK]" not in linkage_vocab:
        linkage_vocab.append("[MASK]")
    
    vocabs = (monomer_vocab, linkage_vocab)
    
    model_sizes = ['ss', 'small', 'medium', 'large']
    
    seeds_to_run = [0, 1, 2]
    classifiers_to_run = ['svm', 'lgbm']

    for size in model_sizes:
        print(f"\n{'='*20} Evaluating Model Size: '{size}' (Multi-task) {'='*20}")
        try:
            model = load_pretrained_multitask_model(size, vocabs, PRETRAINED_DIR, device)
            
        except FileNotFoundError as e:
            print(f"  -> WARNING: {e}. Skipping this model size.")
            continue

        for task_name, params in TASKS.items():
            print(f"\n--- Task Group: '{task_name}' ---")
            label_file_path = os.path.join(LABEL_DIR, params['label_file'])
            if not os.path.exists(label_file_path): continue
            label_df = pd.read_csv(label_file_path)

            feature_cache_path = os.path.join(FEATURE_CACHE_DIR, f"features_{size}_multitask_mask_55_{params['graph_subdir']}.pt")
            
            if os.path.exists(feature_cache_path):
                print(f"Loading cached features: {feature_cache_path}")
                cached_data = torch.load(feature_cache_path)
                features_tensor, cached_graph_ids = cached_data['features'], cached_data['graph_ids']
                feature_df = pd.DataFrame(features_tensor.numpy(), index=cached_graph_ids)
            else:
                unique_graph_ids = label_df['graph_id'].unique()
                graph_dir = os.path.join(PREPROCESSED_DIR, params['graph_subdir'])
                features_tensor = extract_features(model, graph_dir, unique_graph_ids, device)
                torch.save({'graph_ids': unique_graph_ids, 'features': features_tensor}, feature_cache_path)
                print(f"Cached features to: {feature_cache_path}")
                feature_df = pd.DataFrame(features_tensor.numpy(), index=unique_graph_ids)

            merged_df = label_df.merge(feature_df, left_on='graph_id', right_index=True)

            target_cols = params.get('target_cols', [params.get('target_col')])
            for target_col in target_cols:
                print(f"\n--- Evaluating Target: '{target_col}' ---")

                task_data = merged_df[['split', target_col] + list(range(features_tensor.shape[1]))].copy().dropna(subset=[target_col])
                y, class_labels = pd.factorize(task_data[target_col])
                X = task_data.drop(columns=['split', target_col]).values

                train_mask = task_data['split'].isin(['train', 'val', 'validation'])
                test_mask = task_data['split'] == 'test'
                X_train, X_test = X[train_mask], X[test_mask]
                y_train, y_test = y[train_mask], y[test_mask]

                if len(X_test) == 0 or len(X_train) == 0 or len(np.unique(y_test)) < 2:
                    print("  -> WARNING: Not enough data or classes in test set for a meaningful evaluation. Skipping.")
                    continue

                is_binary_task = params.get('type') == 'binary' and len(class_labels) == 2

                for clf_type in classifiers_to_run:
                    print(f"\n  -- Classifier: {clf_type.upper()} --")
                    seed_scores = []
                    for seed in seeds_to_run:
                        scores, best_params = search_and_evaluate(
                            X_train, y_train, X_test, y_test, seed, clf_type, is_binary_task
                            )
                        seed_scores.append(scores)
                        all_params.append({
                            'Model Size': f"{size}_multitask_mask_55",
                            'Task': target_col,
                            'Classifier': f"{clf_type.upper()}",
                            'Seed': seed,
                            **best_params  
                        })

                    df_scores = pd.DataFrame(seed_scores)
                    mean_scores = df_scores.mean(numeric_only=True)
                    std_scores  = df_scores.std(numeric_only=True)

                    print(f"  - {clf_type.upper()} (3-seed HPO Average):")
                    for metric, mean_val in mean_scores.items():
                        std_val = std_scores[metric]
                        print(f"    - {metric}: {mean_val:.4f} ± {std_val:.4f}")
                        all_results.append({
                            'Model Size': f"{size}_multitask_mask_55",
                            'Task': target_col,
                            'Classifier': f"{clf_type.upper()}_HPO",
                            'Metric': metric,
                            'Mean_Score': mean_val,
                            'Std_Score': std_val,
                            'Mask': 55
                        })

    results_df = pd.DataFrame(all_results)
    
    output_path = os.path.join(PROJECT_ROOT, 'analysis', '2_linear_probing', 'node_and_edge', 'mask_55', 'linear_probing_official_split_multitask_mask55_hpo_svm_lgbm_results.csv')
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    results_df.to_csv(output_path, index=False)
    
    params_df = pd.DataFrame(all_params)
    params_output_path = os.path.join(PROJECT_ROOT, 'analysis', '2_linear_probing', 'node_and_edge', 'mask_55',
                                      'linear_probing_official_split_multitask_mask55_hpo_best_params.csv')
    params_df.to_csv(params_output_path, index=False)
    
    print(f"\n🎉🎉🎉 All official split evaluations with HPO (SVM, LGBM) completed! Results saved to {output_path} 🎉🎉🎉")

if __name__ == "__main__":
    main()