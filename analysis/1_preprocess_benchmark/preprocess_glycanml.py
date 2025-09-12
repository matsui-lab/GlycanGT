import pandas as pd
import json
import os
import sys
from tqdm import tqdm
import re

from glycowork.motif.graph import glycan_to_nxGraph

# --- path ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..'))
sys.path.append(os.path.join(PROJECT_ROOT, 'model'))
sys.path.append(os.path.join(PROJECT_ROOT, 'tokenizer'))


GLYCANML_DATA_DIR = '/share3/kitani/GlycanML/data'
GLYCOGT_DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
VOCAB_DIR = os.path.join(GLYCOGT_DATA_DIR, 'vocab_expanded')
OUTPUT_GRAPH_DIR_BASE = os.path.join(GLYCOGT_DATA_DIR, 'processed_benchmark')
OUTPUT_LABEL_DIR = os.path.join(GLYCOGT_DATA_DIR, 'benchmark_labels')

# --- benchmark task ---
TASKS = {
    "taxonomy": {"input_csv": "glycan_classification.csv", "iupac_col": "target", "label_cols": ["domain", "kingdom", "phylum", "class", "order", "family", "genus", "species"]},
    "immunogenicity": {"input_csv": "glycan_immunogenicity.csv", "iupac_col": "glycan", "label_cols": ["immunogenicity"]},
    "glycosylation": {"input_csv": "glycan_properties.csv", "iupac_col": "glycan", "label_cols": ["link"]},
    "interaction": {"input_csv": "glycan_interaction.csv", "iupac_col": "glycan", "protein_col": "target", "label_cols": ["affinity"]}
}

# --- linkage detection helpers (copy from vocab build) ---
_PAT1 = re.compile(r"^[abAB][0-9?/,]+-[0-9?/,]+[abAB]?$")
_PAT2 = re.compile(r"^(?:[abAB])?[0-9?/,]+-[A-Za-z]{1,4}-[0-9?/,]+(?:[abAB])?$")
_PAT3 = re.compile(r"^[0-9?]+(?:[,/][0-9?]+)*-[0-9?]+(?:[,/][0-9?]+)*$")
_PAT4 = re.compile(r"^(?:[abAB])?[0-9?/,]+-$")
_PAT_TRASH = re.compile(r"^[\d?.,/\-\s]+$")

def _normalize_link_label(s: str) -> str:
    return re.sub(r"-([a-zA-Z]{1,4})-", lambda m: f"-{m.group(1).upper()}-", str(s).strip())

def is_linkage(label: str) -> bool:
    if not label or not str(label).strip():
        return True
    s = _normalize_link_label(str(label))
    return bool(_PAT1.fullmatch(s) or _PAT2.fullmatch(s) or _PAT3.fullmatch(s) or _PAT4.fullmatch(s) or _PAT_TRASH.fullmatch(s))

def is_monomer(label: str) -> bool:
    return not is_linkage(label)

def iupac_to_graph_triples(
    iupac: str,
    mono_vocab,  # MonomerVocab
    link_vocab,  # LinkageVocab
):
    """
    IUPAC Condensed → list of triples:
    {in_node_id, in_node_name, edge_id, edge_name, out_node_id, out_node_name,
     in_node_vocab_id, out_node_vocab_id, edge_vocab_id}
    """
    G = glycan_to_nxGraph(iupac)

    monomer_nodes = {
        n: d['string_labels']
        for n, d in G.nodes(data=True)
        if is_monomer(d.get('string_labels', ''))
    }
    # sequential ID 
    node_id_map = {n: idx for idx, n in enumerate(sorted(monomer_nodes))}

    records = []
    for n, d in G.nodes(data=True):
        label = d.get('string_labels', '')
        if not is_monomer(label):  
            preds = list(G.predecessors(n))
            succs = list(G.successors(n))
            for src in preds:
                for tgt in succs:
                    if src in monomer_nodes and tgt in monomer_nodes:
                        records.append({
                            "in_node_id": node_id_map[src],
                            "in_node_name": monomer_nodes[src],
                            "edge_id": n,
                            "edge_name": label,
                            "out_node_id": node_id_map[tgt],
                            "out_node_name": monomer_nodes[tgt],
                            "in_node_vocab_id": mono_vocab.encode(monomer_nodes[src]),
                            "out_node_vocab_id": mono_vocab.encode(monomer_nodes[tgt]),
                            "edge_vocab_id": link_vocab.encode(label),
                        })
    return records

class Vocab:
    def __init__(self, vocab_path):
        with open(vocab_path, 'r') as f:
            self.vocab_list = json.load(f)
        self.vocab_dict = {name: i for i, name in enumerate(self.vocab_list)}
    def encode(self, name: str) -> int:
        return self.vocab_dict.get(name, 1)
    
def get_split_from_row(row):
    if row.get('train') == 1 or row.get('train') == '1' or row.get('train') == True:
        return 'train'
    if row.get('valid') == 1 or row.get('validation') == 1 or row.get('valid') == '1' or row.get('validation') == '1' or row.get('valid') == True or row.get('validation') == True:
        return 'val'
    if row.get('test') == 1 or row.get('test') == '1' or row.get('test') == True:
        return 'test'
    return 'unknown'

def process_standard_task(task_name, params, vocabs):
    """general function for preprocessing long formatted csv"""
    print(f"\n--- Start preprocessing task '{task_name}' ---")
    input_csv_path = os.path.join(GLYCANML_DATA_DIR, params["input_csv"])
    df = pd.read_csv(input_csv_path)
    
    output_graph_dir = os.path.join(OUTPUT_GRAPH_DIR_BASE, task_name)
    os.makedirs(output_graph_dir, exist_ok=True)
    
    label_records = []
    error_count = 0
    for index, row in tqdm(df.iterrows(), total=len(df), desc=f"{task_name} is in process"):
        iupac_string = row.get(params["iupac_col"])
        if pd.isna(iupac_string):
            error_count += 1
            continue

        try:
            graph_triples = iupac_to_graph_triples(iupac_string, vocabs['monomer'], vocabs['linkage'])
            if graph_triples:
                graph_id = f"{task_name}_{index}"
                with open(os.path.join(output_graph_dir, f"{graph_id}.json"), 'w') as f:
                    json.dump(graph_triples, f)
                
                label_info = {'graph_id': graph_id, 'split': get_split_from_row(row)}
                for col in params["label_cols"]:
                    label_info[col] = row.get(col)
                label_records.append(label_info)
            else:
                error_count += 1
        except Exception:
            error_count += 1

    print(f"✅ DONE: graph construction. success: {len(label_records)}, fail/skip: {error_count}")
    if label_records:
        label_df = pd.DataFrame(label_records)
        output_label_file = os.path.join(OUTPUT_LABEL_DIR, f"{task_name}_labels.csv")
        label_df.to_csv(output_label_file, index=False)
        print(f"✅ Saved label information in {output_label_file}.")

def process_interaction_task(task_name, params, vocabs):
    """The function for preprocessing wide formatted interaction.csv"""
    print(f"\n--- Start preprocessing task '{task_name}'. ---")
    input_csv_path = os.path.join(GLYCANML_DATA_DIR, params["input_csv"])
    df = pd.read_csv(input_csv_path)
    
    output_graph_dir = os.path.join(OUTPUT_GRAPH_DIR_BASE, task_name)
    os.makedirs(output_graph_dir, exist_ok=True)

    # Unique glycan to graph
    glycan_cols = [col for col in df.columns if col not in ['target', 'train', 'valid', 'test']]
    print(f"Number of glycans found: {len(glycan_cols)}. Converting into graph format ...")
    glycan_to_graph_id = {}
    for i, iupac_string in enumerate(tqdm(glycan_cols, desc="Converting all glycan into Graph")):
        try:
            graph_triples = iupac_to_graph_triples(iupac_string, vocabs['monomer'], vocabs['linkage'])
            if graph_triples:
                graph_id = f"{task_name}_glycan_{i}"
                with open(os.path.join(output_graph_dir, f"{graph_id}.json"), 'w') as f:
                    json.dump(graph_triples, f)
                glycan_to_graph_id[iupac_string] = graph_id
        except Exception:
            continue
    print(f"✅ Saved {len(glycan_to_graph_id)} unique glycan graphs")

    # Converting into long format
    label_records = []
    protein_col = params['protein_col']
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Creating interaction pairs"):
        protein_seq = row.get(protein_col)
        split = get_split_from_row(row)
        for glycan_iupac in glycan_cols:
            affinity = row.get(glycan_iupac)
            if pd.notna(affinity) and glycan_iupac in glycan_to_graph_id:
                label_info = {
                    'graph_id': glycan_to_graph_id[glycan_iupac],
                    'protein_sequence': protein_seq,
                    'affinity': affinity,
                    'split': split
                }
                label_records.append(label_info)

    print(f"✅ Created {len(label_records)} Protein-Glycan interaction pairs.")
    if label_records:
        label_df = pd.DataFrame(label_records)
        output_label_file = os.path.join(OUTPUT_LABEL_DIR, f"{task_name}_labels.csv")
        label_df.to_csv(output_label_file, index=False)
        print(f"✅ Saved label information in {output_label_file} ")

def main():
    print("====== START Preprocessing: GLYCANML Benchmark Data ======")
    os.makedirs(OUTPUT_LABEL_DIR, exist_ok=True)

    vocabs = {
        'monomer': Vocab(os.path.join(VOCAB_DIR, 'monomer.json')),
        'linkage': Vocab(os.path.join(VOCAB_DIR, 'linkage.json'))
    }
    
    for task_name, params in TASKS.items():
        if task_name == 'interaction':
            process_interaction_task(task_name, params, vocabs)
        else:
            process_standard_task(task_name, params, vocabs)
    
    print("\n🎉🎉🎉 COMPLETED ALL PREPROCESS 🎉🎉🎉")

if __name__ == "__main__":
    main()