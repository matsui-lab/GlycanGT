import os, json, sys, torch, numpy as np, pandas as pd
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score

# ===== path setting =====
PROJECT_ROOT = "/pato/to/glycoGT"
MODEL_PATH   = f"{PROJECT_ROOT}/outputs/mask_35/pretrained_tokengt_large_multitask_mask_35_final.pt"
EVAL_DIR     = f"{PROJECT_ROOT}/data/processed_overlapped"
VOCAB_DIR    = f"{PROJECT_ROOT}/data/vocab_expanded"
OUT_DIR      = f"{PROJECT_ROOT}/analysis/6_downstream/external_test"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_JSON = f"{OUT_DIR}/eval_large_multitask_mask35_sweep.json"
OUT_CSV  = f"{OUT_DIR}/eval_large_multitask_mask35_sweep.csv"

# ===== imports from project =====
sys.path.append(f"{PROJECT_ROOT}/model")
from glycan_tokengt_multitask import GlycanTokenGTMultiTask
from config_tokengt import get_config
from utils import GlycanGraphDataset, collate_fn_for_tokengt

# ===== reproducibility =====
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ===== vocab load =====
with open(os.path.join(VOCAB_DIR, "monomer.json")) as f:
    monomer_vocab = json.load(f)
with open(os.path.join(VOCAB_DIR, "linkage.json")) as f:
    linkage_vocab = json.load(f)
if "[MASK]" not in monomer_vocab:  monomer_vocab.append("[MASK]")
if "[MASK]" not in linkage_vocab: linkage_vocab.append("[MASK]")

NODE_PAD_ID = monomer_vocab.index("[PAD]") if "[PAD]" in monomer_vocab else 0
EDGE_PAD_ID = linkage_vocab.index("[PAD]") if "[PAD]" in linkage_vocab else 0

MASK_IDS = {
    "node": monomer_vocab.index("[MASK]"),
    "edge": linkage_vocab.index("[MASK]")
}

assert MASK_IDS["node"] != NODE_PAD_ID, "node: [MASK] と [PAD] のIDが同じです"
assert MASK_IDS["edge"] != EDGE_PAD_ID, "edge: [MASK] と [PAD] のIDが同じです"

# ===== model =====
device  = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
config  = get_config("large")
config.update({"num_atoms": len(monomer_vocab),
               "num_edges": len(linkage_vocab),
               "orf_node_id": True, "lap_node_id": False})
model = GlycanTokenGTMultiTask(cfg=config).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")

# ===== data loader =====
file_paths = [os.path.join(EVAL_DIR, f) for f in os.listdir(EVAL_DIR) if f.endswith(".json")]
dataset    = GlycanGraphDataset(file_paths)
loader     = DataLoader(dataset, batch_size=256, shuffle=False,
                        collate_fn=collate_fn_for_tokengt,
                        num_workers=4, pin_memory=True)
print(f"✅ Loaded {len(dataset):,} graphs for evaluation")

K_LIST = [1, 2, 3, 5, 10, 20, 30]

# ===== helpers =====
def make_mask_valid_only(x: torch.Tensor, ratio: float, pad_id: int) -> torch.Tensor:
    valid = (x != pad_id)
    if ratio <= 0.0:
        return torch.zeros_like(x, dtype=torch.bool, device=x.device)
    if ratio >= 1.0:
        return valid
    rand = torch.rand_like(x.float(), device=x.device)
    return (rand < ratio) & valid

def slice_node_logits_like(node_logits_all: torch.Tensor,
                           node_data: torch.Tensor,
                           batch: dict) -> torch.Tensor:
    bsz = len(batch["node_num"])
    all_nodes = int(node_data.numel())
    if node_logits_all.size(0) == all_nodes:
        return node_logits_all
    start = 2 * bsz
    end   = start + all_nodes
    if node_logits_all.size(0) >= end:
        return node_logits_all[start:end]
    raise RuntimeError(
        f"Unexpected node logits shape: {tuple(node_logits_all.shape)}, "
        f"expected all_nodes={all_nodes} with or without 2*bsz offset (bsz={bsz})."
    )

def append_metrics_for(prefix: str,
                       trues: torch.Tensor,
                       preds: torch.Tensor,
                       logits_sel: torch.Tensor,
                       store: dict):
    if trues.numel() == 0:  # safety
        return
    trues_cpu = trues.cpu()
    preds_cpu = preds.cpu()
    loss = torch.nn.functional.cross_entropy(logits_sel, trues.to(logits_sel.device)).item()
    store[f"{prefix}_loss"].append(loss)
    store[f"{prefix}_acc"].append((preds_cpu == trues_cpu).float().mean().item())
    top5_idx = logits_sel.topk(5, dim=1).indices.cpu()
    store[f"{prefix}_top5"].append((top5_idx == trues_cpu.unsqueeze(1)).any(1).float().mean().item())
    store[f"{prefix}_f1"].append(f1_score(trues_cpu, preds_cpu, average="macro", zero_division=0))
    store[f"{prefix}_prec"].append(precision_score(trues_cpu, preds_cpu, average="macro", zero_division=0))
    store[f"{prefix}_rec"].append(recall_score(trues_cpu, preds_cpu, average="macro", zero_division=0))

def ranks_for(logits_sel: torch.Tensor, trues_sel: torch.Tensor) -> list:
    if trues_sel.numel() == 0:
        return []
    trues_gpu = trues_sel.to(logits_sel.device)
    return (
        logits_sel.argsort(dim=1, descending=True)
                  .eq(trues_gpu.unsqueeze(1))
                  .nonzero(as_tuple=False)[:, 1]
                  .cpu().tolist()
    )

def coverage(ranks: list, k: int) -> float:
    return float(np.mean(np.array(ranks) < k)) if ranks else 0.0

@torch.no_grad()
def evaluate_condition(node_ratio: float, edge_ratio: float):
    metrics = defaultdict(list)
    node_ranks, edge_ranks = [], []

    total_node_valid = total_edge_valid = 0
    total_node_masked_valid = total_edge_masked_valid = 0
    total_node_elems = total_edge_elems = 0
    total_node_pad   = total_edge_pad   = 0

    for batch in tqdm(loader, desc=f"Eval node={int(node_ratio*100)}% edge={int(edge_ratio*100)}%"):
        node_data = batch["node_data"].to(device, non_blocking=True)
        edge_data = batch["edge_data"].to(device, non_blocking=True)
        assert (node_data != 0).all(), "node_data に 0（PAD 予約ID）が含まれています"
        assert (edge_data != 0).all(), "edge_data に 0（PAD 予約ID）が含まれています"
        

        node_valid = (node_data != NODE_PAD_ID)
        edge_valid = (edge_data != EDGE_PAD_ID)

        node_mask  = make_mask_valid_only(node_data, node_ratio, NODE_PAD_ID)
        edge_mask  = make_mask_valid_only(edge_data, edge_ratio, EDGE_PAD_ID)
        if node_mask.sum().item() == 0 and edge_mask.sum().item() == 0:
            continue

        total_node_valid         += int(node_valid.sum().item())
        total_edge_valid         += int(edge_valid.sum().item())
        total_node_masked_valid  += int((node_mask & node_valid).sum().item())
        total_edge_masked_valid  += int((edge_mask & edge_valid).sum().item())

        total_node_elems         += int(node_data.numel())
        total_edge_elems         += int(edge_data.numel())
        total_node_pad           += int((node_data == NODE_PAD_ID).sum().item())
        total_edge_pad           += int((edge_data == EDGE_PAD_ID).sum().item())

        masked_batch = {k: v.clone() if torch.is_tensor(v) else v for k, v in batch.items()}

        masked_batch["node_data"] = node_data.clone().masked_fill_(node_mask, MASK_IDS["node"])
        masked_batch["edge_data"] = edge_data.clone().masked_fill_(edge_mask, MASK_IDS["edge"])
        for k, v in masked_batch.items():
            if isinstance(v, torch.Tensor):
                masked_batch[k] = v.to(device, non_blocking=True)

        logits_dict, _ = model(masked_batch)
        node_logits_all, edge_logits = logits_dict["node"], logits_dict["edge"]

        # ---- node ----
        node_logits = slice_node_logits_like(node_logits_all, node_data, batch)
        node_sel = node_mask & node_valid
        if node_sel.any():
            node_logits_sel = node_logits[node_sel]
            node_trues_sel  = node_data[node_sel]
            node_preds_sel  = node_logits_sel.argmax(dim=1)
            append_metrics_for("node", node_trues_sel, node_preds_sel, node_logits_sel, metrics)
            node_ranks.extend(ranks_for(node_logits_sel, node_trues_sel))

        # ---- edge ----
        edge_sel = edge_mask & edge_valid
        if edge_sel.any():
            edge_logits_sel = edge_logits[edge_sel]
            edge_trues_sel  = edge_data[edge_sel]
            edge_preds_sel  = edge_logits_sel.argmax(dim=1)
            append_metrics_for("edge", edge_trues_sel, edge_preds_sel, edge_logits_sel, metrics)
            edge_ranks.extend(ranks_for(edge_logits_sel, edge_trues_sel))

    report = {m: float(np.mean(v)) for m, v in metrics.items() if len(v) > 0}
    for k in K_LIST:
        report[f"node_hit@{k}"] = coverage(node_ranks, k)
        report[f"edge_hit@{k}"] = coverage(edge_ranks, k)

    report["node_mask_ratio_requested"] = float(node_ratio)
    report["edge_mask_ratio_requested"] = float(edge_ratio)
    report["node_mask_ratio_realized_over_valid"] = (
        float(total_node_masked_valid / total_node_valid) if total_node_valid > 0 else 0.0
    )
    report["edge_mask_ratio_realized_over_valid"] = (
        float(total_edge_masked_valid / total_edge_valid) if total_edge_valid > 0 else 0.0
    )

    report["node_pad_fraction_overall"] = (
        float(total_node_pad / total_node_elems) if total_node_elems > 0 else 0.0
    )
    report["edge_pad_fraction_overall"] = (
        float(total_edge_pad / total_edge_elems) if total_edge_elems > 0 else 0.0
    )

    report["num_graphs"] = len(dataset)
    return report

# ===== sweep settings =====
node_ratios_A = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # edge=0%
node_ratios_B = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]  # edge=100%

results = {}
rows_for_csv = []

for r in node_ratios_A:
    rep = evaluate_condition(node_ratio=r, edge_ratio=0.0)
    key = f"node{int(r*100)}_edge0"
    results[key] = rep
    rows_for_csv.append({"condition": key, **rep})

for r in node_ratios_B:
    rep = evaluate_condition(node_ratio=r, edge_ratio=1.0)
    key = f"node{int(r*100)}_edge100"
    results[key] = rep
    rows_for_csv.append({"condition": key, **rep})

# ===== save =====
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
pd.DataFrame(rows_for_csv).to_csv(OUT_CSV, index=False)
print(f"\n✅ Saved JSON to {OUT_JSON}")
print(f"✅ Saved CSV to {OUT_CSV}")
