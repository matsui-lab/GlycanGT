import os, sys, json, torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import re

# ====== path settings ======
PROJECT_ROOT = "/path/to/glycoGT"
MODEL_PATH   = f"{PROJECT_ROOT}/outputs/mask_35/pretrained_tokengt_large_multitask_mask_35_final.pt"
VOCAB_DIR    = f"{PROJECT_ROOT}/data/vocab_expanded"
JSON_DIR     = f"{PROJECT_ROOT}/data/processed_ambiguous"
SOURCE_CSV   = f"{PROJECT_ROOT}/data/clean/glycosmos_with_ambiguous.csv"
OUT_DIR      = f"{PROJECT_ROOT}/analysis/6_downstream/ambiguous_fill_large"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_LONG_CSV = os.path.join(OUT_DIR, "ambiguous_predictions_long.csv")
OUT_BY_GID   = os.path.join(OUT_DIR, "ambiguous_predictions_by_glycan.csv")
OUT_MERGED   = os.path.join(OUT_DIR, "glycosmos_with_ambiguous_with_predictions.csv")

# ====== imports from project ======
sys.path.append(f"{PROJECT_ROOT}/model")
from glycan_tokengt_multitask import GlycanTokenGTMultiTask
from config_tokengt import get_config
from utils import GlycanGraphDataset, collate_fn_for_tokengt

# ====== reproducibility ======
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# ====== load vocabs ======
with open(os.path.join(VOCAB_DIR, "monomer.json")) as f:
    monomer_vocab: List[str] = json.load(f)
with open(os.path.join(VOCAB_DIR, "linkage.json")) as f:
    linkage_vocab: List[str] = json.load(f)

if "[MASK]" not in monomer_vocab:
    monomer_vocab.append("[MASK]")
if "[MASK]" not in linkage_vocab:
    linkage_vocab.append("[MASK]")

NODE_PAD_ID = monomer_vocab.index("[PAD]") if "[PAD]" in monomer_vocab else 0
EDGE_PAD_ID = linkage_vocab.index("[PAD]") if "[PAD]" in linkage_vocab else 0
NODE_MASK_ID = monomer_vocab.index("[MASK]")
EDGE_MASK_ID = linkage_vocab.index("[MASK]")

assert NODE_MASK_ID != NODE_PAD_ID, "node: [MASK] and [PAD] IDs are same"
assert EDGE_MASK_ID != EDGE_PAD_ID, "edge: [MASK] and [PAD] IDs are same"

# ====== model ======
device  = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
config  = get_config("large")
config.update({
    "num_atoms": len(monomer_vocab),
    "num_edges": len(linkage_vocab),
    "orf_node_id": True,
    "lap_node_id": False
})
model = GlycanTokenGTMultiTask(cfg=config).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"✅ Loaded model from {MODEL_PATH}")

_RE_EDGE_AMBIG = re.compile(r"""
    (                              # 1) ( ...?... ) 括弧内に?あり
        \(
        [^)]* \? [^)]*
        \)
    )
    |
    (                              # 2) 末尾が (?... で未閉じ
        \(
        [^)]* \? [^)]*
        \Z
    )
    |
    (                              # 3) 括弧なしの結合で?を含む: ?1-?, a1-?, ?1-6 など
        (?<![A-Za-z0-9\)])                         # 左側が英数や ')' に隣接しない
        (                                         
            [abAB]?                                # 任意の anomer
            (?:[\d/,]*\?[\d/,]*|[\d/,]+)          # 左側: ? を含む もしくは通常（後段で右側?に期待）
            -
            (?:[\d/,]*\?[\d/,]*|[\d/,]+)          # 右側: ? を含む もしくは通常（前段で左側?に期待）
            [abAB]?                                # 任意の anomer
        )
        (?![A-Za-z0-9\(])                          # 右側が英数や '(' に隣接しない
    )
""", re.VERBOSE)

_RE_NODE_AMBIG = re.compile(r"[^\s()\[\]]*\?[^\s()\[\]]*")

_RE_ANY_Q = re.compile(r"[^\s()\[\]]*?\?[^\s()\[\]]*")

def replace_iupac_sequential(iupac: str, node_preds: List[str], edge_preds: List[str]) -> str:
    s = iupac or ""

    def _edge_repl(m):
        rep = edge_preds.pop(0) if edge_preds else m.group(0)
        g0 = m.group(0)

        starts_paren = g0.startswith("(")
        ends_paren   = g0.endswith(")")
        if starts_paren and ends_paren:
            if not (rep.startswith("(") and rep.endswith(")")):
                rep = f"({rep.strip('()')})"
        elif starts_paren and not ends_paren:
            if not rep.startswith("("):
                rep = "(" + rep
        return rep

    s = _RE_EDGE_AMBIG.sub(_edge_repl, s)

    def _node_repl(m):
        return node_preds.pop(0) if node_preds else m.group(0)

    s = _RE_NODE_AMBIG.sub(_node_repl, s)

    def _any_repl(m):
        if edge_preds:
            return edge_preds.pop(0)
        if node_preds:
            return node_preds.pop(0)
        return m.group(0)

    s = _RE_ANY_Q.sub(_any_repl, s)
    return s

# ====== helpers ======
def load_q_id_sets(json_path: Path):
    with json_path.open("r") as f:
        triples = json.load(f)
    q_nodes, q_edges = set(), set()
    for rec in triples:
        if rec.get("in_node_has_q") or "?" in str(rec.get("in_node_name_raw", "")):
            q_nodes.add(int(rec["in_node_id"]))
        if rec.get("out_node_has_q") or "?" in str(rec.get("out_node_name_raw", "")):
            q_nodes.add(int(rec["out_node_id"]))

        if rec.get("edge_has_q") or "?" in str(rec.get("edge_name_raw", "")):
            if "edge_local_id" in rec:
                q_edges.add(int(rec["edge_local_id"]))
            else:
                q_edges.add(int(rec["edge_id"]))
    return q_nodes, q_edges

def slice_node_logits_like(node_logits_all: torch.Tensor,
                           node_data: torch.Tensor,
                           batch: Dict[str, Any]) -> torch.Tensor:
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

@torch.no_grad()
def softmax_rows(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(x, dim=1)

def id2tok(vocab: List[str], ids: torch.Tensor) -> List[str]:
    return [vocab[int(i)] for i in ids.tolist()]

# ====== IO ======
src_df = pd.read_csv(SOURCE_CSV, dtype=str)
gid_list = src_df["GlyTouCan ID"].astype(str).tolist()

json_dir = Path(JSON_DIR)

# ====== main loop (per glycan) ======
rows_long: List[Dict[str, Any]] = []
by_gid: Dict[str, Dict[str, Any]] = {}
skipped = []

print(f"Processing {len(gid_list)} glycans...")
for gid in tqdm(gid_list):
    
    entry = by_gid.setdefault(gid, {
    "nodes": [], "edges": [],
    "node_preds_ordered": [],
    "edge_preds_ordered": []})
    
    json_path = json_dir / f"{gid}.json"
    if not json_path.exists():
        skipped.append((gid, "json_not_found"))
        continue

    try:
        dataset = GlycanGraphDataset([str(json_path)])
        loader  = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=False,
            collate_fn=collate_fn_for_tokengt, num_workers=0
        )

        for batch in loader:
            if batch is None:
                skipped.append((gid, "collate_none"))
                continue

            node_data = batch["node_data"]      # LongTensor
            edge_data = batch["edge_data"]      # LongTensor

            if node_data.numel() == 0 and edge_data.numel() == 0:
                skipped.append((gid, "empty_graph"))
                continue

            node_valid = (node_data != NODE_PAD_ID)
            edge_valid = (edge_data != EDGE_PAD_ID)
            q_node_ids, q_edge_ids = load_q_id_sets(json_path)
            node_q_mask = torch.zeros_like(node_data, dtype=torch.bool)
            edge_q_mask = torch.zeros_like(edge_data, dtype=torch.bool)

            for nid in q_node_ids:
                if 0 <= nid < node_q_mask.numel():
                    node_q_mask[nid] = True
            for eid in q_edge_ids:
                if 0 <= eid < edge_q_mask.numel():
                    edge_q_mask[eid] = True

            node_q_mask &= (node_data != NODE_PAD_ID)
            edge_q_mask &= (edge_data != EDGE_PAD_ID)

            if not (node_q_mask.any() or edge_q_mask.any()):
                continue

            masked_batch = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
            masked_batch["node_data"][node_q_mask] = NODE_MASK_ID
            masked_batch["edge_data"][edge_q_mask] = EDGE_MASK_ID
            for k, v in masked_batch.items():
                if isinstance(v, torch.Tensor):
                    masked_batch[k] = v.to(device, non_blocking=True)

            with torch.no_grad():
                logits_dict, _ = model(masked_batch)
            node_logits_all, edge_logits = logits_dict["node"], logits_dict["edge"]

            node_logits_flat = slice_node_logits_like(node_logits_all, node_data, batch)  # [num_nodes, Vnode]
            node_sel_idx = node_q_mask.view(-1).nonzero(as_tuple=False).view(-1)          # [Mnode]
            if node_sel_idx.numel() > 0:
                nl = node_logits_flat[node_sel_idx.to(node_logits_flat.device)]
                nl[:, NODE_PAD_ID] = -1e9
                nl[:, NODE_MASK_ID] = -1e9
                nprob = softmax_rows(nl)
                n_top1_prob, n_top1_id = torch.max(nprob, dim=1)

                n_orig_id = node_data.view(-1)[node_sel_idx]
                n_orig_tok = id2tok(monomer_vocab, n_orig_id)
                n_pred_id = n_top1_id.cpu()
                n_pred_tok = id2tok(monomer_vocab, n_pred_id)

                for pos, orig_id_i, orig_tok_i, pred_id_i, pred_tok_i, prob_i in zip(
                    node_sel_idx.cpu().tolist(),
                    n_orig_id.cpu().tolist(),
                    n_orig_tok,
                    n_pred_id.tolist(),
                    n_pred_tok,
                    n_top1_prob.cpu().tolist()
                ):
                    rows_long.append({
                        "glytoucan_id": gid,
                        "type": "node",
                        "position": int(pos),
                        "orig_id": int(orig_id_i),
                        "orig_token": orig_tok_i,
                        "pred_id": int(pred_id_i),
                        "pred_token": pred_tok_i,
                        "pred_prob": float(prob_i),
                    })
                    entry["node_preds_ordered"].append(pred_tok_i)

                for j, pos in enumerate(node_sel_idx.cpu().tolist()):
                    entry["nodes"].append({
                        "position": int(pos),
                        "orig_id": int(n_orig_id[j].item()),
                        "orig_token": n_orig_tok[j],
                        "pred_id": int(n_pred_id[j].item()),
                        "pred_token": n_pred_tok[j],
                        "pred_prob": float(n_top1_prob[j].item()),
                    })

            # ===== edge =====
            edge_sel_idx = edge_q_mask.view(-1).nonzero(as_tuple=False).view(-1)
            if edge_sel_idx.numel() > 0:
                el = edge_logits[edge_sel_idx.to(edge_logits.device)]
                el[:, EDGE_PAD_ID] = -1e9
                el[:, EDGE_MASK_ID] = -1e9
                eprob = softmax_rows(el)
                e_top1_prob, e_top1_id = torch.max(eprob, dim=1)

                e_orig_id = edge_data.view(-1)[edge_sel_idx]
                e_orig_tok = id2tok(linkage_vocab, e_orig_id)
                e_pred_id = e_top1_id.cpu()
                e_pred_tok = id2tok(linkage_vocab, e_pred_id)

                for pos, orig_id_i, orig_tok_i, pred_id_i, pred_tok_i, prob_i in zip(
                    edge_sel_idx.cpu().tolist(),
                    e_orig_id.cpu().tolist(),
                    e_orig_tok,
                    e_pred_id.tolist(),
                    e_pred_tok,
                    e_top1_prob.cpu().tolist()
                ):
                    rows_long.append({
                        "glytoucan_id": gid,
                        "type": "edge",
                        "position": int(pos),
                        "orig_id": int(orig_id_i),
                        "orig_token": orig_tok_i,
                        "pred_id": int(pred_id_i),
                        "pred_token": pred_tok_i,
                        "pred_prob": float(prob_i),
                    })
                    entry["edge_preds_ordered"].append(pred_tok_i)

                for j, pos in enumerate(edge_sel_idx.cpu().tolist()):
                    entry["edges"].append({
                        "position": int(pos),
                        "orig_id": int(e_orig_id[j].item()),
                        "orig_token": e_orig_tok[j],
                        "pred_id": int(e_pred_id[j].item()),
                        "pred_token": e_pred_tok[j],
                        "pred_prob": float(e_top1_prob[j].item()),
                        })

    except Exception as e:
        skipped.append((gid, f"exception:{type(e).__name__}"))
        continue

# ====== save outputs ======
pd.DataFrame(rows_long).to_csv(OUT_LONG_CSV, index=False)
print(f"✅ Saved per-position predictions: {OUT_LONG_CSV}")

rows_by = []
for gid, d in by_gid.items():
    rows_by.append({
        "GlyTouCan ID": gid,
        "pred_nodes_json": json.dumps(d.get("nodes", []), ensure_ascii=False),
        "pred_edges_json": json.dumps(d.get("edges", []), ensure_ascii=False),
    })

pred_df = pd.DataFrame(rows_by)

if pred_df.empty:
    pred_df = pd.DataFrame(columns=["GlyTouCan ID", "pred_nodes_json", "pred_edges_json"])

pred_df.to_csv(OUT_BY_GID, index=False)
print(f"✅ Saved by-glycan predictions: {OUT_BY_GID}")

src_df2 = src_df.copy()
src_df2["GlyTouCan ID"] = src_df2["GlyTouCan ID"].astype(str)
if not pred_df.empty:
    pred_df["GlyTouCan ID"] = pred_df["GlyTouCan ID"].astype(str)

merged = src_df2.merge(pred_df, on="GlyTouCan ID", how="left")
merged.to_csv(OUT_MERGED, index=False)
print(f"✅ Saved merged CSV: {OUT_MERGED}")

rows_simple = []
for _, r in src_df.iterrows():
    gid = str(r["GlyTouCan ID"])
    orig = r.get("IUPAC Condensed", "")

    if gid in by_gid:
        node_list = list(by_gid[gid].get("node_preds_ordered", []))
        edge_list = list(by_gid[gid].get("edge_preds_ordered", []))
    else:
        node_list, edge_list = [], []

    pred_iupac = replace_iupac_sequential(orig, node_list, edge_list)
    rows_simple.append({"GlyTouCan ID": gid, "IUPAC Condensed": orig, "IUPAC_Predicted": pred_iupac})

OUT_SIMPLE = os.path.join(OUT_DIR, "ambiguous_predictions_simple.csv")
pd.DataFrame(rows_simple).to_csv(OUT_SIMPLE, index=False)
print(f"✅ Saved simple 3-col CSV: {OUT_SIMPLE}")

if skipped:
    skip_path = os.path.join(OUT_DIR, "skipped_ids.tsv")
    pd.DataFrame(skipped, columns=["GlyTouCan ID", "reason"]).to_csv(skip_path, sep="\t", index=False)
    print(f"ℹ️ Skipped {len(skipped)} glycans. Log: {skip_path}")
