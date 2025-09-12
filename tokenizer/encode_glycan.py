from glycowork.motif.graph import glycan_to_nxGraph
import re

_PAT1 = re.compile(r"^[abAB][0-9?/,]+-[0-9?/,]+[abAB]?$")                # b2-1a, a1-4
_PAT2 = re.compile(r"^(?:[abAB])?[0-9?/,]+-[A-Za-z]{1,4}-[0-9?/,]+(?:[abAB])?$")  # 1-P-3, a1-SH-6, b1-N-4
_PAT3 = re.compile(r"^[0-9?]+(?:[,/][0-9?]+)*-[0-9?]+(?:[,/][0-9?]+)*$")  # 2-1, 2,3-6
_PAT4 = re.compile(r"^(?:[abAB])?[0-9?/,]+-$")                           # a1-, b2-, 1-
_PAT_TRASH = re.compile(r"^[\d?.,/\-\s]+$")                              

def _normalize_link_label(s: str) -> str:
    return re.sub(r"-([a-zA-Z]{1,4})-", lambda m: f"-{m.group(1).upper()}-", s.strip())

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

    monomer_nodes = {}
    for n, d in G.nodes(data=True):
        raw = d.get('string_labels', '')
        if not raw:
            continue
        lab = _normalize_link_label(raw)
        if is_monomer(lab):
            monomer_nodes[n] = lab

    node_id_map = {n: idx for idx, n in enumerate(sorted(monomer_nodes))}
    records = []

    for n, d in G.nodes(data=True):
        raw = d.get('string_labels', '')
        if not raw:
            continue
        lab = _normalize_link_label(raw)
        if is_linkage(lab):
            preds = list(G.predecessors(n))
            succs = list(G.successors(n))
            for src in preds:
                for tgt in succs:
                    if src in monomer_nodes and tgt in monomer_nodes:
                        records.append({
                            "in_node_id": node_id_map[src],
                            "in_node_name": monomer_nodes[src],
                            "edge_id": n,              # linkageノードID
                            "edge_name": lab,          
                            "out_node_id": node_id_map[tgt],
                            "out_node_name": monomer_nodes[tgt],
                            "in_node_vocab_id": mono_vocab.encode(monomer_nodes[src]),
                            "out_node_vocab_id": mono_vocab.encode(monomer_nodes[tgt]),
                            "edge_vocab_id": link_vocab.encode(lab),
                        })
    return records
