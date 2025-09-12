# step2: creating monomer and linkage vocabulary

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd
import re
from glycowork.motif.graph import glycan_to_nxGraph

import sys
sys.path.append("/share3/kitani/glycoGT")
from tokenizer.monomer_vocab import MonomerVocab
from tokenizer.linkage_vocab import LinkageVocab

IUPAC_COL = "IUPAC Condensed"

# --- linkage detection helpers (ADD) ---
_PAT1 = re.compile(r"^[abAB][0-9?/,]+-[0-9?/,]+[abAB]?$")                # b2-1a, a1-4
_PAT2 = re.compile(r"^(?:[abAB])?[0-9?/,]+-[A-Za-z]{1,4}-[0-9?/,]+(?:[abAB])?$")  # 1-P-3, a1-SH-6, b1-N-4
_PAT3 = re.compile(r"^[0-9?]+(?:[,/][0-9?]+)*-[0-9?]+(?:[,/][0-9?]+)*$")  # 2-1, 2,3-6, 2-1,3
_PAT4 = re.compile(r"^(?:[abAB])?[0-9?/,]+-$")                           # a1-, b2-, 1-
_PAT_TRASH = re.compile(r"^[\d?.,/\-\s]+$")                              

def _normalize_link_label(s: str) -> str:
    return re.sub(r"-([a-zA-Z]{1,4})-", lambda m: f"-{m.group(1).upper()}-", s.strip())

def is_linkage(label: str) -> bool:
    if not label or not str(label).strip():
        return True
    s = _normalize_link_label(str(label))
    return bool(_PAT1.fullmatch(s) or _PAT2.fullmatch(s) or _PAT3.fullmatch(s) or _PAT4.fullmatch(s) or _PAT_TRASH.fullmatch(s))

def is_not_monomer(label: str) -> bool:
    return is_linkage(label)

def extract_tokens(csv_file: Path) -> tuple[Set[str], Set[str]]:
    df = pd.read_csv(csv_file, sep=None, engine="python", dtype=str)
    monos: Set[str] = set()
    links: Set[str] = set()
    n_err = 0

    for s in df[IUPAC_COL].dropna():
        try:
            g = glycan_to_nxGraph(s)
        except Exception as e:
            n_err += 1
            continue

        # node: string_labels
        for _, d in g.nodes(data=True):
            label = d.get("string_labels")
            if label is not None:
                lab = _normalize_link_label(label)
                if is_not_monomer(lab):
                    links.add(lab)
                else:
                    monos.add(lab)

        # edge: 'label'
        for _, _, d in g.edges(data=True):
            l = d.get("label")
            if l is not None:
                links.add(_normalize_link_label(l))

    if n_err:
        print(f"[Info] {n_err} parse errors were skipped")
    return monos, links

def main():
    parser = argparse.ArgumentParser(description="Build monomer/linkage vocab json from clean glycan CSV (glycowork ver, exclude linkage-like labels)")
    parser.add_argument('--csv', required=True, help='clean CSV path')
    parser.add_argument('--out_dir', required=True, help='output directory')
    args = parser.parse_args()

    src = Path(args.csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    monos, links = extract_tokens(src)

    MonomerVocab.build(monos).save(out_dir / "monomer.json")
    LinkageVocab.build(links).save(out_dir / "linkage.json")

    print(
        f"Monomers: {len(monos):,}\n"
        f"Linkages: {len(links):,}\n"
        f"Saved to: {out_dir}"
    )

if __name__ == "__main__":
    main()