# step3-2: IUPAC -> Graph dict
# encode_glycan_bulk.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import pandas as pd

import sys
sys.path.append("/path/to/glycoGT")
from tokenizer.monomer_vocab import MonomerVocab
from tokenizer.linkage_vocab import LinkageVocab
from tokenizer.encode_glycan import iupac_to_graph_triples

IUPAC_COL = "IUPAC Condensed"
ID_COL = "GlyTouCan ID"

def main():
    ap = argparse.ArgumentParser(description="Encode IUPAC→TokenGT triple dict (bulk)")
    ap.add_argument("--csv", required=True, help="clean csv path")
    ap.add_argument("--mono_vocab", required=True)
    ap.add_argument("--link_vocab", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    mono_vocab = MonomerVocab.load(args.mono_vocab)
    link_vocab = LinkageVocab.load(args.link_vocab)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv, dtype=str)

    manifest: List[str] = []
    for i, row in df.iterrows():
        gid = row[ID_COL]
        iupac = row[IUPAC_COL]
        try:
            triples = iupac_to_graph_triples(iupac, mono_vocab, link_vocab)
        except Exception as e:
            print(f"[Warn] skip {gid}: {e}")
            continue
        out_path = out_dir / f"{gid}.json"
        with out_path.open("w") as f:
            json.dump(triples, f, indent=2)
        manifest.append(str(out_path))
        if (i + 1) % 1000 == 0:
            print(f"  processed {i+1}/{len(df)}")

    (out_dir / "manifest.txt").write_text("\n".join(manifest))
    print(f"Finished.  files={len(manifest)} manifest={out_dir/'manifest.txt'}")

if __name__ == "__main__":
    main()
