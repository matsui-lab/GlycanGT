# monomer_vocab.py
"""Monomer vocabulary helper.

Keeps a bidirectional mapping between canonical monomer (single residue)
identifiers and integer IDs.  Numbers start from 1; ID 0 is reserved for
`[PAD]/[UNK]` so that we can safely use 0 as mask value.

Usage
-----
```python
vocab = MonomerVocab.load("data/vocab/monomer.json")
idx  = vocab.encode("GalNAc-α")
name = vocab.decode(idx)
```
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterable
import json

_PAD = "[PAD]"
_UNK = "[UNK]"


class MonomerVocab:
    """A thin wrapper around two `dict`s."""

    def __init__(self, idx2tok: List[str]):
        # ensure 0:"[PAD]",1:"[UNK]"
        if idx2tok[0] != _PAD or idx2tok[1] != _UNK:
            raise ValueError("idx2tok must start with '[PAD]', '[UNK]'")
        self._idx2tok: List[str] = idx2tok
        self._tok2idx: Dict[str, int] = {t: i for i, t in enumerate(idx2tok)}

    # ------------------------------------------------------------------
    # factory -----------------------------------------------------------
    # ------------------------------------------------------------------
    @classmethod
    def build(cls, tokens: Iterable[str]) -> "MonomerVocab":
        uniq = sorted(set(tokens))
        idx2tok = [_PAD, _UNK] + uniq
        return cls(idx2tok)

    @classmethod
    def load(cls, path: str | Path) -> "MonomerVocab":
        path = Path(path)
        with path.open() as f:
            idx2tok = json.load(f)
        return cls(idx2tok)

    # ------------------------------------------------------------------
    # io ----------------------------------------------------------------
    # ------------------------------------------------------------------
    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._idx2tok, f, indent=2)

    # ------------------------------------------------------------------
    # mapping -----------------------------------------------------------
    # ------------------------------------------------------------------
    def encode(self, token: str) -> int:
        """Return int ID; `[UNK]` (1) if OOV."""
        return self._tok2idx.get(token, 1)

    def decode(self, idx: int) -> str:
        if idx < 0 or idx >= len(self._idx2tok):
            return _UNK
        return self._idx2tok[idx]

    # ------------------------------------------------------------------
    # utils -------------------------------------------------------------
    # ------------------------------------------------------------------
    def __len__(self) -> int:  # noqa: D401
        return len(self._idx2tok)
