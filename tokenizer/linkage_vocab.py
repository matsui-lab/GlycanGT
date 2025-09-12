# linkage_vocab.py
"""Vocabulary helper for glycosidic linkage types."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Iterable
import json

_PAD = "[PAD]"
_UNK = "[UNK]"


class LinkageVocab:
    def __init__(self, idx2tok: List[str]):
        if idx2tok[0] != _PAD or idx2tok[1] != _UNK:
            raise ValueError("idx2tok must start with '[PAD]', '[UNK]'")
        self._idx2tok = idx2tok
        self._tok2idx: Dict[str, int] = {t: i for i, t in enumerate(idx2tok)}

    # ------------------------------------------------------------------
    @classmethod
    def build(cls, tokens: Iterable[str]) -> "LinkageVocab":
        uniq = sorted(set(tokens))
        idx2tok = [_PAD, _UNK] + uniq
        return cls(idx2tok)

    @classmethod
    def load(cls, path: str | Path) -> "LinkageVocab":
        path = Path(path)
        with path.open() as f:
            idx2tok = json.load(f)
        return cls(idx2tok)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self._idx2tok, f, indent=2)

    # ------------------------------------------------------------------
    def encode(self, token: str) -> int:
        return self._tok2idx.get(token, 1)

    def decode(self, idx: int) -> str:
        if idx < 0 or idx >= len(self._idx2tok):
            return _UNK
        return self._idx2tok[idx]

    def __len__(self):
        return len(self._idx2tok)
