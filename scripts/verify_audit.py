"""Reproduce the dataset audit and the locked split on CPU, and check the SHA-256 lock.

Runs notebook sections 2-4 (audit, canonical dataset, splits) exactly as written in
notebooks/AGA_Ordinal_Grading.ipynb, with no TensorFlow and no GPU. Takes about a minute.

    python scripts/verify_audit.py                       # data in data/raw, outputs to data/splits
    python scripts/verify_audit.py --data data/raw --out results/audit

Exit code 0 if the split manifest hashes to the value published in the paper.
"""
import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

EXPECTED_LOCK = "dc94878c954130dc3aa7140dae1043de7f5660f27d41f910b7ec846a3f51e193"
REPO = Path(__file__).resolve().parent.parent
FIRST_SECTION, STOP_SECTION = "## 2 · Data audit", "## 5 · Input pipeline"


def notebook_cells(nb_path):
    """Code cells from the start of section 2 up to (not including) section 5."""
    cells = json.load(open(nb_path, encoding="utf-8"))["cells"]
    out, inside = [], False
    for c in cells:
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            if src.startswith(FIRST_SECTION): inside = True
            if src.startswith(STOP_SECTION): break
        elif inside:
            out.append(src)
    return out


def plotting(fig_dir):
    """Real matplotlib (Agg) if installed, otherwise a no-op stand-in."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        def savefig(name):
            plt.savefig(fig_dir / f"{name}.png", dpi=150, bbox_inches="tight"); plt.close("all")
    except ImportError:
        from unittest.mock import MagicMock
        plt = MagicMock()
        plt.subplots.side_effect = lambda *a, **k: (MagicMock(), MagicMock())
        pd.DataFrame.plot = property(lambda self: MagicMock())

        def savefig(name): pass
    return plt, savefig


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=REPO / "data" / "raw", type=Path, help="unzipped Roboflow export")
    ap.add_argument("--out", default=REPO / "data" / "splits", type=Path, help="where manifests are written")
    ap.add_argument("--notebook", default=REPO / "notebooks" / "AGA_Ordinal_Grading.ipynb", type=Path)
    args = ap.parse_args()

    if not (args.data / "train" / "_classes.csv").exists():
        sys.exit(f"dataset not found under {args.data} -- see data/README.md")
    (args.out / "figures").mkdir(parents=True, exist_ok=True)
    plt, savefig = plotting(args.out / "figures")

    # the subset of the notebook's CFG that sections 2-4 read
    cfg = SimpleNamespace(SEED=1337, PHASH_THRESHOLD=5, EXCLUDE_AMBIGUOUS=True, IMG_SIZE=224,
                          N_CLASSES=7, ONE_IMAGE_PER_GROUP=True, TEST_FRACTION=0.20,
                          N_FOLDS=5, N_REPEATS=1, OUT_DIR=str(args.out))
    random.seed(cfg.SEED); np.random.seed(cfg.SEED)
    ns = dict(CFG=cfg, np=np, pd=pd, plt=plt, os=os, random=random, Path=Path,
              DATA_ROOT=args.data.resolve(), savefig=savefig, FIGURES={},
              STAGE_COLORS=["#2a9d8f", "#57b8a0", "#e9c46a", "#f4a261", "#ee8959", "#e76f51", "#bc4749"],
              display=lambda x: print(x.to_string() if hasattr(x, "to_string") else x))
    for src in notebook_cells(args.notebook):
        exec(src, ns)

    # the audit manifest carries absolute local paths; keep it portable
    audit_csv = args.out / "audit_manifest.csv"
    pd.read_csv(audit_csv).drop(columns=["path", "_is_original"], errors="ignore") \
      .to_csv(audit_csv, index=False, lineterminator="\n")

    lock = hashlib.sha256((args.out / "split_manifest.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    print(f"\nsplit manifest SHA-256 : {lock}\npublished in the paper  : {EXPECTED_LOCK}")
    if lock != EXPECTED_LOCK:
        sys.exit("MISMATCH -- the locked test set was not reproduced")
    print("LOCK VERIFIED -- identical locked test set")


if __name__ == "__main__":
    main()
