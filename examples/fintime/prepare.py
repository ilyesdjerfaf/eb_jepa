"""One-time HF-parquet -> memmap conversion for FinTime (PROVIDED plumbing).

Decodes each split of ``thesven/fintime-decoder-dataset`` into a contiguous float32
memmap ``{split}_X.npy`` [N, C, T] + ``{split}_meta.npz`` (targets/metadata), so the
dataloader is fast. Run once on a node with pyarrow:

  python -m examples.fintime.prepare <hf_dir> <out_dir>
"""

import glob
import os
import sys

import numpy as np
import pyarrow.parquet as pq

SPLITS = {
    "train": "train-*.parquet",
    "validation": "validation-*.parquet",
    "test": "test-*.parquet",
}
META = [
    "target_next_return",
    "target_direction",
    "window_end_date",
    "ticker",
    "subcategory",
]


def convert(src, dst, split, pattern):
    files = sorted(glob.glob(os.path.join(src, "data", pattern)))
    if not files:
        print(f"[prep] {split}: no shards")
        return
    n = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    a0 = np.asarray(
        pq.ParquetFile(files[0])
        .read_row_group(0, columns=["series"])
        .to_pylist()[0]["series"]
    )
    C, T = a0.shape
    X = np.lib.format.open_memmap(
        os.path.join(dst, f"{split}_X.npy"),
        mode="w+",
        dtype=np.float32,
        shape=(n, C, T),
    )
    meta = {k: [] for k in META}
    i = 0
    for f in files:
        pf = pq.ParquetFile(f)
        for rg in range(pf.num_row_groups):
            d = pf.read_row_group(rg).to_pydict()
            for k in range(len(d["series"])):
                X[i] = np.asarray(d["series"][k], dtype=np.float32)
                i += 1
            for m in META:
                meta[m].extend(d[m])
    X.flush()
    np.savez(
        os.path.join(dst, f"{split}_meta.npz"),
        y_return=np.asarray(meta["target_next_return"], np.float32),
        y_direction=np.asarray(meta["target_direction"], np.int64),
        window_end_date=np.asarray(meta["window_end_date"]),
        ticker=np.asarray(meta["ticker"]),
        subcategory=np.asarray(meta["subcategory"]),
    )
    print(f"[prep] {split}: wrote [{n},{C},{T}]")


def main():
    src = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "/lustre/work/pdl17890/udl806719/datasets/Finance/fintime-decoder-dataset"
    )
    dst = (
        sys.argv[2]
        if len(sys.argv) > 2
        else "/lustre/work/pdl17890/udl806719/datasets/Finance/fintime_prep"
    )
    os.makedirs(dst, exist_ok=True)
    for s, p in SPLITS.items():
        convert(src, dst, s, p)


if __name__ == "__main__":
    main()
