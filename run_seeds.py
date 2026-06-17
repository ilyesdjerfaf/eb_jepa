"""Launch multiple seeds in parallel on a single GPU (memory-shared)."""
import multiprocessing as mp
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
SEEDS = [1, 1000, 10000]
GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "0")


def run_seed(seed: int):
    os.chdir(REPO_ROOT)
    sys.path.insert(0, str(REPO_ROOT))
    # Each process inherits CUDA_VISIBLE_DEVICES — all share the same GPU
    from eb_jepa.training_utils import load_config
    from examples.ac_video_jepa.main import run

    cfg = load_config("examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml", {"meta.seed": seed})
    run(cfg=cfg)


if __name__ == "__main__":
    print(f"Launching {len(SEEDS)} seeds on GPU {GPU}: {SEEDS}")
    ctx = mp.get_context("spawn")
    procs = []
    for seed in SEEDS:
        p = ctx.Process(target=run_seed, args=(seed,), name=f"seed_{seed}")
        p.start()
        procs.append(p)
        print(f"  started seed={seed} pid={p.pid}")

    for p in procs:
        p.join()
        print(f"  {p.name} exited with code {p.exitcode}")
