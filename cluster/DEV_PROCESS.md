# Development process on the DALIA cluster

## Cluster specs

**Compute nodes:** 18 × `dalianvl` nodes  
**GPU:** 4 × NVIDIA GB200 per node — **185 GB VRAM each** (740 GB total per node)  
**CPU:** 144 ARM Neoverse-V2 cores per node  
**Partition:** `defq` — 2-day time limit, no QOS restriction  
**Storage:** 3 GB home (code only) · 10 TB work (checkpoints, datasets, logs)

Fair-share allocation: **36 CPUs and 1 GPU per job** is a reasonable default, leaving room for co-tenants on the same node. You can go up to 120 CPUs if the node is free, but prefer staying polite unless you need it.

---

## Dev philosophy

VRAM is abundant (185 GB per GPU). The bottleneck is almost never the model itself — it is data throughput, CPU generation speed, or logging overhead. The goal is to keep the GPU at maximum utilisation throughout training by removing everything that makes it wait.

Concrete checklist before launching a long run:

1. **Disable in-training evals.** Plan evals are expensive (~20 min each on two_rooms). Run them as a separate job after training.
2. **Use AMP.** `bfloat16` halves memory bandwidth and speeds up matmuls on GB200 with no loss in training quality.
3. **Compile the model.** `torch.compile` reduces Python overhead on the training loop. `mode=reduce-overhead` enables cudagraphs for an additional ~20% speedup.
4. **Saturate the DataLoader.** Use enough workers and a prefetch queue so the GPU never idles waiting for the next batch.
5. **Use pinned memory + non-blocking transfers.** With `pin_mem: true`, CPU→GPU copies overlap with the previous forward pass.

---

## Training configs

Two configs are provided in `examples/ac_video_jepa/cfgs/train/two_rooms/`:

| Config | Purpose |
|--------|---------|
| `train.yaml` | Reference config — in-training evals on, verbose logging, 1 checkpoint per epoch. Use to debug or validate a new idea quickly. |
| `train_ex_reduction_of_walltime.yaml` | Drop-in for long runs — same model and optimizer, all overhead stripped. See below. |

### `train_ex_reduction_of_walltime.yaml` — what it disables

| Setting | `train.yaml` | `train_ex_reduction_of_walltime.yaml` | Why |
|---------|-------------|--------------------------------------|-----|
| `enable_plan_eval` | `true` | **`false`** | Saves ~7h across 12 epochs |
| `log_every` | 10 | **100** | 10× less W&B traffic |
| `save_every_n_epochs` | 1 | **100** | No intermediate checkpoint I/O |
| `tqdm_silent` | `false` | **`true`** | No per-step stdout |
| `num_workers` | 16 | **24** | More DataLoader prefetch workers |
| `prefetch_factor` | — | **4** | Deeper prefetch queue |
| `compile_mode` | — | **`reduce-overhead`** | cudagraphs, ~20% extra speedup |

All model hyperparameters (architecture, regularizer, optimizer, scheduler) are identical to `train.yaml`. It is safe to train with `train_ex_reduction_of_walltime.yaml` and evaluate the resulting checkpoint with `eval.yaml`.

---

## Typical workflow

All SLURM submission goes through the unified launcher `examples/launch_sbatch.py`
(submitit) — it snapshots the code, handles seeds/sweeps/wandb, and accepts cluster
overrides (`--partition/--account/--cpus-per-task/--time-min/--gpus-per-node`).

```
1. Prototype locally (small config, few steps, no SLURM)
   python -m examples.ac_video_jepa.main --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml --optim.epochs 1

2. Smoke-test on the cluster (single job, check loss decreases + GPU utilisation)
   python -m examples.launch_sbatch --example ac_video_jepa --single

3. Full training run (3-seed sweep, auto-named)
   python -m examples.launch_sbatch --example ac_video_jepa \
       --fname examples/ac_video_jepa/cfgs/train/two_rooms/train_ex_reduction_of_walltime.yaml

4. Eval — re-run on the produced checkpoint, eval-only (separate job).
   Pass the TRAIN config (it carries the model/optim sections + points eval_cfg_path at
   eval.yaml); eval.yaml is a fragment consumed via cfg.eval, not a top-level --fname.
   python -m examples.launch_sbatch --example ac_video_jepa --single \
       --fname examples/ac_video_jepa/cfgs/train/two_rooms/train.yaml \
       --meta.eval_only_mode True --meta.model_folder <ckpt_dir_from_step_3>
```

Check GPU utilisation mid-run with:
```bash
gpus        # per-node GPU allocation
log -f      # tail the running job's stdout
```

A healthy run shows the GPU busy >80% of the time. If it is lower, the bottleneck is the data pipeline — see `examples/ac_video_jepa/cfgs/train/two_rooms/data/` for faster pipeline options.
