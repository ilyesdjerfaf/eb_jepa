# cluster/ — Utility scripts for the HTW DALIA cluster

Scripts for monitoring jobs, GPU usage, and logs. All read-only — they never submit or cancel anything.

Add to PATH for convenience (already done if you source `env.sh` with the snippet below):

```bash
export PATH="$EBJEPA_REPO/cluster:$PATH"   # $EBJEPA_REPO is set by env.sh
```

---

## Scripts

### `sq` — My jobs

```
sq
```

Shows your running and pending jobs, color-coded:
- **green** = RUNNING
- **yellow** = PENDING
- **red** = FAILED / CANCELLED / TIMEOUT

Prints a state summary at the bottom.

---

### `qall` — All jobs on defq

```
qall
```

Shows every job on the `defq` partition, sorted by user, with a per-user GPU/CPU/node summary at the bottom.

---

### `log` — View a job log

```
log [JOBID] [-f]
```

| Invocation | Behavior |
|------------|----------|
| `log` | Show the most recent job's stdout |
| `log 62022` | Show job 62022 stdout |
| `log -f` | Tail the most recent job's log (live) |
| `log 62022 -f` | Tail job 62022 live |

- Auto-discovers the log file path via `scontrol` (running jobs) or `sacct` (completed jobs).
- If stdout is empty, falls back to showing stderr automatically (useful for FAILED jobs).
- Works with both `slurm_test.sh` jobs and submitit-launched training jobs.

---

### `gpus` — GPU allocation per node

```
gpus
```

Shows a per-node table for all 18 `dalianvl` nodes:

| Column | Meaning |
|--------|---------|
| TOT | Total GPUs on the node (always 4 × GB200) |
| USED | Currently allocated |
| FREE | Available |
| STATE | SLURM node state (idle / mixed / allocated / drained) |
| CPU_LOAD | Current CPU load average |
| FREE_MEM | Free RAM in GB |

Color: green = fully free, yellow = partially used, red = fully allocated.

Shows total GPU counts (used / total / free) at the bottom.

---

### `users` — Resource usage per user

```
users
```

Shows GPU, CPU, and node counts per user on `defq`, sorted by GPU usage descending. Also shows job state counts at the bottom.

---

## Two architectures, two venvs (important)

This cluster mixes **two CPU architectures**, and compiled wheels (torch, …) are
**not portable across them** — so there are **two separate venvs**, one per arch:

| where | arch | venv | role |
|---|---|---|---|
| **login node** (where you type & submit) | `x86_64` | `venvs/eb_jepa_x86_64` | runs the **launcher** (`launch_sbatch.py`) — builds the config and **submits** the SLURM jobs (imports `submitit` + `eb_jepa`, which pulls in `torch`) |
| **compute nodes** (`dalianvl`, the GPUs) | `aarch64` | `venvs/eb_jepa_aarch64` | runs the **actual training/eval** (torch + CUDA `cu128`) |

`env.sh` derives the venv from `$(uname -m)`, so the same `source env.sh` picks the
right one on each node:
```bash
UV_PROJECT_ENVIRONMENT=$WORK/venvs/eb_jepa_$ARCH   # ARCH = x86_64 (login) | aarch64 (compute)
```

**`setup.sh` sets up BOTH** automatically:
1. on the login node it runs `uv sync` → builds `eb_jepa_x86_64` (torch-cpu, submitit, …);
2. then it submits a short SLURM job that runs `uv sync` on a compute node → builds
   `eb_jepa_aarch64` with **torch+cu128**.

> ⚠️ Wait for that second job (`eb_jepa_setup`) to finish before launching training —
> until it does, the `aarch64` venv is incomplete and jobs will fail to import torch.
> A login-only manual venv (skipping `setup.sh`) will also be incomplete: always set up
> via `setup.sh`.

`EBJEPA_COMPUTE_ARCH` (default `aarch64`) drives `COMPUTE_PYTHON` in `launch_sbatch.py`,
i.e. the Python binary submitit uses on compute nodes. Override before sourcing to target
a different arch:
```bash
export EBJEPA_COMPUTE_ARCH=x86_64 && source env.sh
```

---

## W&B toggle

```bash
# Disable W&B globally (before sourcing env.sh)
export WANDB_DISABLED=true && source env.sh

# Re-enable
export WANDB_DISABLED=false && source env.sh
```

> ⚠️ The training configs default to `logging.log_wandb: true`. A shell env var does
> **not** propagate through submitit to the compute job, so without W&B credentials the
> job crashes (`You must call wandb.init() before wandb.log()`). With the launcher,
> disable it **via the config override** (lowercase `false` so it parses as a bool):
> ```bash
> python -m examples.launch_sbatch --example ac_video_jepa --single --logging.log_wandb false
> ```

Or per-run via config override:
```bash
python -m examples.launch_sbatch --example ac_video_jepa --single  # uses train.yaml default (log_wandb: true)
```
