# L40 Baseline Setup

This note adapts the lab documents to the current project so the baseline can run on the LadHPC server with Slurm.

## What The Lab Docs Mean For Us

From the lab manuals:

- you must connect through VPN first;
- SSH uses port `2222`;
- all serious compute jobs must be submitted through `Slurm`;
- the only partition is `compute`;
- code and datasets should live under `~/ssd` or `~/hdd`, not directly in the home directory;
- the server has `8 x NVIDIA L40`, Ubuntu `22.04`, and Miniconda available.

For this project, the simplest good workflow is:

1. keep the project in `~/ssd/da_final_project`;
2. create a Conda environment for PyTorch and the project dependencies;
3. submit baseline jobs with `sbatch`;
4. keep long-term outputs in `~/hdd` if needed after the run finishes.

## Files Added For The Cluster

The following project files are prepared for the lab server:

- `environment-l40.yml`
- `scripts/setup_l40_env.sh`
- `scripts/submit_baseline_l40.slurm`
- `scripts/sync_to_l40.sh`

## One-Time Setup On The Server

After VPN login and SSH login, sync the project to the server.

From your local machine:

```bash
cd /Volumes/UBag/Documents/Homeworks/DA/final_project
bash scripts/sync_to_l40.sh
```

Then on the server:

```bash
ssh -p 2222 <your_username>@172.16.43.21
cd ~/ssd/da_final_project
bash scripts/setup_l40_env.sh
```

If this is your first time using the server and Conda is not initialized in the shell yet, follow the lab’s first-login workflow first, then reconnect and rerun the setup script.

## Tonight's Quick Path

If your goal is to get one job training before sleep, do exactly this:

1. connect to the lab VPN;
2. sync the project:

```bash
cd /Volumes/UBag/Documents/Homeworks/DA/final_project
bash scripts/sync_to_l40.sh
```

3. SSH to the server and prepare the environment:

```bash
ssh -p 2222 <your_username>@172.16.43.21
cd ~/ssd/da_final_project
bash scripts/setup_l40_env.sh
```

4. submit the smoke test first:

```bash
MODEL=simvp \
RUN_NAME=simvp_smoke \
EPOCHS=2 \
TRAIN_SEQUENCES=256 \
VAL_SEQUENCES=64 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
sbatch scripts/submit_baseline_l40.slurm
```

5. confirm it entered the queue and watch the log:

```bash
squeue -u $USER
tail -f slurm_baseline_<job_id>.out
```

6. if the smoke test reaches epoch output cleanly, submit the overnight baseline:

```bash
MODEL=simvp \
RUN_NAME=simvp_baseline_v1 \
EPOCHS=50 \
TRAIN_SEQUENCES=4000 \
VAL_SEQUENCES=800 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
sbatch scripts/submit_baseline_l40.slurm
```

## Submit The Recognized Baseline

The default Slurm script is configured to run the `SimVP` baseline on a single L40:

```bash
cd ~/ssd/da_final_project
sbatch scripts/submit_baseline_l40.slurm
```

This uses defaults:

- `MODEL=simvp`
- `EPOCHS=50`
- `BATCH_SIZE=64`
- `TRAIN_SEQUENCES=4000`
- `VAL_SEQUENCES=800`
- `1 GPU`
- `8 CPUs`
- `32G` memory
- `NUM_WORKERS=8`

## Useful Overrides

You can change the run without editing the Slurm file by exporting variables before submission.

Example 1: quick smoke test

```bash
MODEL=simvp \
RUN_NAME=simvp_smoke \
EPOCHS=2 \
TRAIN_SEQUENCES=256 \
VAL_SEQUENCES=64 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
sbatch scripts/submit_baseline_l40.slurm
```

Example 2: run the open-loop baseline instead

```bash
MODEL=openloop \
RUN_NAME=openloop_baseline \
EPOCHS=20 \
sbatch scripts/submit_baseline_l40.slurm
```

## Monitor Jobs

Use the standard Slurm commands from the lab manual:

```bash
squeue -u $USER
scontrol show job <job_id>
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed
```

For GPU usage during the run:

```bash
nvidia-smi
```

The Slurm stdout/stderr logs will be written to:

- `slurm_baseline_<job_id>.out`
- `slurm_baseline_<job_id>.err`

The training outputs will be saved under:

- `outputs/<RUN_NAME>/`

## Recommended First Run

Before launching a long baseline, do this:

```bash
MODEL=simvp \
RUN_NAME=simvp_smoke \
EPOCHS=2 \
TRAIN_SEQUENCES=256 \
VAL_SEQUENCES=64 \
BATCH_SIZE=16 \
NUM_WORKERS=8 \
sbatch scripts/submit_baseline_l40.slurm
```

If that works, move to a more serious baseline:

```bash
MODEL=simvp \
RUN_NAME=simvp_baseline_v1 \
EPOCHS=50 \
TRAIN_SEQUENCES=4000 \
VAL_SEQUENCES=800 \
BATCH_SIZE=64 \
NUM_WORKERS=8 \
sbatch scripts/submit_baseline_l40.slurm
```

## Notes

- The cluster was not reachable from the current network while preparing this setup, so the remote login and submission steps were prepared from the lab documentation rather than executed live.
- The Slurm script currently uses one GPU because this is enough for the baseline and keeps queue pressure low.
- Once the DA layer is ready, we can either keep the same single-GPU pattern or add a separate multi-GPU script if needed.
