# RLDAL (ACDC)

Self-contained RLDAL active-learning pipeline for ACDC 2D slices. Everything needed to run (models, data loaders, utils) lives under this folder, so it can be pushed to GitHub as-is.

## Layout
- `run_rldal.py` — CLI entry; flags mirror the original `utils/parser.py`.
- `config.py` — defaults copied from `utils/parser.py` to stay aligned with `run.py`.
- `trainer.py` — pretrain + active-learning loop (loss-based reward + DQN selection) using the original utilities.
- `data_acdc.py` — wraps `data.data_utils.get_data` for ACDC.
- `data/`, `models/`, `utils/` — copied from the original project so this folder runs standalone.
- `scripts/prepare_acdc_2d.py`, `scripts/make_acdc_al_splits.py` — data conversion and split generation.

## Install
```bash
pip install -r RLDAL/requirements.txt
```

## Prepare data
1) Convert MRI H5 volumes to 2D slices (keeps slices with foreground):
```bash
python RLDAL/scripts/prepare_acdc_2d.py \
  --dataset-root ACDC-dataset/ACDC \
  --output-root  ACDC-dataset/ACDC/2d
```

2) Create AL splits (already shipped as `RLDAL/data/acdc_al_splits.npy`, regenerate if needed):
```bash
python RLDAL/scripts/make_acdc_al_splits.py \
  --train-dir ACDC-dataset/ACDC/2d/image/train \
  --output RLDAL/data/acdc_al_splits.npy \
  --seed 42
```

Expected tree under `--data-path` (default `./scratch/`):
```
ACDC-dataset/ACDC/2d/
  image/{train,val,test}/*.png
  mask/{train,val,test}/*.png
```

## Train (defaults = original parser)
```bash
python RLDAL/run_rldal.py \
  --data-path ./scratch/ \
  --ckpt-path ./scratch/ckpt_seg \
  --exp-name rldal_acdc \
  --dataset ACDC \
  --budget-labels 100 \
  --num-each-iter 1 \
  --rl-pool 50 \
  --rl-episodes 50 \
  --train-batch-size 8 \
  --val-batch-size 4
```

Process overview:
- Pretrain: supervised pretraining on the pretrain split (`train` in `trainer.py`), save `pre_train.pth`.
- RL loop: per episode, build candidate pool (`num_each_iter * rl_pool`), `compute_state` extracts region features, `select_action` picks regions (DQN/entropy/random), `add_labeled_images` expands the labeled set and rebuilds the dataloader.
- Segmentation training: run `train` on the updated labeled set; reward = batch loss minus its mean.
- DQN update: `optimize_model_conv` updates the policy net with replay; target net updated periodically.
- After budget, you can switch to full-supervised/EMA fine-tune (see `train_final_ema` in `utils/final_utils.py`).

Key knobs (defaults):
- Optim: `--lr 0.001 --momentum 0.95 --weight-decay 1e-4 --gamma 0.998`
- DQN: `--lr-dqn 0.0001 --dqn-bs 16 --dqn-gamma 0.9 --dqn-action-select softmax --dqn-temp 0.7 --rl-buffer 3200`
- Regions/input: `--region-size 64 64 --input-size 256 256 --rl-pool 50`
- Misc: `--checkpointer --load-weights --load-opt` (optional; same semantics as the original `run.py`)

Checkpoints/logs: written to `ckpt_path/exp_name/` (pretrain weights, DQN checkpoints, reward/loss logs).

## Test / resume
- Resume: `--load-weights --load-opt --exp-name-toload <exp>` (seg) and `--exp-name-toload-rl <exp>` (policy).
- Validation-only: `--test`.

## Notes
- Behavioural parity: reward, action selection, and defaults follow the original `run.py`/`utils/parser.py`.
- Default dataset is ACDC; to switch, change `--dataset` and ensure a loader exists under `RLDAL/data/`.
- This folder is standalone: pushing only `RLDAL/` to GitHub keeps the pipeline runnable.
