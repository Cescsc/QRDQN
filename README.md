# QR-DQN — distributional RL experiments

> Personal project developed during my Bachelor's thesis work in Spring 2024.  
> This repository captures my experiments with distributional reinforcement learning while exploring QR-DQN in parallel with my thesis work.

QR-DQN (Quantile Regression Deep Q-Network)  is the distributional RL variant of DQN, that models the full return distribution instead of estimating a single expected Q-value. Distributional RL is particularly interesting because it tracks uncertainty in returns rather than only their mean, giving richer learning signals and often more stable training dynamics in complex decision tasks.

This repository contains a compact implementation and experiments using QR-DQN. The long-term goal is to train agents on Atari 2600 games, but the code in this repo is currently exercised on simpler environments as part of my learning and thesis-related exploration.

## Highlights
- Small, self-contained QR-DQN implementation for learning and evaluation.
- Training/evaluation scripts and basic logging/plots for debugging.
- Intended as a learning resource and thesis-adjacent experimental codebase — straightforward and easy to read, not production-ready.

## What is Distributional RL (short)
Traditional RL methods estimate the expected return. Distributional RL models the distribution of returns (random variable over future discounted rewards). QR-DQN approximates that distribution by learning a set of quantiles rather than one scalar estimate for each action.

## Structure (example)
- `train.py` — training loop (example name; adapt to your repo layout)
- `agents/` — agent implementations (QR-DQN core logic)
- `envs/` — wrappers and environment utilities
- `utils/` — logging, plotting, replay buffer, schedules
- `scripts/` — helper scripts for evaluation and plotting
- `requirements.txt` — Python dependencies (if present)

(Your repo might differ; adjust paths/filenames when running commands.)

## Requirements
- Python 3.8+
- PyTorch
- Gym (OpenAI Gym or Gymnasium)
- NumPy, Matplotlib (or Seaborn)
- Optional: TensorBoard for visualization

Install typical packages with:
```bash
pip install -r requirements.txt
# or manually
pip install torch gym numpy matplotlib seaborn tensorboard
