# QR-DQN — personal experiments in Distributional RL

> Personal learning project — not an official Bachelor thesis.  
> Implementations and experiments done on the side to learn distributional reinforcement learning.

QR-DQN (Quantile Regression Deep Q-Network) is a distributional reinforcement learning algorithm that models the full return distribution instead of estimating a single expected Q-value. Distributional RL is a modern paradigm that often provides more stable learning and richer signals for exploration and risk-sensitive control.

This repository contains a compact implementation and experiments using QR-DQN. The long-term goal is to train agents on Atari 2600 games, but the code in this repo is currently exercised on the simpler CartPole environment for learning and debugging.

## Highlights
- Small, self-contained QR-DQN implementation for learning and evaluation.
- Training/evaluation scripts and basic logging/plots for debugging.
- Intended as a learning resource — straightforward and easy to read, not production-ready.

## What is Distributional RL (short)
Traditional RL methods estimate the expected return. Distributional RL models the distribution of returns (random variable over future discounted rewards). QR-DQN approximates that distribution by learning a set of quantiles and using quantile regression to update them. See Dabney et al., 2018 for the original paper: "Distributional Reinforcement Learning with Quantile Regression."

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
