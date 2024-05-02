from stable_baselines3.common.env_util import make_atari_env
import gymnasium as gym
from qrdqn_tweaked import QRDQN
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()

env = make_atari_env("PongNoFrameskip-v4", seed=0)

model = QRDQN("CnnPolicy", env, tensorboard_log="kth/kex/qrdqn/logs/")
model.learn(total_timesteps=1_000_000)

model.save("kth/kex/qrdqn/qrdqn_pong")
