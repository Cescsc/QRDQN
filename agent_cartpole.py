from stable_baselines3.common.env_util import make_atari_env
import gymnasium as gym
from modified_qr import QRDQN as modQRDQN
from qrdqn_tweaked import QRDQN
from stable_baselines3 import A2C
from torch.utils.tensorboard import SummaryWriter


writer = SummaryWriter()
env = gym.make("CartPole-v1")

def dqn():
    model = A2C("MlpPolicy", env, tensorboard_log="kth/kex/qrdqn/logs/")
    model.learn(total_timesteps=100_000)

def qr():
    policy_kwargs = dict(n_quantiles=50)
    model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="kth/kex/qrdqn/logs/")
    model.learn(total_timesteps=100_000, log_interval=4, tb_log_name="cartpole_qr")

    #model.save("kth/kex/qrdqn/qrdqn_cartpole")

def mod_qr():
    policy_kwargs = dict(n_quantiles=50)
    model = modQRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, tensorboard_log="kth/kex/qrdqn/logs/")
    model.learn(total_timesteps=100_000, log_interval=4, tb_log_name="cartpole_mqr")

mod_qr()