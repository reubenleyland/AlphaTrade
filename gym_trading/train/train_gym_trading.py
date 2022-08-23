# %% ==========================================================================
# clear warnings
from email import policy
import warnings
import gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.vec_env import DummyVecEnv
from gym_trading.envs.base_environment import BaseEnv
from gym_trading.data.data_pipeline import ExternalData
warnings.filterwarnings("ignore")
Flow = ExternalData.get_sample_order_book_data()
num_cpu = 10 
# env = gym.make("GymTrading-v1",Flow = Flow) ## TODO
env = DummyVecEnv([lambda: gym.make("GymTrading-v1",Flow = Flow)] * num_cpu)
check_env(env)

model = PPO("MultiInputPolicy", env, verbose=1, tensorboard_log="./ppo_gymtrading_tensorboard8/")
# %time model.learn(total_timesteps=int(1e7), n_eval_episodes = int(1e5))
# model.learn(total_timesteps=int(3e6), n_eval_episodes = int(1e5))
model.learn(total_timesteps=int(1e10), tb_log_name="short_horizon")
model.save("gym_trading-v1") 










