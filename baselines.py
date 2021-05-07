import stable_baselines3
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import StopTrainingOnRewardThreshold, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_atari_env
from stable_baselines3.common.vec_env import VecFrameStack

env = make_atari_env('PongNoFrameskip-v4', n_envs=32, seed=0)
# Stack 4 frames
env = VecFrameStack(env, n_stack=1)

# # Create environment
# env_id = 'CartPole-v1'
# eval_env = gym.make(env_id)
# env = make_vec_env(env_id, n_envs=16, seed=0)
# # Instantiate the agent
# model = A2C('MlpPolicy', env, verbose=1,seed=0)
model = A2C('CnnPolicy', env, verbose=1, seed=0)
# callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=475, verbose=1)
# eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)
# Train the agent
model.learn(total_timesteps=int(1e7))
model.save("a2c_pong")