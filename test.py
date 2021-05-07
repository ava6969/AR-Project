import cv2
import numpy as np
import torch

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.algo import gail
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.utils import get_render_func, get_vec_normalize
from evaluation import evaluate

from envs import basic_env

actor_critic, obs_rms  = torch.load("/home/dewe/pytorch-a2c-ppo-acktr-gail-master/"
                                    "trained_models/a2c/pos.pt", map_location='cpu')

rendering =True
env = make_vec_envs(
    basic_env.BasicFlatDiscreteEnv,
    250,
    1,
    None,
    None,
    device='cpu',
    allow_early_resets=False,
    gripper_type='RobotiqThreeFingerGripper',
    # gripper_type='RobotiqThreeFingerDexterousGripper',
    horizon=500,
    robot='Panda',
    controller='JOINT_POSITION',
    has_renderer=rendering, use_camera=False)

# Get a render function
render_func = get_render_func(env)


vec_norm = get_vec_normalize(env)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.obs_rms = obs_rms

recurrent_hidden_states = torch.zeros(1, actor_critic.recurrent_hidden_state_size)
masks = torch.zeros(1, 1)

episode = 10
if rendering and  render_func is not None:
    render_func()

for i in range(episode):

    obs = env.reset()
    done = False
    rewards = 0
    while not done:
        with torch.no_grad():
            value, action, _, recurrent_hidden_states = actor_critic.act(
                obs, recurrent_hidden_states, masks, deterministic=True)

        # Obser reward and next obs
        obs, reward, done, _ = env.step(action)

        masks.fill_(0.0 if done else 1.0)
        rewards += reward
        if rendering and render_func is not None:
            render_func()

        else:
            px = obs.view(-1)[-84*84*3:].view(84, 84, 3).data.cpu().numpy() *255
            cv2.imshow('Img', px)
            cv2.waitKey(1)
    print(rewards)