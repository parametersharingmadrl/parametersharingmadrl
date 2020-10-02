from sisl_games.pursuit.pursuit import env as custom_env
import ray
from ray.tune.registry import register_trainable, register_env
from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.agents.dqn import ApexTrainer
from ray.rllib.agents.ppo import PPOTrainer
import os
import pickle
import numpy as np
from ray.rllib.models import ModelCatalog

# 2 cases: APEX_DQN, RAINBOW_DQN or everything else
MLPv2methods = ["APEX_DQN", "RAINBOW_DQN"]
METHOD = "RAINBOW_DQN"  # APEX_DQN # ""

if METHOD in MLPv2methods:
    from parameterSharingPursuit import MLPModelV2
    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
else:
    from parameterSharingPursuit import MLPModel
    ModelCatalog.register_custom_model("MLPModel", MLPModel)

env_name = "pursuit"
# path should end with checkpoint-<> data file
# checkpoint_path = "./ray_results/pursuit/APEX_DQN/checkpoint_1000/checkpoint-1000"
# Trainer = ApexTrainer
checkpoint_path = "./ray_results/pursuit/RAINBOW_DQN/checkpoint_6630/checkpoint-6630"
Trainer = DQNTrainer

# TODO: see ray/rllib/rollout.py -- `run` method for checkpoint restoring

# register env -- For some reason, ray is unable to use already registered env in config
def env_creator(args):
    return custom_env()

env = env_creator(1)
register_env(env_name, env_creator)

# get the config file - params.pkl
config_path = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_path, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

ray.init()

RLAgent = Trainer(env=env_name, config=config)
RLAgent.restore(checkpoint_path)

# init obs, action, reward
observations = env.reset()
rewards, action_dict = {}, {}
for agent_id in env.agent_ids:
    assert isinstance(agent_id, int), "Error: agent_ids are not ints."
    # action_dict = dict(zip(env.agent_ids, [np.array([0,1,0]) for _ in range(len(env.agent_ids))])) # no action = [0,1,0]
    rewards[agent_id] = 0

totalReward = 0
done = False
# action_space_len = 3 # for all agents

# TODO: extra parameters : /home/miniconda3/envs/maddpg/lib/python3.7/site-packages/ray/rllib/policy/policy.py

iteration = 0
while not done:
    action_dict = {}
    # compute_action does not cut it. Go to the policy directly
    for agent_id in env.agent_ids:
        # print("id {}, obs {}, rew {}".format(agent_id, observations[agent_id], rewards[agent_id]))
        action, _, _ = RLAgent.get_policy("policy_0").compute_single_action(observations[agent_id], prev_reward=rewards[agent_id]) # prev_action=action_dict[agent_id]
        # print(action)
        action_dict[agent_id] = action

    observations, rewards, dones, info = env.step(action_dict)
    env.render()
    totalReward += sum(rewards.values())
    done = any(list(dones.values()))
    print("iter:", iteration, sum(rewards.values()))
    iteration += 1

env.close()
print("done", done, totalReward)
