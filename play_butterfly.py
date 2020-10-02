from sisl_games.pursuit import pursuit
import ray
from ray.tune.registry import register_trainable, register_env

import ray.rllib.agents.a3c.a2c as a2c  # A2CTrainer
import ray.rllib.agents.dqn.apex as adqn  # ApexTrainer
import ray.rllib.agents.dqn as dqn  # DQNTrainer
import ray.rllib.agents.impala as impala  # ImpalaTrainer
import ray.rllib.agents.ppo as ppo  # PPOTrainer

import os
import pickle
import numpy as np
import pandas as pd
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
import sys
import matplotlib.pyplot as plt
from pettingzoo.butterfly import prospector_v1, pistonball_v0, knights_archers_zombies_v2, cooperative_pong_v1
from pettingzoo_env import ParallelPettingZooEnv
from parameterSharingButterfly import make_env_creator
import parameterSharingButterfly
import ray.rllib.agents.ddpg as ddpg  # TD3Trainer
import ray.rllib.agents.ddpg.td3 as td3  # TD3Trainer


plt.rcParams["font.family"] = "serif"

tf = try_import_tf()

env_name = sys.argv[1]
if env_name == "KAZ":
    env_constr = knights_archers_zombies_v2
    method = "APEX"
    model = None
elif env_name == "pistonball":
    env_constr = pistonball_v0
    method = "APEX"
    model = "MLPModelV2"
elif env_name == "prospector":
    env_constr = prospector_v1
    method = "PPO"
    model = None
elif env_name == "pong":
    env_constr = cooperative_pong_v1
    method = "APEX"
    model = "MLPModelV2"
elif env_name == "prison":
    env_constr = prison_v1
    method = "APEX"
    model = "MLPModelV2"
else:
    assert False, "argv"
parameterSharingButterfly.model = model
algorithm = method if method != "APEX" else "ADQN"
data_path = sys.argv[2]
checkpoint_number = int(sys.argv[3])


print("pre_init",flush=True)
ray.init()
print("initted",flush=True)

checkpoint_path = data_path+"/checkpoint_"+str(checkpoint_number)+'/checkpoint-'+str(checkpoint_number)

env_creator = make_env_creator(env_constr)

env = env_creator(1)
register_env(env_name, env_creator)

config_path = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_path, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)
    config["num_workers"] = 1
    config["num_gpus"] = 0


if algorithm == 'A2C':
    RLAgent = a2c.A2CTrainer(env=env_name, config=config)
elif algorithm == 'ADQN':
    RLAgent = adqn.ApexTrainer(env=env_name, config=config)
elif algorithm == 'DQN':
    RLAgent = dqn.DQNTrainer(env=env_name, config=config)
elif algorithm == 'IMPALA':
    RLAgent = impala.ImpalaTrainer(env=env_name, config=config)
elif algorithm == 'PPO':
    RLAgent = ppo.PPOTrainer(env=env_name, config=config)
elif algorithm == 'RDQN':
    RLAgent = dqn.DQNTrainer(env=env_name, config=config)
elif algorithm == "DDPG":
    RLAgent = ddpg.DDPGTrainer(env=env_name, config=config)

print(checkpoint_path,flush=True)
#RLAgent.restore(checkpoint_path)

num_runs = 50
totalRewards = np.empty((num_runs,))

policy = RLAgent.get_policy("policy_0")
for j in range(num_runs):
    observations = env.reset()
    rewards, action_dict = {}, {}
    for agent_id in env.agents:
        rewards[agent_id] = 0

    totalReward = 0
    done = False
    iteration = 0
    while not done:
        action_dict = {}
        for agent_id in env.agents:
            #print("hit get_policy",flush=True)
            action, _, _ = policy.compute_single_action(observations[agent_id], prev_reward=rewards[agent_id]) # prev_action=action_dict[agent_id]
            #print("action: ", action)
            action_dict[agent_id] = action

        observations, rewards, dones, info = env.step(action_dict)
        env.render()
        totalReward += sum(rewards.values())
        done = any(list(dones.values()))
        # if sum(rewards.values()) > 0:
        #     print("rewards", rewards)
        # print("iter:", iteration, sum(rewards.values()))
        iteration += 1
    print("episode finished")
    totalRewards[j] = totalReward

env.close()

print("\n\ndone: ", done, ', Mean Total Reward: ',np.mean(totalRewards), 'Total Reward: ', totalRewards)
print("\nMean Total Reward: ", np.mean(totalRewards))

df = pd.read_csv(os.path.join(data_path,'progress.csv'))
df = df[['training_iteration', "episode_reward_mean", "episodes_total"]]

iter_range = list(range(10,50000,10))
df2 = df[df['training_iteration'].isin(iter_range)]
iter_max = df2.loc[df2['episode_reward_mean'].idxmax(), ['training_iteration', "episode_reward_mean", "episodes_total"]]

rew = df.loc[df['training_iteration'] == checkpoint_number, ['episode_reward_mean']]
rew = rew.to_numpy()[0][0]
epi = df.loc[df['training_iteration'] == checkpoint_number, ['episodes_total']]
rew_max = df['episode_reward_mean'].max()
epi_max = df.loc[df['episode_reward_mean'].idxmax(), ['episodes_total','training_iteration']]
epi = epi.to_numpy()[0][0]

print("Progress Report Reward: ", rew)
print("Reward Error Factor: ", np.mean(totalRewards)/rew, '\n')
print("Max of ",rew_max, " at ", int(epi_max[0]), " episodes (",int(epi_max[1]),' iterations)')
print("Episodes Total: ", epi, "\n")
print("Max Possible Reward of {} at {} episodes ({} iterations)\n\n".format(iter_max[1],int(iter_max[2]),int(iter_max[0])))
