from sisl_games.waterworld import waterworld
import ray
from ray.tune.registry import register_trainable, register_env

import ray.rllib.agents.a3c.a2c as a2c  # A2CTrainer
import ray.rllib.agents.ddpg.apex as apex  # ApexDDPGTrainer
import ray.rllib.agents.ddpg as ddpg  # TD3Trainer
import ray.rllib.agents.impala as impala  # ImpalaTrainer
import ray.rllib.agents.ppo as ppo  # PPOTrainer
import ray.rllib.agents.sac as sac  # SACTrainer
import ray.rllib.agents.ddpg.td3 as td3  # TD3Trainer

import os
import pickle
import numpy as np
import pandas as pd
from ray.rllib.models import Model, ModelCatalog
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.utils import try_import_tf
import sys
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"

tf = try_import_tf()

env_name = "waterworld"
algorithm = sys.argv[1].upper()
checkpoint_number = int(sys.argv[2])

methods = ["A2C", "APEX_DDPG", "DDPG", "IMPALA", "PPO", "SAC", "TD3"]

assert algorithm in methods, "{} is not part of {}".format(algorithm, methods)

ray.init()

data_path = "../ray_results/waterworld/SA_"+algorithm
checkpoint_path = data_path+"/checkpoint_"+str(checkpoint_number)+'/checkpoint-'+str(checkpoint_number)

def env_creator(args):
    if env_name == 'waterworld':
        return waterworld.env()
    elif env_name == 'multiwalker':
        return multiwalker.env()
    elif env_name == 'pursuit':
        return pursuit.env()
    

env = env_creator(1)
register_env(env_name, env_creator)

config_path = os.path.dirname(checkpoint_path)
config_path = os.path.join(config_path, "../params.pkl")
with open(config_path, "rb") as f:
    config = pickle.load(f)

class MLPModel(Model):
    def _build_layers_v2(self, input_dict, num_outputs, options):
        last_layer = tf.layers.dense(
                input_dict["obs"], 400, activation=tf.nn.relu, name="fc1")
        last_layer = tf.layers.dense(
            last_layer, 300, activation=tf.nn.relu, name="fc2")
        output = tf.layers.dense(
            last_layer, num_outputs, activation=None, name="fc_out")
        return output, last_layer

class MLPModelV2(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="my_model"):
        super().__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        # Simplified to one layer.
        input = tf.keras.layers.Input(obs_space.shape, dtype=obs_space.dtype)
        output = tf.keras.layers.Dense(num_outputs, activation=None)
        self.base_model = tf.keras.models.Sequential([input, output])
        self.register_variables(self.base_model.variables)
    def forward(self, input_dict, state, seq_lens):
        return self.base_model(input_dict["obs"]), []

ModelCatalog.register_custom_model("MLPModel", MLPModel)
ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)

if algorithm == 'A2C':
    RLAgent = a2c.A2CTrainer(env=env_name, config=config)
elif algorithm == 'APEX_DDPG':
    RLAgent = apex.ApexDDPGTrainer(env=env_name, config=config)
elif algorithm == 'DDPG':
    RLAgent = ddpg.DDPGTrainer(env=env_name, config=config)
elif algorithm == 'IMPALA':
    RLAgent = impala.ImpalaTrainer(env=env_name, config=config)
elif algorithm == 'PPO':
    RLAgent = ppo.PPOTrainer(env=env_name, config=config)
elif algorithm == 'SAC':
    RLAgent = sac.SACTrainer(env=env_name, config=config)
elif algorithm == 'TD3':
    RLAgent = td3.TD3Trainer(env=env_name, config=config)
RLAgent.restore(checkpoint_path)

num_runs = 50
totalRewards = np.empty((num_runs,))

for j in range(num_runs):
    observations = env.reset()
    rewards, action_dict = {}, {}
    for agent_id in env.agent_ids:
        assert isinstance(agent_id, int), "Error: agent_ids are not ints."
        action_dict = dict(zip(env.agent_ids, [env.action_space_dict[i].sample() for i in env.agent_ids]))
        rewards[agent_id] = 0

    totalReward = 0
    done = False
    iteration = 0
    while not done:
        action_dict = {}
        for agent_id in env.agent_ids:
            action, _, _ = RLAgent.get_policy("policy_{}".format(agent_id)).compute_single_action(observations[agent_id], prev_reward=rewards[agent_id]) # prev_action=action_dict[agent_id]
            # print(action)
            for ii in range(len(action)):
                if action[ii] < -1:
                    action[ii] = -1
                elif action[ii] > 1:
                    action[ii] = 1
            action_dict[agent_id] = action

        observations, rewards, dones, info = env.step(action_dict)
        # env.render()
        totalReward += sum(rewards.values())
        done = any(list(dones.values()))
        # if sum(rewards.values()) > 0:
        #     print("rewards", rewards)
        # print("iter:", iteration, sum(rewards.values()))
        iteration += 1
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
