import sys
import gym
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
import pettingzoo
from pettingzoo.butterfly import prospector_v1, pistonball_v0, knights_archers_zombies_v2, pistonball_v0, cooperative_pong_v1, prison_v1
import supersuit
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from pettingzoo_env import PettingZooEnv
import tensorflow as tf


from ray.rllib.env.multi_agent_env import MultiAgentEnv



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

class AtariModel(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config,
                 name="atari_model"):
        super(AtariModel, self).__init__(obs_space, action_space, num_outputs, model_config,
                         name)
        inputs = tf.keras.layers.Input(shape=obs_space.shape, name='observations')
        # Convolutions on the frames on the screen
        layer1 = tf.keras.layers.Conv2D(
                32,
                [8, 8],
                strides=(4, 4),
                activation="relu",
                data_format='channels_last')(inputs)
        layer2 = tf.keras.layers.Conv2D(
                64,
                [4, 4],
                strides=(2, 2),
                activation="relu",
                data_format='channels_last')(layer1)
        layer3 = tf.keras.layers.Conv2D(
                64,
                [3, 3],
                strides=(1, 1),
                activation="relu",
                data_format='channels_last')(layer2)
        layer4 = tf.keras.layers.Flatten()(layer3)
        layer5 = tf.keras.layers.Dense(
                512,
                activation="relu",
                kernel_initializer=normc_initializer(1.0))(layer4)
        action = tf.keras.layers.Dense(
                num_outputs,
                activation="linear",
                name="actions",
                kernel_initializer=normc_initializer(0.01))(layer5)
        value_out = tf.keras.layers.Dense(
                1,
                activation=None,
                name="value_out",
                kernel_initializer=normc_initializer(0.01))(layer5)
        self.base_model = tf.keras.Model(inputs, [action, value_out])
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict["obs"])
        return model_out, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])

ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
ModelCatalog.register_custom_model("MLPModel", MLPModel)
ModelCatalog.register_custom_model("AtariModel", AtariModel)

def make_env_creator(env_constr):
    def env_creator(args):
        env = env_constr.env()#killable_knights=False, killable_archers=False)
        resize_size = 84 if model == None else 32
        env = supersuit.resize_v0(env,resize_size, resize_size, linear_interp=True)
        env = supersuit.color_reduction_v0(env)
        env = supersuit.pad_action_space_v0(env)
        env = supersuit.pad_observations_v0(env)
        # env = supersuit.frame_stack_v0(env,2)
        env = supersuit.dtype_v0(env, np.float32)
        env = supersuit.normalize_obs_v0(env)
        if model == "MLPModelV2":
            env = supersuit.flatten_v0(env)
        env = PettingZooEnv(env)
        return env
    return env_creator

# DQN and Apex-DQN do not work with continuous actions
if __name__ == "__main__":
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

    # multiwalker
    env_creator = make_env_creator(env_constr)

    env = env_creator(1)
    print(env.observation_space.dtype)
    register_env("butterfly", env_creator)

    agent = env.agents[0]
    obs_space = env.observation_spaces[agent]  # gym.spaces.Box(low=0, high=1, shape=(148,), dtype=np.float32)
    act_space = env.action_spaces[agent]  # gym.spaces.Discrete(5)

    def gen_policy(i):
        config = {
            "gamma": 0.99,
        }
        if model is not None:
            config['model'] =  {
                "custom_model": model,
            }
        return (None, obs_space, act_space, config)

    policies = {"policy_0": gen_policy(0)}
    policy_ids = list(policies.keys())

    # APEX-DQN
    if method == "PPO":
        tune.run(
            method,
            name="PPO-"+env_name,
            stop={"episodes_total": 50000},
            checkpoint_freq=10,
            config={

                # Enviroment specific
                "env": "butterfly",

                # General
                "log_level": "INFO",
                "num_gpus": 1,
                "num_workers": 12,
                "num_envs_per_worker": 4,
                "compress_observations": False,
                "gamma": .99,
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "sample_batch_size": 25,
                "sgd_minibatch_size": 256,
                "num_sgd_iter": 100,
                "batch_mode": 'truncate_episodes',
                "vf_share_layers": True,

                #"n_step": 3,
                # "lr": .00001,
                # "prioritized_replay_alpha": 0.5,
                # "final_prioritized_replay_beta": 1.0,
                # "target_network_update_freq": 20000,
                # "timesteps_per_iteration": 15000,

                # Method specific

                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )
    elif method == "APEX":
        tune.run(
            "APEX",
            name="ADQN-"+env_name,
            stop={"episodes_total": 50000},
            checkpoint_freq=10,
            #local_dir="~/ray_results_atari/"+env_name,
            config={

                # Enviroment specific
                "env": "butterfly",
                "double_q": True,
                "dueling": True,
                "num_atoms": 1,
                "noisy": False,
                "n_step": 3,
                "lr": 0.0001,
                #"lr": 0.0000625,
                "adam_epsilon": 1.5e-4,
                "buffer_size": int(4e5),
                "exploration_config": {
                    "final_epsilon": 0.01,
                    "epsilon_timesteps": 200000,
                },
                "prioritized_replay": True,
                "prioritized_replay_alpha": 0.5,
                "prioritized_replay_beta": 0.4,
                "final_prioritized_replay_beta": 1.0,
                "prioritized_replay_beta_annealing_timesteps": 2000000,

                "num_gpus": 1,

                "log_level": "ERROR",
                "num_workers": 12,
                "num_envs_per_worker": 4,
                "rollout_fragment_length": 32,
                "train_batch_size": 512,
                "target_network_update_freq": 10000,
                "timesteps_per_iteration": 15000,
                "learning_starts": 10000,
                "compress_observations": False,
                "gamma": 0.99,
                # Method specific
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[0]),
                },
            },
        )
