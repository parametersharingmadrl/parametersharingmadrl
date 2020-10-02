import sys
import gym
import random
import numpy as np

import ray
from ray import tune
from ray.rllib.models import Model, ModelCatalog
from ray.tune.registry import register_env
from ray.rllib.utils import try_import_tf
from sisl_games.pursuit import pursuit
from ray.rllib.models.tf.tf_modelv2 import TFModelV2

tf = try_import_tf()

# RDQN - Rainbow DQN
# ADQN - Apex DQN
methods = ["A2C", "ADQN", "DQN", "IMPALA", "PPO", "RDQN"]

assert len(sys.argv) == 2, "Input the learning method as the second argument"
method = sys.argv[1]
assert method in methods, "Method should be one of {}".format(methods)

def env_creator(args):
    return pursuit.env()

env = env_creator(1)
register_env("pursuit", env_creator)

obs_space = gym.spaces.Box(low=0, high=1, shape=(148,), dtype=np.float32)
act_space = gym.spaces.Discrete(5)

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


num_agents = env.num_agents

if method == "ADQN":
    ModelCatalog.register_custom_model("MLPModelV2", MLPModelV2)
    def gen_policyV2(i):
        config = {
            "model": {
                "custom_model": "MLPModelV2",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_{}".format(i): gen_policyV2(i) for i in range(num_agents)}
    
else:
    ModelCatalog.register_custom_model("MLPModel", MLPModel)
    def gen_policy(i):
        config = {
            "model": {
                "custom_model": "MLPModel",
            },
            "gamma": 0.99,
        }
        return (None, obs_space, act_space, config)
    policies = {"policy_{}".format(i): gen_policy(i) for i in range(num_agents)}
    

# for all methods
policy_ids = list(policies.keys())

if __name__ == "__main__":
    if method == "A2C":
        tune.run(
            "A2C",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
        
                # Enviroment specific
                "env": "pursuit",
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "lr_schedule": [[0, 0.0007],[20000000, 0.000000000001]],
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )

    elif method == "ADQN":
        # APEX-DQN
        tune.run(
            "APEX",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
        
                # Enviroment specific
                "env": "pursuit",
        
                # General
                "log_level": "INFO",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )

    elif method == "DQN":
        # plain DQN
        tune.run(
            "DQN", 
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
                # Enviroment specific
                "env": "pursuit",
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
                # Method specific
                "dueling": False,
                "double_q": False,
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )

    elif method == "IMPALA":
        tune.run(
            "IMPALA",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
        
                # Enviroment specific
                "env": "pursuit",
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                "clip_rewards": True,
                "lr_schedule": [[0, 0.0005],[20000000, 0.000000000001]],
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )

    elif method == "PPO":
        tune.run(
            "PPO",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
        
                # Enviroment specific
                "env": "pursuit",
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "compress_observations": False,
                "gamma": .99,
        
        
                "lambda": 0.95,
                "kl_coeff": 0.5,
                "clip_rewards": True,
                "clip_param": 0.1,
                "vf_clip_param": 10.0,
                "entropy_coeff": 0.01,
                "train_batch_size": 5000,
                "sample_batch_size": 100,
                "sgd_minibatch_size": 500,
                "num_sgd_iter": 10,
                "batch_mode": 'truncate_episodes',
                "vf_share_layers": True,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )

    # psuedo-rainbow DQN
    elif method == "RDQN":
        tune.run(
            "DQN",
            stop={"episodes_total": 60000},
            checkpoint_freq=10,
            config={
        
                # Enviroment specific
                "env": "pursuit",
        
                # General
                "log_level": "ERROR",
                "num_gpus": 1,
                "num_workers": 8,
                "num_envs_per_worker": 8,
                "learning_starts": 1000,
                "buffer_size": int(1e5),
                "compress_observations": True,
                "sample_batch_size": 20,
                "train_batch_size": 512,
                "gamma": .99,
        
                # Method specific
        
                "multiagent": {
                    "policies": policies,
                    "policy_mapping_fn": (
                        lambda agent_id: policy_ids[agent_id]),
                },
            },
        )


