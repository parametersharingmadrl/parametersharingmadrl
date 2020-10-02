import sys
from sisl_games.multiwalker.multiwalker import env as multiwalker_env
from sisl_games.pursuit.pursuit import env as pursuit_env
from sisl_games.waterworld.waterworld import env as waterworld_env
import ray
from ray.tune.registry import register_env
import ray.rllib.rollout as rollout

"""
python3 RunFromCheckpoint.py ~/ray_results/APEX_DDPG/APEX_DDPG_waterworld_918f836c_2020-05-14_22-27-05xn5ektmd/checkpoint_1330/checkpoint-1330 --run APEX_DDPG --env waterworld --no-render --track-progress --episodes 60000 --out rollouts.pkl
"""

envs = ["multiwalker", "pursuit", "waterworld"]
env_name = ""
for arg in sys.argv:
    if arg in envs:
        env_name = arg
        if arg == "multiwalker":
            def env_creator(args):
                return multiwalker_env()
        if arg == "pursuit":
            def env_creator(args):
                return pursuit_env()
        if arg == "waterworld":
            def env_creator(args):
                return waterworld_env()

env = env_creator(1)
register_env(env_name, env_creator)

if __name__ == "__main__":
    parser = rollout.create_parser()
    args = parser.parse_args()
    rollout.run(args, parser)
