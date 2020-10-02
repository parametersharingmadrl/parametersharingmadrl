from .multiagentenv import MultiAgentEnv
from sisl_games.pursuit import pursuit
from sisl_games.waterworld import waterworld
from sisl_games.multiwalker import multiwalker
from supersuit import action_lambda_v0,flatten_v0
import gym
import numpy as np

def make_env(env_name):
    if env_name == "pursuit":
        return pursuit.env()
    elif env_name == "waterworld":
        return waterworld.env()
    elif env_name == "multiwalker":
        return multiwalker.env()
    else:
        raise RuntimeError("bad environment name")

class PettingZooEnv(MultiAgentEnv):
    def __init__(self, env_name, seed=None):
        self.env = make_env(env_name)#.env()
        self.env.observation_spaces = self.env.observation_space_dict
        self.env.action_spaces = self.env.action_space_dict
        self.env.agents = self.env.agent_ids
        act_space = self.env.action_spaces[self.env.agents[0]]
        if isinstance(act_space, gym.spaces.Box):
            self.num_discrete_acts = 51
            self.all_actions = [act_space.sample() for _ in range(self.num_discrete_acts)]
        else:
            self.num_discrete_acts = act_space.n
            self.all_actions = list(range(self.num_discrete_acts))

        # self.env = action_lambda_v0(self.env,
        #     lambda action, space : random_acts[action],
        #     lambda space: gym.spaces.Discrete(num_discrete_acts))
        # self.env = flatten_v0(self.env)
        self.env.reset()
        self.episode_limit = 505#self.env.max_frames
        self.n_agents = self.env.num_agents

    def step(self, actions):
        action_dict = {agent: self.all_actions[act] for agent, act in zip(self.env.agents, actions)}
        obs, rews,dones, infos =self.env.step(action_dict)#, observe=False)
        # print(dones)
        self.ep_len += 1
        self.observations = [obs[agent] for agent in self.env.agents]
        return sum(rews.values())/self.env.num_agents, all(dones.values()) or self.ep_len >= 500, {}

    def get_stats(self):
        return {}

    def get_obs(self):
        return [self.get_obs_agent(agent) for agent in self.env.agents]

    def get_obs_agent(self, agent_id):
        # print(self.observations[agent_id].flatten().shape)
        # print(self.get_obs_size())
        return self.observations[agent_id].flatten()

    def get_obs_size(self):
        return int(np.prod(next(iter(self.env.observation_spaces.values())).shape))

    def get_state(self):
        return np.concatenate([self.get_obs_agent(o) for o in self.env.agents],axis=0)

    def get_state_size(self):
        return  self.get_obs_size()*self.env.num_agents

    def get_avail_actions(self):
        return [[1]*self.get_total_actions()]*self.n_agents

    def get_avail_agent_actions(self, agent_id):
        return [1]*self.get_total_actions()

    def get_total_actions(self):
        return self.num_discrete_acts

    def reset(self):
        obs = self.env.reset()
        self.ep_len = 0
        self.observations = [obs[agent] for agent in self.env.agents]
        return self.get_obs(), self.get_state()

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def seed(self):
        pass#raise NotImplementedError

    def save_replay(self):
        pass#raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
