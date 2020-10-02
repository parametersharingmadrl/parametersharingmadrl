import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("pgf")
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "font.size": 6,
    "legend.fontsize": 5,
    "text.usetex": True,
    "pgf.rcfonts": False
});

plt.figure(figsize=(2.65, 1.5))
data_path = "data/waterworld/"

df = pd.read_csv(os.path.join(data_path,'sa_a2c.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], '--', label='A2C', linewidth=0.6, color='tab:purple', linestyle=(0, (3, 3)))

df = pd.read_csv(os.path.join(data_path, 'sa_sac.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], filtered, label='SAC', linewidth=0.6, color='tab:pink', linestyle='solid')

df = pd.read_csv(os.path.join(data_path,'sa_apex_ddpg.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/30)+1, 5)
plt.plot(data[:, 0], filtered, '--', label='ApeX DDPG', linewidth=0.6, color='tab:brown', linestyle=(0, (5, 2, 1, 2)))

df = pd.read_csv(os.path.join(data_path,'sa_ddpg.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], '--', label='DDPG', linewidth=0.6, color='steelblue', linestyle=(0, (5, 2, 1, 2)))

df = pd.read_csv(os.path.join(data_path,'sa_impala.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA', linewidth=0.6, color='tab:green', linestyle='solid')

df = pd.read_csv(os.path.join(data_path, 'sa_ppo.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50)+1,5)
plt.plot(data[:, 0], data[:, 1], label='PPO', linewidth=0.6, color='tab:orange', linestyle=(0, (5, 2, 1, 2)))

df = pd.read_csv(os.path.join(data_path,'sa_td3.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1],int(len(data[:, 1])/50),5)
plt.plot(data[:, 0], filtered, label='TD3', linewidth=0.6, color='tab:olive', linestyle=(0, (1, 1)))

plt.plot(np.array([0,60000]),np.array([-6.82,-6.82]), label='Random', linewidth=0.6, color='red', linestyle=(0, (1, 1)))

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Waterworld')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
plt.yticks(ticks=[-10,0,10],labels=['-10','0','10'])
plt.ylim(-20, 20)
plt.tight_layout()
# plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("SAWaterworldGraph_camera.pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("SAWaterworldGraph_camera.png", bbox_inches = 'tight',pad_inches = .025, dpi=600)

