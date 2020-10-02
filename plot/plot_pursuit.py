import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import scipy
from scipy import signal

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
data_path = "data/pursuit/"

data = np.genfromtxt(os.path.join(data_path,'gupta_trpo.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], '--', label='TRPO (Gupta)', linewidth=0.6, color='black', linestyle='solid')

data = np.genfromtxt(os.path.join(data_path,'gupta_dqn.csv'), delimiter=',')
plt.plot(data[:, 0], data[:, 1], label='DQN (Gupta)', linewidth=0.6, color='grey', linestyle=(0, (1, 1)))

df = pd.read_csv(os.path.join(data_path,'a2c_2.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, label='A2C', linewidth=0.6, color='tab:purple', linestyle=(0, (3, 3)))

df = pd.read_csv(os.path.join(data_path, 'dqn_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1]/8, '--', label='DQN', linewidth=0.6, color='tab:cyan', linestyle=(0, (3, 3)))

df = pd.read_csv(os.path.join(data_path,'impala_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='IMPALA', linewidth=0.6, color='tab:green', linestyle='solid')

df = pd.read_csv(os.path.join(data_path, 'ppo_1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='PPO', linewidth=0.6, color='tab:orange', linestyle=(0, (5, 2, 1, 2)))

df = pd.read_csv(os.path.join(data_path, 'rainbow_dqn.csv'))
df = df[['training_iteration', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/110)+1, 5)
plt.plot(data[:, 0]*8, filtered, label='Rainbow DQN', linewidth=0.6, color='tab:blue', linestyle=(0, (3, 3)))

df = pd.read_csv(os.path.join(data_path, 'apex_dqn.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/110)+2, 5)
plt.plot(data[:, 0], filtered, label='ApeX DQN', linewidth=0.6, color='tab:brown', linestyle=(0, (1, 1)))

plt.plot(np.array([0,60000]),np.array([31.03,31.03]), label='Random', linewidth=0.6, color='red', linestyle=(0, (1, 1)))

plt.xlabel('Episode', labelpad=1) 
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Pursuit')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
plt.yticks(ticks=[0,150,300,450,600],labels=['0','150','300','450','600'])
plt.ylim(-150, 750)
plt.tight_layout()
# plt.legend(loc='lower right', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("pursuitGraph_camera.pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("pursuitGraph_camera.png", bbox_inches = 'tight',pad_inches = .025, dpi = 600)
