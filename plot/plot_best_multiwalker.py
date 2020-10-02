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
    "ytick.labelsize": 4,
    "text.usetex": True,
    "pgf.rcfonts": False
});

plt.figure(figsize=(2.65, 1.5))
data_path = "data/multi_walker/"

df = pd.read_csv(os.path.join(data_path, 'sa_ppo.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/30), 5)
plt.plot(data[:, 0], filtered, label='PPO (Independent)', linewidth=0.6, linestyle=(0, (5, 2, 1, 2)))

df = pd.read_csv(os.path.join(data_path, 'ppo1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/30)+1, 5)
plt.plot(data[:, 0], filtered, label='PPO (Shared)', linewidth=0.6, linestyle=(0, (1, 1)))

df = pd.read_csv(os.path.join(data_path, 'maddpg.csv'))
df = df[['episode', "reward"]]
data = df.to_numpy()
plt.plot(data[:, 0], data[:, 1], label='MADDPG', linewidth=0.6, color='grey', linestyle='solid')

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Multiwalker')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
plt.yticks(ticks=[-100,-50,0],labels=['-100','-50','0'])
plt.ylim(-150, 50)
plt.tight_layout()
plt.legend(loc='center right', ncol=1, labelspacing=.2, columnspacing=.25, borderpad=.25)
plt.margins(x=0)
plt.savefig("BestMultiwalkerGraph_camera.pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("BestMultiwalkerGraph_camera.png", bbox_inches = 'tight',pad_inches = .025, dpi = 600)

