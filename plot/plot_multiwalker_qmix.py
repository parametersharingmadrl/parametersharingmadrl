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


colors = {'gupta_ddpg': 'grey',
        'gupta_trpo': 'black',
        'rainbow_dqn': 'tab:blue',
        'ppo': 'tab:orange',
        'impala': 'tab:green',
        'apex_dqn': 'tab:brown',
        'a2c': 'tab:purple',
        'apex_ddpg': 'tab:brown',
        'sac': 'tab:pink',
        'td3': 'tab:olive',
        'dqn': 'tab:cyan',
        'ddpg': 'steelblue',
        'maddpg': 'grey',
        'random': 'red'
        }

linestyles = {'gupta_ddpg': 'solid',
        'gupta_trpo': (0, (1, 1)),
        'rainbow_dqn': (0, (3, 3)),
        'ppo': (0, (5, 2, 1, 2)),
        'impala': 'solid',
        'apex_dqn': (0, (1, 1)),
        'a2c': (0, (3, 3)),
        'apex_ddpg': (0, (5, 2, 1, 2)),
        'sac': 'solid',
        'td3': (0, (1, 1)),
        'dqn': (0, (3, 3)),
        'ddpg': (0, (5, 2, 1, 2)),
        'maddpg': 'solid',
        'random': (0, (1, 1))
        }


plt.figure(figsize=(2.65, 1.5))
data_path = "data/multi_walker/"

df = pd.read_csv(os.path.join(data_path, 'ppo1.csv'))
df = df[['episodes_total', "episode_reward_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/30)+1, 5)
plt.plot(data[:, 0], filtered, label='PPO (parameter sharing)', linewidth=0.6, color='tab:blue', linestyle='-')

df = pd.read_csv(os.path.join(data_path, 'maddpg.csv'))
df = df[['episode', "reward"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1], int(len(data[:, 1])/30)+1, 5)
plt.plot(data[:, 0], filtered, label='MADDPG', linewidth=0.6, color='tab:orange', linestyle='-')

df = pd.read_csv(os.path.join('qmix_results/multiwalker', 'cout.txt.csv'))
df = df[['episode', "return_mean"]]
data = df.to_numpy()
filtered = scipy.signal.savgol_filter(data[:, 1]*3, int(len(data[:, 1])/30)+1, 5)
plt.plot(data[:, 0], filtered, label='QMIX', linewidth=0.6, color='tab:green', linestyle='-')

#plt.plot(np.array([0,60000]),np.array([-102.05,-102.05]), label='Random', linewidth=0.6, color='red', linestyle=(0, (1, 1)))

plt.xlabel('Episode', labelpad=1)
plt.ylabel('Average Total Reward', labelpad=1)
plt.title('Multiwalker')
plt.xticks(ticks=[10000,20000,30000,40000,50000],labels=['10k','20k','30k','40k','50k'])
plt.xlim(0, 60000)
plt.yticks(ticks=[-150,-100,-50,0],labels=['-150','-100','-50','0'])
plt.ylim(-200, 50)
plt.tight_layout()
plt.legend(loc='lower center', ncol=2, labelspacing=.2, columnspacing=.25, borderpad=.25, bbox_to_anchor=(0.5, -0.6))
plt.margins(x=0)
plt.savefig("qmix_multiwalker.pgf", bbox_inches = 'tight',pad_inches = .025)
plt.savefig("qmix_multiwalker.png", bbox_inches = 'tight',pad_inches = .025, dpi=600)
