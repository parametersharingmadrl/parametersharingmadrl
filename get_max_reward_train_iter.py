import sys
import os
import numpy as np


assert len(sys.argv) == 2, "Provide path to the directory that holds progress.csv, as path/to/progress.csv"

path_to_csv = sys.argv[1]
assert os.path.exists(path_to_csv), "progress.csv does not exist in the given path"

# number of values to report
n = 20
with open(path_to_csv, 'r') as f:
    num_lines = sum(1 for _ in f) - 1
    print("Number of lines in csv = ", num_lines)
    f.seek(0, 0)
    for idx, l in enumerate(f):
        s = l.split(',')
        if idx == 0:
            reward_id = s.index("episode_reward_mean")
            train_id =  s.index("training_iteration")
            print("Indices of episode_reward_mean = {} and training_iteration = {}".format(reward_id, train_id))
            dtype = [("reward", float), ("iter", int)]
            arr = np.array([(0.0, 0)] * num_lines, dtype=dtype)
            continue
        reward = float(s[reward_id])
        if np.isnan(reward): reward = -1000.0
        arr[idx-1] = (reward, int(s[train_id]))

    sorted_arr = np.sort(arr, order='reward')[::-1]
    print("Top {} rewards and their training_iterations".format(n))
    print(*sorted_arr[:n], sep='\n')
    print("Average of top {} max rewards = {:.3f}".format(n, sum([i for i,j in sorted_arr[:n]])/n))
