import argparse
import pickle

def parse_args():
    parser = argparse.ArgumentParser("")
    # Environment
    parser.add_argument("pickle_file", type=str, help="name of the pickle input file")
    parser.add_argument("csv_file", type=str, help="name of the csv output file")
    parser.add_argument("iters_per_step", type=int, help="number of iters per save")
    return parser.parse_args()

def main(arglist):
    csv_rows = [["episode", "reward"]]
    with open(arglist.pickle_file, 'rb') as fp:
        ep_rewards = pickle.load(fp)
        for i, rew in enumerate(ep_rewards):
            csv_rows.append([str(i*arglist.iters_per_step),str(rew)])

    with open(arglist.csv_file, 'w') as fp:
        fp.write("\n".join([",".join(row) for row in csv_rows]))

if __name__ == "__main__":
    main(parse_args())
