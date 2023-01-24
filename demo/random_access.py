"""Load small parts of dataset at random positions to test performance."""


import argparse
from time import time

import gymnasium as gym
import numpy as np

import trifinger_rl_datasets  # noqa


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument(
        "--env",
        type=str,
        default="trifinger-cube-push-real-expert-image-mini-v0",
        help="Name of dataset environment to load.",
    )
    argparser.add_argument(
        "--n-parts",
        type=int,
        default=500,
        help="Number of contiguous parts to load from file.",
    )
    argparser.add_argument(
        "--part-size",
        type=int,
        default=10,
        help="Number of transitions to load per part.",
    )
    argparser.add_argument(
        "--h5path",
        type=str,
        default=None,
        help="Path to HDF5 file to load.",
    )
    args = argparser.parse_args()

    # create environment
    env = gym.make(
        args.env,
        disable_env_checker=True
    )

    stats = env.get_dataset_stats(h5path=args.h5path)
    print("Number of timesteps in dataset: ", stats["n_timesteps"])

    t0 = time()
    # load subsets of the dataset at random positions
    for i in range(args.n_parts):
        start = np.random.randint(0, stats["n_timesteps"] - args.part_size)
        rng = (
            start,
            start + args.part_size
        )
        part = env.get_dataset(rng=rng, h5path=args.h5path)
    t1 = time()
    print(f"Loaded {args.n_parts} parts of size {args.part_size} in {t1 - t0:.2f} s")