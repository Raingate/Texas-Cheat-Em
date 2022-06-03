''' An example of training a reinforcement learning agent on the environments in RLCard
'''
import os
import sys
import argparse
from tqdm import trange
import numpy as np
import random

import rlcard
from rlcard.utils import (
    set_seed,
    reorganize
)

from agents.load_agent import load_agent
from utils.cheating_sheet import CheatingSheet


def train_cheeting_sheet(args):
    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize cheating sheet
    cheat_sheet = CheatingSheet(env, class_num=45)

    # Initialize the agent and use random agents as opponents
    agent1 = load_agent('random2', env)
    agent2 = load_agent('random2', env)

    agents = [agent1, agent2]
    env.set_agents(agents)

    fold_count = 0
    # Start training
    for episode in trange(args.num_episodes):
        # Generate data from the environment
        trajectories, payoffs = env.run(is_training=False)

        # Reorganaize the data to be state, action, reward, next_state, done
        trajectories = reorganize(trajectories, payoffs)

        if len(trajectories[0]) * len(trajectories[1]) == 0 or \
                trajectories[0][-1][1] == 2 or trajectories[1][-1][1] == 2:
            fold_count += 1

        cheat_sheet.update(trajectories[0])

    cheat_sheet.save_model(save_path=args.save_path)
    print(f"Cheating sheet is saved at {args.save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training Cheating-Sheet")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_episodes', type=int, default=1000000)
    parser.add_argument('--save_path', type=str, default='networks/cheating_sheet.npy')
    args = parser.parse_args()

    os.makedirs('networks', exist_ok=True)
    args.env = 'limit-holdem'
    train_cheeting_sheet(args)

