import os
import torch
import argparse
import numpy as np
from tqdm import trange

import rlcard
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

from agents.load_agent import load_agent


def train(args):
    print(f"--> {args.player1} VS {args.player2}")
    print(f"--> Running on {args.device}")
    save_model_list = ['dqn', 'ppo', 'cheater']

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    agent1 = load_agent(args.player1, env, args.device)
    agent2 = load_agent(args.player2, env, args.device)

    agents = [agent1, agent2]
    env.set_agents(agents)

    # Start training
    with Logger(args.log_dir) as logger:
        for episode in trange(args.num_episodes):

            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)

            # get start agent idx
            if len(trajectories[0]) != 0:
                ts = trajectories[0][0]
                first_idx = ts[0]['action_record'][0][0]

                # train agent1
                if args.player1 == 'cheater':
                    agent1.cheat_sheet.update(trajectories[0])

                    for i in range(len(trajectories[0])):
                        ts = trajectories[0][i]
                        if first_idx != agent1.idx:
                            oppo_state = trajectories[1][i][0]
                            oppo_act = trajectories[1][i][1]
                        elif i == 0:
                            oppo_state = None
                            oppo_act = None
                        else:
                            oppo_state = trajectories[1][i - 1][0]
                            oppo_act = trajectories[1][i - 1][1]

                        act_dis = agent1.feed(ts, oppo_state, oppo_act)
                elif args.player1 in save_model_list:
                    for ts in trajectories[0]:
                        agent1.feed(ts)

                # train agent2
                if args.player2 in save_model_list:
                    for ts in trajectories[1]:
                        agent2.feed(ts)

            # Evaluate the performance.
            if episode % args.evaluate_every == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games,
                    )[0]
                )

            if episode == args.num_episodes - 1:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        args.num_eval_games*10,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, args.player1)

    # Save model
    if args.player1 in save_model_list:
        save_path1 = os.path.join(args.log_dir, f'{args.player1}.pth')
        torch.save(agent1, save_path1)
        print('Model1 saved in', save_path1)

    if args.player2 in save_model_list:
        save_path2 = os.path.join(args.log_dir, f'{args.player2}.pth')
        torch.save(agent2, save_path2)
        print('Model2 saved in', save_path2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training in RLCard")
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--player1', type=str, default='cheater',
                        choices=['ppo', 'dqn', 'rule', 'raise_rule', 'random', 'cheater', 'table'])
    parser.add_argument('--player2', type=str, default='ppo',
                        choices=['ppo', 'dqn', 'rule', 'raise_rule', 'random', 'table'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_eval_games', type=int, default=1000)
    parser.add_argument('--load_path', type=str, default='')
    parser.add_argument('--evaluate_every', type=int, default=100, help="eval step")
    parser.add_argument('--num_episodes', type=int, default=5000, help='training epochs')
    args = parser.parse_args()

    args.env = 'limit-holdem'
    args.log_dir = os.path.join('experiments', args.env + '_' + args.player1 + '_' + args.player2 + '_result')

    train(args)
