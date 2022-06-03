import argparse
import rlcard

from rlcard.utils import (
    set_seed,
    tournament,
)

from agents.load_agent import load_agent


def evaluate(args):
    print(f"--> Running on the {args.device}")

    # Seed numpy, torch, random
    set_seed(args.seed)

    # Make the environment with seed
    env = rlcard.make(args.env, config={'seed': args.seed})

    # Initialize the agent and use random agents as opponents
    agent1 = load_agent(args.player1, env, load_path=args.load_path1)
    agent2 = load_agent(args.player2, env, load_path=args.load_path2)

    agents = [agent1, agent2]
    env.set_agents(agents)

    # Evaluation
    print(f'{args.player1} vs {args.player2} payoffs:{tournament(env, args.num_eval_games)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Evaluation in RLCard")
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--player1', type=str, default='cheater',
                        choices=['ppo', 'dqn', 'rule', 'raise_rule', 'random', 'cheater', 'table'])
    parser.add_argument('--player2', type=str, default='random',
                        choices=['ppo', 'dqn', 'rule', 'raise_rule', 'random', 'table'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_eval_games', type=int, default=2000)
    parser.add_argument('--load_path1', type=str, default='')
    parser.add_argument('--load_path2', type=str, default='')
    parser.add_argument('--evaluate_every', type=int, default=100)
    parser.add_argument('--num_episodes', type=int, default=5000)
    args = parser.parse_args()

    args.env = 'limit-holdem'

    print(f'player1: {args.player1}')
    print(f'player2: {args.player2}')
    print(f'game round: {args.num_eval_games}')
    evaluate(args)
