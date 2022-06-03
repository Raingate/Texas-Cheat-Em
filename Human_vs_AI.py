import numpy as np
import rlcard
import argparse
import torch
from rlcard.agents import LimitholdemHumanAgent as HumanAgent
from rlcard.utils.utils import print_card

from agents.load_agent import load_agent


if __name__ == '__main__':
    # get configuration
    parser = argparse.ArgumentParser("Human Player VS AI")
    parser.add_argument('--player', type=str, default='cheater',
                        choices=['ppo', 'dqn', 'rule', 'raise_rule', 'random', 'cheater', 'table'])
    parser.add_argument('--load_path', type=str, default='')
    args = parser.parse_args()

    # Make environment
    env = rlcard.make('limit-holdem')
    human_agent = HumanAgent(env.num_actions)
    agent = load_agent(args.player, env, load_path=args.load_path)
    env.set_agents([
        human_agent,
        agent
    ])

    print(f">> Human Player VS {args.player}")

    while (True):
        print(">> Start a new game")

        trajectories, payoffs = env.run(is_training=False)
        # If the human does not take the final action, we need to
        # print other players action
        if len(trajectories[0]) != 0:
            final_state = trajectories[0][-1]

            _action_list = []
            action_record = final_state['action_record']
            for i in range(1, len(action_record)+1):
                """
                if action_record[-i][0] == state['current_player']:
                    break
                """
                _action_list.insert(0, action_record[-i])
            for pair in _action_list:
                print('>> Player', pair[0], 'chooses', pair[1])

        # Let's take a look at what the agent card is
        print(f'=============   {args.player} Agent   ============')
        print_card(env.get_perfect_information()['hand_cards'][1])

        print('===============     Result     ===============')
        if payoffs[0] > 0:
            print('You win {} chips!'.format(payoffs[0]))
        elif payoffs[0] == 0:
            print('It is a tie.')
        else:
            print('You lose {} chips!'.format(-payoffs[0]))
        print('')

        input("Press any key to continue...")
