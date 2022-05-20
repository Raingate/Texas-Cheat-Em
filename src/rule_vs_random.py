import rlcard
import os
import torch
from tqdm import tqdm
from rlcard.agents import RandomAgent
from rlcard.models.uno_rule_models import UNORuleAgentV1
from rlcard.utils import (
    get_device,
    set_seed,
    tournament,
    reorganize,
    Logger,
    plot_curve,
)

num_episode = 10000
log_dir = '../experiments/rule_vs_random'

def train():
    # Check whether gpu is available
    device = get_device()
    # Seed numpy, torch, random
    set_seed(42)

    # Make the environment with seed
    env = rlcard.make('uno')

    # Initialize the agent and use random agents as opponents
    Ruleagent = UNORuleAgentV1()
    agents = [Ruleagent]

    for _ in range(1, env.num_players):
        agents.append(RandomAgent(num_actions=env.num_actions))
    env.set_agents(agents)

    # Start training
    with Logger(log_dir) as logger:
        for episode in tqdm(range(num_episode)):
            # Generate data from the environment
            trajectories, payoffs = env.run(is_training=True)

            # Reorganaize the data to be state, action, reward, next_state, done
            trajectories = reorganize(trajectories, payoffs)
            # Feed transitions into agent memory, and train the agent
            # Here, we assume that DQN always plays the first position
            # and the other players play randomly (if any)
            # for ts in trajectories[0]:
            #     DQNagent.feed(ts)

            # Evaluate the performance. Play with random agents.
            if episode % 100 == 0:
                logger.log_performance(
                    env.timestep,
                    tournament(
                        env,
                        2000,
                    )[0]
                )

        # Get the paths
        csv_path, fig_path = logger.csv_path, logger.fig_path

    # Plot the learning curve
    plot_curve(csv_path, fig_path, 'rule')

    # Save model
    save_path = os.path.join(log_dir, 'model.pth')
    torch.save(Ruleagent, save_path)
    print('Model saved in', save_path)


if __name__ == '__main__':
    train()
