import torch
from rlcard.agents import RandomAgent
from rlcard.agents import DQNAgent
from rlcard.models.limitholdem_rule_models import LimitholdemRuleAgentV1

from agents.cheat_agent import CheatAgent
from agents.ppo_agent import PPOAgent
from agents.agents import *


def load_agent(player_type, env, device='cpu', load_path=''):
    """ load agent

    :param player_type: type of player, str
    :param env: env name, str
    :param device: cpu or cuda
    :param idx: player id, int
    :param load_path: pretrained model path, str
    :return: agent
    """
    if player_type == 'dqn':
        if len(load_path) == 0:
            agent = DQNAgent(
                num_actions=env.num_actions,
                state_shape=env.state_shape[0],
                mlp_layers=[64, 64],
                device=device,
            )
        else:
            agent = torch.load(load_path)
            print(load_path, " is loaded.")
    elif player_type == 'ppo':
        if len(load_path) == 0:
            agent = PPOAgent(
                num_actions=env.num_actions,
                device=device,
            )
        else:
            agent = torch.load(load_path)
            print(load_path, " is loaded.")
    elif player_type == 'cheater':
        if len(load_path) == 0:
            agent = CheatAgent(
                num_actions=env.num_actions,
                device=device,
            )
        else:
            agent = torch.load(load_path)
            print(load_path, " is loaded.")
    elif player_type == 'rule':
        agent = LimitholdemRuleAgentV1()
    elif player_type == 'rule2':
        agent = Rule2Agent(num_actions=env.num_actions)
    elif player_type == 'table':
        agent = TableAgent(num_actions=env.num_actions, load_path=load_path)
    elif player_type == 'raise_rule':
        agent = BraveAgent(num_actions=env.num_actions)
    elif player_type == 'random':
        agent = RandomAgent(num_actions=env.num_actions)
    elif player_type == 'random2':
        agent = RandomAgent2(num_actions=env.num_actions)

    return agent