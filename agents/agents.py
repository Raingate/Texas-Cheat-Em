from pickle import TRUE
from matplotlib.pyplot import table
import numpy as np
from utils.cheating_sheet import CheatingSheet


class TableAgent(object):
    ''' An agent choose action based on cheating sheet.
    '''

    def __init__(self, num_actions, load_path=''):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.value_table = CheatingSheet(load_path=load_path)

    # @staticmethod
    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game
        Returns:
            action (str): Predicted action
        '''
        best_value = -99
        best_a = 0
        for a in list(state['legal_actions'].keys()):
            value = self.value_table.get_value(state, a)
            if value > best_value:
                best_value = value
                best_a = a
        return best_a

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])

        info = {'probs': {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]]
                          for i in range(len(state['legal_actions']))}}

        return self.step(state), info


class RandomAgent2(object):
    ''' A random agent. Random agents is for running toy examples on the card games
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.fold_ratio = 0.2

    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game
        Returns:
            action (str): Predicted action
        '''
        # if np.random.randint(0,8)<1:
        action_list = list(state['legal_actions'].keys())
        if np.random.randint(0, 100) > 100 * self.fold_ratio:
            action_list.remove(2)
        action = np.random.choice(action_list)
        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])

        info = {'probs': {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]]
                          for i in range(len(state['legal_actions']))}}

        return self.step(state), info


class Rule2Agent(object):
    ''' A ruled agent.
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions
        self.RANK_TO_STRING = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                               7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        self.STRING_TO_RANK = {v: k for k, v in self.RANK_TO_STRING.items()}

    def good_or_bad(self, hand):
        ranks = []
        for card in hand:
            ranks.append(int(self.STRING_TO_RANK[card[1]]))
        if ranks[0] == ranks[1]:
            return True
        if max(ranks) < 11:
            return False
        return True

    def step(self, state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game
        Returns:
            action (str): Predicted action
        '''
        # print(state)
        good = self.good_or_bad(state['raw_obs']['hand'])
        if good:
            legal_actions = state['raw_legal_actions']
            if 'raise' in legal_actions:
                action = 1  # 'raise'
            elif 'call' in legal_actions:
                action = 0  # 'call'
            elif 'check' in legal_actions:
                action = 3  # 'check'
            else:
                action = 2  # 'fold'
        else:
            action = 2

        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])

        info = {'probs': {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]]
                          for i in range(len(state['legal_actions']))}}

        return self.step(state), info


class BraveAgent(object):
    ''' Agent only choose 'raise'.
    '''

    def __init__(self, num_actions):
        ''' Initilize the random agent

        Args:
            num_actions (int): The size of the ouput action space
        '''
        self.use_raw = False
        self.num_actions = num_actions

    @staticmethod
    def step(state):
        ''' Predict the action when given raw state. A simple rule-based AI.
        Args:
            state (dict): Raw state from the game
        Returns:
            action (str): Predicted action
        '''
        legal_actions = state['raw_legal_actions']
        if 'raise' in legal_actions:
            action = 1  # 'raise'
        elif 'call' in legal_actions:
            action = 0  # 'call'
        elif 'check' in legal_actions:
            action = 3  # 'check'
        else:
            action = 2  # 'fold'

        return action

    def eval_step(self, state):
        ''' Predict the action given the current state for evaluation.
            Since the random agents are not trained. This function is equivalent to step function

        Args:
            state (dict): An dictionary that represents the current state

        Returns:
            action (int): The action predicted (randomly chosen) by the random agent
            probs (list): The list of action probabilities
        '''
        probs = [0 for _ in range(self.num_actions)]
        for i in state['legal_actions']:
            probs[i] = 1 / len(state['legal_actions'])

        info = {'probs': {state['raw_legal_actions'][i]: probs[list(state['legal_actions'].keys())[i]]
                          for i in range(len(state['legal_actions']))}}

        return self.step(state), info

