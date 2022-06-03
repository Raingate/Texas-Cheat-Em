import torch
import random
import numpy as np
import torch.nn as nn
from matplotlib.cbook import flatten
from rlcard.utils.utils import remove_illegal
from collections import namedtuple
import torch.nn.functional as F
from utils.cheating_sheet import CheatingSheet

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done', 'opp_act', 'value', 'p'])


class CheatAgent(object):

    def __init__(self,
                 num_actions,
                 device,
                 learning_rate=1e-3,
                 replay_memory_size=48,
                 batch_size=48,
                 replay_memory_init_size=48,
                 train_every=50,
                 use_batch=True,
                 idx=0):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.idx = idx

        self.policy = Network(16, 1, num_actions, using_batch_norm=True).to(self.device)
        self.cheat_sheet = CheatingSheet(load_path='networks/cheating_sheet.npy', lr=1e-4)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.logsigmoid = nn.LogSigmoid()

        self.use_batch = use_batch
        self.mini_batch = 12

        # Opponent Info
        self.opp_act_count = np.zeros((num_actions))
        self.opp_act_conditional = np.zeros(
            (num_actions))  # opponent action distribution conditional on my card is better.
        self.opp_act_history = []

        # Total timesteps
        self.total_t = 0
        # Total training step
        self.train_t = 0

        self.memory = Memory(replay_memory_size, batch_size)
        self.replay_memory_init_size = replay_memory_init_size
        self.train_every = train_every
        self.num_actions = num_actions
        self.miniepoch = 5
        self.epsilon = 0.1
        self.batch_size = batch_size
        self.reward_normalize = True

        self.use_raw = False

        self.card2idx = {"SA": 0, "S2": 1, "S3": 2, "S4": 3, "S5": 4, "S6": 5, "S7": 6, "S8": 7, "S9": 8, "ST": 9,
                         "SJ": 10, "SQ": 11, "SK": 12, \
                         "HA": 13, "H2": 14, "H3": 15, "H4": 16, "H5": 17, "H6": 18, "H7": 19, "H8": 20, "H9": 21,
                         "HT": 22, "HJ": 23, "HQ": 24, "HK": 25, \
                         "DA": 26, "D2": 27, "D3": 28, "D4": 29, "D5": 30, "D6": 31, "D7": 32, "D8": 33, "D9": 34,
                         "DT": 35, "DJ": 36, "DQ": 37, "DK": 38, \
                         "CA": 39, "C2": 40, "C3": 41, "C4": 42, "C5": 43, "C6": 44, "C7": 45, "C8": 46, "C9": 47,
                         "CT": 48, "CJ": 49, "CQ": 50, "CK": 51}
        self.str2action = {'call': 0, 'raise': 1, 'fold': 2, 'check': 3}

        self.action_distribution = np.zeros((self.num_actions))

        self.loss1_list = []
        self.loss2_list = []
        self.reward_list = []

    def update_oppo(self, my_state, opp_state, opp_act):
        if opp_state is None or opp_act is None:
            return
        if type(opp_act) == str:
            actions = ['call', 'raise', 'fold', 'check']
            opp_act = actions.index(opp_act)
        opp_action = opp_act
        hand1 = my_state['raw_obs']['hand']
        hand2 = opp_state['raw_obs']['hand']
        public = my_state['raw_obs']['public_cards']
        self.opp_act_count[opp_action] += 1
        if self.compare_quality(hand1, hand2, public):
            self.opp_act_conditional[opp_action] += 1

    def compare_quality(self, hand1, hand2, public):
        '''
        Return:
            True if Q(hand1) > Q(hand2), else False
        '''
        level1 = self.cheat_sheet.state_process.State2Class(hand1, public)
        level2 = self.cheat_sheet.state_process.State2Class(hand2, public)
        if level1 == level2:
            return np.random.randint(0, 2) < 1
        return level1 < level2  # larger level means worse card quality

    def get_state_value(self, state):
        value = []
        for i in range(self.num_actions):
            value.append(self.cheat_sheet.get_value(state, i))
        return np.array(value)

    def feed(self, ts, opp_state=None, opp_act=None):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)
        if done:
            self.reward_list.append(reward)

        self.update_oppo(state, opp_state, opp_act)
        if opp_act is None:
            opp_act = -1

        self.feed_memory(state, action, reward, next_state, list(next_state['legal_actions'].keys()), done, \
                         opp_act, self.get_state_value(state), self.get_P(state, opp_act))
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size * 4

        self.action_distribution[action] += 1
        if self.memory.is_ready() and tmp % self.train_every == 0:
            self.train()
        return self.action_distribution

    def feed_memory(self, state, action, reward, next_state, legal_actions, done, opp_act, value, p):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done, opp_act, value, p)

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        _, action_batch, reward_batch, _, _, _, _, value_batch, p_batch = self.memory.sample()

        batch = self.trans2tensor({"action": action_batch, "reward": reward_batch, "value": value_batch, "p": p_batch})

        if self.reward_normalize:
            r = batch["reward"]
            batch["reward"] = (r - r.mean()) / (r.std() + 1e-8)

        for _ in range(self.miniepoch):
            self.update(batch)

        self.train_t += 1

    def update(self, batch):

        mini_batch = self.mini_batch
        self.policy.train()
        idx_list = list([i for i in range(self.batch_size)])
        random.shuffle(idx_list)

        for i in range(self.batch_size // mini_batch):
            a = torch.tensor(batch['action'][idx_list[i * mini_batch:(i + 1) * mini_batch]], dtype=torch.int64)
            r = batch["reward"].reshape(-1, 1)[idx_list[i * mini_batch:(i + 1) * mini_batch]]
            value = batch["value"][idx_list[i * mini_batch:(i + 1) * mini_batch]]
            p = batch["p"][idx_list[i * mini_batch:(i + 1) * mini_batch]].reshape(-1, 1)
            p_out = self.policy(p) * 0.01
            dist = nn.functional.softmax(p_out, dim=-1).reshape(-1, self.num_actions)  # (B,4)
            a_prob = p_out.gather(1, a.view(-1, 1))
            value_dist = nn.functional.softmax(value, dim=-1).reshape(-1, self.num_actions)
            dist = dist * value_dist
            dist = dist / (torch.sum(dist, dim=-1).reshape(-1, 1))
            total_reward = torch.sum(dist * value + r * a_prob * 0.1)

            loss1 = nn.functional.l1_loss(value_dist, dist)
            loss2 = - total_reward * 0.1
            self.loss1_list.append(loss1.item())
            self.loss2_list.append(loss2.item())
            loss = loss1 + loss2

            self.opt_policy.zero_grad()
            loss.backward()  # 计算策略网络的梯度
            self.opt_policy.step()

    def trans2tensor(self, batch):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device=self.device)
            elif isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.cat(batch[k]).to(device=self.device)
            else:
                batch[k] = torch.tensor(batch[k], device=self.device, dtype=torch.float32)

        return batch

    def get_P(self, state, opp_action):
        '''
        state: oringinal state from env
        opt_action: (int), the action of opponent in last step
        return: the probability that agent has better card than opponent condition on  state and act
        '''
        hand, public = state['raw_obs']['hand'], state['raw_obs']['public_cards']
        p_hand = self.has_better_card(hand, public)

        if opp_action == None:
            return p_hand
        if isinstance(opp_action, str):
            opp_action = self.str2action[opp_action]
        total_act = np.sum(self.opp_act_count)
        total_act_cond = np.sum(self.opp_act_conditional)
        if total_act == 0:
            p_opp_act = 1
        else:
            p_opp_act = self.opp_act_count[opp_action] / total_act

        if total_act_cond == 0:
            p_opp_cond = 1
        else:
            p_opp_cond = self.opp_act_conditional[opp_action] / total_act_cond
        return p_opp_cond * p_hand / (p_opp_act + 1e-9)

    def has_better_card(self, hand, public_card=[]):
        '''
        hand: my hand card
        Return: the probability that I have better card quality
        '''
        better_times = 0
        total_time = 10
        for _ in range(total_time):
            myhand = hand.copy()
            public = public_card.copy()
            seen_card = myhand.copy()
            seen_card.extend(public)
            self.init_hand(seen_card, public, length=5)
            seen_card = public.copy()
            seen_card.extend(myhand)
            my_level = self.cheat_sheet.state_process.State2Class(myhand, public)

            for _ in range(total_time):
                opp_hand = []
                self.init_hand(seen_card, hand=opp_hand)
                opp_level = self.cheat_sheet.state_process.State2Class(opp_hand, public)
                if my_level < opp_level:
                    better_times += 1
                if my_level == opp_level:
                    better_times += 0.5

        return better_times / total_time ** 2

    def init_hand(self, seen_card=[], hand=[], length=2):
        '''
        append card in hand
        '''
        all_card = seen_card.copy()
        while len(hand) < length:
            card = self.cheat_sheet.state_process.deal_card(all_card)
            hand.append(self.card_reverse(str(card)))
            all_card.append(self.card_reverse(str(card)))

    def card_reverse(self, card):
        str_card = str(card)
        r, s = str_card[0], str_card[1]
        return s + r

    def plot_curve(self, ):
        np.save('loss1.npy', self.loss1_list)
        np.save('loss2.npy', self.loss2_list)
        np.save('reward.npy', self.reward_list)

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        # action_record
        best_action, dist = self.predict(state, scale_factor=5.0)
        info = {'values': {state['raw_legal_actions'][i]: float(dist[list(state['legal_actions'].keys())[i]])
                           for i in range(len(state['legal_actions']))}}

        return best_action, info

    def predict(self, state, scale_factor=1.0):
        ''' Predict the masked Q-values
        Args:
            state (numpy.array): current state

        Returns:
            action: int
            dist_: distribution of policy
        '''
        if len(state['action_record']) == 0:
            opp_action = None
        else:
            opp_action = state['action_record'][-1][1]
        p = self.get_P(state, opp_action)

        value_dist = []
        for i in range(self.num_actions):
            value_dist.append(self.cheat_sheet.get_value(state, i))
        s = [p]
        s = torch.from_numpy(np.array(s, dtype=np.float32), ).to(self.device)
        value_dist = torch.from_numpy(np.array(value_dist, dtype=np.float32), ).to(self.device) * scale_factor
        with torch.no_grad():
            self.policy.eval()
            p_out = self.policy(s.unsqueeze(0))

            dist = nn.functional.softmax(p_out, dim=-1).reshape(-1, self.num_actions)
            value_dist = nn.functional.softmax(value_dist, dim=-1).reshape(-1, self.num_actions)
            dist = dist * value_dist
            dist_ = dist / torch.sum(dist, dim=-1)

        dist_ = dist.cpu().numpy().reshape(-1)

        masked_dist = np.zeros(self.num_actions, dtype=float)
        legal_actions = list(state['legal_actions'].keys())
        masked_dist[legal_actions] = dist_[legal_actions]
        masked_dist = masked_dist / np.sum(masked_dist)

        action = np.random.choice(len(masked_dist), 1, p=masked_dist).item()
        return action, masked_dist

    def step(self, state):
        ''' Predict the action for genrating training data but
            have the predictions disconnected from the computation graph

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
        '''
        action, _ = self.predict(state)
        return action


class Network(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, hidden_size, input_size, output_size, layer_num=2, using_batch_norm=False):
        ''' Initialize the Q network

        Args:
            num_actions (int): number of legal actions
            state_shape (list): shape of state tensor
            mlp_layers (list): output size of each fc layer
        '''
        super(Network, self).__init__()

        layers = [nn.Flatten()]
        last_size = input_size
        for i in range(layer_num - 1):
            layers.append(torch.nn.Linear(last_size, hidden_size))
            if using_batch_norm:
                layers.append(torch.nn.BatchNorm1d(hidden_size))
            layers.append(torch.nn.ReLU())
            last_size = hidden_size
        layers.append(torch.nn.Linear(last_size, output_size))
        self._net = torch.nn.Sequential(*layers)
        self.flatten = nn.Flatten()

    def forward(self, s):
        ''' Predict action values

        Args:
            s  (Tensor): (batch, state_shape)
        '''
        return self._net(s)


class Memory(object):
    ''' Memory for saving transitions
    '''

    def __init__(self, memory_size, batch_size, action_num=4):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = [[] for _ in range(action_num)]
        self.action_num = action_num

    def is_ready(self, ):
        for i in range(self.action_num):
            if len(self.memory[i]) < self.memory_size:
                return False
        return True

    def save(self, state, action, reward, next_state, legal_actions, done, opp_act, value, p):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory[action]) == self.memory_size:
            self.memory[action].pop(0)
        transition = Transition(state, action, reward, next_state, legal_actions, done, opp_act, value, p)
        self.memory[action].append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = []
        for i in range(self.action_num):
            samples.extend(random.sample(self.memory[i], self.batch_size // 4))
        random.shuffle(samples)

        return map(np.array, zip(*samples))