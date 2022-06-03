import random
from matplotlib.cbook import flatten
import numpy as np
import torch
import torch.nn as nn
from rlcard.utils.utils import remove_illegal
from collections import namedtuple
import torch.nn.functional as F

Transition = namedtuple('Transition',
                        ['state', 'action', 'reward', 'next_state', 'legal_actions', 'done', "log_prob", 'adv'])


class PPOAgent(object):

    def __init__(self,
                 num_actions,
                 device,
                 hidden_size=64,
                 learning_rate=5e-5,
                 replay_memory_size=50,
                 batch_size=50,
                 replay_memory_init_size=50,
                 train_every=50,
                 use_batch=False,):
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.state_shape = int(72)  # 72+52 # np.prod(state_shape)
        self.policy = Network(hidden_size, self.state_shape, num_actions, using_batch_norm=use_batch).to(self.device)
        self.value = Network(hidden_size, self.state_shape, 1, using_batch_norm=use_batch).to(self.device)
        self.target_value = Network(hidden_size, self.state_shape, 1, using_batch_norm=use_batch).to(self.device)
        for target_param, param in zip(self.target_value.parameters(), self.value.parameters()):
            target_param.data.copy_(target_param.data * 0.0 + param.data * 1.0)
        self.opt_policy = torch.optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.opt_value = torch.optim.Adam(self.value.parameters(), lr=learning_rate * 10)
        self.logsigmoid = nn.LogSigmoid()

        self.use_batch = use_batch
        self.mini_batch = 1

        # Total timesteps
        self.total_t = 0
        # Total training step
        self.train_t = 0

        self.memory = Memory(replay_memory_size, batch_size)
        self.replay_memory_init_size = replay_memory_init_size
        self.train_every = train_every
        self.num_actions = num_actions
        self.gamma = 0.99
        self.miniepoch = 5
        self._clip_range = 0.2
        self._tau = 0.01
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

        self.action_distribution = np.zeros(self.num_actions)

    def state_represetation(self, state):
        """
        state: dict
        """
        if state['obs'].shape[0] == self.state_shape:
            return state
        new_representation = np.zeros((self.state_shape), dtype=np.float64)
        for card in state['raw_obs']['hand']:
            idx = self.card2idx[card]
            new_representation[idx] = 1

        for card in state['raw_obs']['hand']:
            idx = self.card2idx[card]
            new_representation[int(idx + 52)] = 1

        new_representation[104:] = state['obs'][52:]
        state['obs'] = new_representation
        return state

    def copy_network(self, src_network, tgt_network):
        for target_param, param in zip(tgt_network.parameters(), src_network.parameters()):
            target_param.data.copy_(param.data)

    def copy_parameters_to(self, ppo_agent):
        self.copy_network(self.policy, ppo_agent.policy)
        self.copy_network(self.value, ppo_agent.value)
        self.copy_network(self.target_value, ppo_agent.target_value)

    def feed(self, ts):
        ''' Store data in to replay buffer and train the agent. There are two stages.
            In stage 1, populate the memory without training
            In stage 2, train the agent every several timesteps

        Args:
            ts (list): a list of 5 elements that represent the transition
        '''
        (state, action, reward, next_state, done) = tuple(ts)

        s = torch.from_numpy(np.expand_dims(state['obs'], 0)).to(self.device).float()
        s1 = torch.from_numpy(np.expand_dims(next_state['obs'], 0)).to(self.device).float()
        self.policy.eval()
        p_out = self.policy(s)
        log_prob, _ = self.dist(p_out, action)
        log_prob = log_prob.detach().cpu().numpy()
        adv = self.compute_adv(s, action, reward, s1, done).detach().cpu().numpy()
        self.feed_memory(state['obs'], action, reward, next_state['obs'], list(next_state['legal_actions'].keys()),
                         done, log_prob, adv)
        self.total_t += 1
        tmp = self.total_t - self.replay_memory_init_size

        self.action_distribution[action] += 1
        if tmp >= 0 and tmp % self.train_every == 0:
            self.action_distribution = np.zeros((self.num_actions))
            self.train()

    def feed_memory(self, state, action, reward, next_state, legal_actions, done, log_prob, adv):
        ''' Feed transition to memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        self.memory.save(state, action, reward, next_state, legal_actions, done, log_prob, adv)

    def eval_step(self, state):
        ''' Predict the action for evaluation purpose.

        Args:
            state (numpy.array): current state

        Returns:
            action (int): an action id
            info (dict): A dictionary containing information
        '''
        best_action, dist = self.predict(state)
        info = {}
        info['values'] = {state['raw_legal_actions'][i]: float(dist[list(state['legal_actions'].keys())[i]]) for i in
                          range(len(state['legal_actions']))}

        return best_action, info

    def predict(self, state):
        ''' Predict the masked Q-values
        Args:
            state (numpy.array): current state

        Returns:
            action: int
            dist_: distribution of policy
        '''
        s = torch.from_numpy(np.expand_dims(state['obs'], 0)).to(self.device).float()

        with torch.no_grad():
            self.policy.eval()
            p_out = self.policy(s)
        dist = nn.functional.softmax(p_out, dim=-1).reshape(-1)
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
        if np.random.rand(1) < self.epsilon:
            action = np.random.randint(0, self.num_actions)
        return action

    def dist(self, p_out, action=None):
        dist = nn.functional.softmax(p_out, dim=-1).reshape(-1)

        if action is None:
            dist_ = list(dist.numpy().reshape(-1))
            dist_ = dist_ / np.sum(dist_)
            action = np.random.choice(len(dist_), 1, p=dist_)
        log_prob = torch.log(dist[action])
        return log_prob.reshape(-1, 1), torch.Tensor(action).reshape(-1, 1)

    def compute_adv(self, s, a, r, s1, done):
        gamma = self.gamma
        with torch.no_grad():
            self.value.eval()
            adv = r + gamma * (1 - done) * self.value(s1) - self.value(s)
        return adv

    def update(self, batch):
        gamma = self.gamma
        mini_batch = self.mini_batch
        self.policy.train()
        self.value.train()
        for i in range(self.batch_size // mini_batch):
            s = batch["state"][i * mini_batch:(i + 1) * mini_batch]
            a = torch.tensor(batch['action'][i * mini_batch:(i + 1) * mini_batch], dtype=torch.int64)
            r = batch["reward"].reshape(-1, 1)[i * mini_batch:(i + 1) * mini_batch]
            s1 = batch["next_state"][i * mini_batch:(i + 1) * mini_batch]
            adv = batch["adv"][i * mini_batch:(i + 1) * mini_batch].detach()
            done = batch["done"].reshape(-1, 1)[i * mini_batch:(i + 1) * mini_batch]
            old_log_prob = batch["log_prob"][i * mini_batch:(i + 1) * mini_batch].detach()

            td_target = r + gamma * self.target_value(s1) * (1 - done)
            policy_out = torch.softmax(self.policy(s), dim=1)
            log_probs = torch.log(policy_out.gather(1, a.view(-1, 1)))  ##
            probs_ratio = log_probs.exp() / (old_log_prob.exp() + 1e-8)

            loss1 = probs_ratio * adv
            loss2 = torch.clamp(probs_ratio, 1 - self._clip_range, 1 + self._clip_range) * adv
            actor_loss = -torch.mean(torch.min(loss1, loss2))
            critic_loss = torch.mean(F.mse_loss(td_target.detach(), self.value(s)))

            self.opt_policy.zero_grad()
            self.opt_value.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.opt_policy.step()
            self.opt_value.step()

        self.soft_update(self.value, self.target_value, self._tau)

    def soft_update(self, source, target, tau=None):
        if tau is None:
            tau = self._tau
        with torch.no_grad():
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def trans2tensor(self, batch):
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(device=self.device)
            elif isinstance(batch[k][0], torch.Tensor):
                batch[k] = torch.cat(batch[k]).to(device=self.device)
            else:
                batch[k] = torch.tensor(batch[k], device=self.device, dtype=torch.float32)

        return batch

    def train(self):
        ''' Train the network

        Returns:
            loss (float): The loss of the current batch.
        '''
        state_batch, action_batch, reward_batch, next_state_batch, legal_actions_batch, done_batch, log_prob_batch, adv_batch = self.memory.sample()

        batch = self.trans2tensor({"state": state_batch, "action": action_batch,
                                   "log_prob": log_prob_batch, "adv": adv_batch,
                                   "next_state": next_state_batch, "done": done_batch,
                                   "reward": reward_batch, })

        if self.reward_normalize:
            r = batch["reward"]
            batch["reward"] = (r - r.mean()) / (r.std() + 1e-8)

        for _ in range(self.miniepoch):
            self.update(batch)

        self.train_t += 1

    def set_device(self, device):
        self.device = device
        self.policy.device = device
        self.value.device = device
        self.target_value.device = device


class Network(nn.Module):
    ''' The function approximation network for Estimator
        It is just a series of tanh layers. All in/out are torch.tensor
    '''

    def __init__(self, hidden_size, input_size, output_size, layer_num=3, using_batch_norm=False):
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

    def __init__(self, memory_size, batch_size):
        ''' Initialize
        Args:
            memory_size (int): the size of the memroy buffer
        '''
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.memory = []

    def save(self, state, action, reward, next_state, legal_actions, done, log_prob, adv):
        ''' Save transition into memory

        Args:
            state (numpy.array): the current state
            action (int): the performed action ID
            reward (float): the reward received
            next_state (numpy.array): the next state after performing the action
            legal_actions (list): the legal actions of the next state
            done (boolean): whether the episode is finished
        '''
        if len(self.memory) == self.memory_size:
            self.memory.pop(0)
        transition = Transition(state, action, reward, next_state, legal_actions, done, log_prob, adv)
        self.memory.append(transition)

    def sample(self):
        ''' Sample a minibatch from the replay memory

        Returns:
            state_batch (list): a batch of states
            action_batch (list): a batch of actions
            reward_batch (list): a batch of rewards
            next_state_batch (list): a batch of states
            done_batch (list): a batch of dones
        '''
        samples = random.sample(self.memory, self.batch_size)
        return map(np.array, zip(*samples))
