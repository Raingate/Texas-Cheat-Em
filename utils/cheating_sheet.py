import numpy as np
import random
from utils.handestimator import HandEstimator
from utils.opportunity_cost import card_reverse
from rlcard.games.base import Card


class StateClassification(object):
    def __init__(self, env, class_num=2):
        self.class_num = class_num
        self.RANK_TO_STRING = {2: "2", 3: "3", 4: "4", 5: "5", 6: "6",
                               7: "7", 8: "8", 9: "9", 10: "T", 11: "J", 12: "Q", 13: "K", 14: "A"}
        self.STRING_TO_RANK = {v: k for k, v in self.RANK_TO_STRING.items()}
        self.RANK_LOOKUP = "23456789TJQKA"
        self.env = env
        self.desk = []
        self.value_estimator = HandEstimator(all_cards=None)

    def init_standard_deck(self):
        ''' Initialize a standard deck of 52 cards

        Returns:
            (list): A list of Card object
        '''
        suit_list = ['S', 'H', 'D', 'C']
        rank_list = ['A', '2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K']
        self.desk = [Card(suit, rank) for suit in suit_list for rank in rank_list]
        random.shuffle(self.desk)

    def deal_card(self, all_card=None):
        if all_card is None:
            all_card = []
        if len(self.desk) == 0:
            self.init_standard_deck()
        card = self.desk.pop(0)
        while card_reverse(card) in all_card:
            if len(self.desk) == 0:
                self.init_standard_deck()
            card = self.desk.pop(0)
        return card

    def good_or_bad(self, hand):
        ranks = []
        for card in hand:
            ranks.append(int(self.STRING_TO_RANK[card[1]]))
        if ranks[0] == ranks[1]:
            return True
        if max(ranks) < 11:
            return False
        return True

    def State2Class(self, hand, public=None):
        '''
        hand: list of string, eg ['D4', 'HK']
        '''

        # Only have hand card, class_id in [20: 45]
        if public is None:
            public = []
        if len(public) == 0:  # [20,32]
            if hand[0][1] == hand[1][1]:
                pair_rank = hand[0][1]
                pair_rank = self.STRING_TO_RANK[pair_rank]
                class_id = 34 - pair_rank
            else:  # [33,44]
                rank1, rank2 = hand[0][1], hand[1][1]
                rank1, rank2 = self.STRING_TO_RANK[rank1], self.STRING_TO_RANK[rank2]
                single_rank = max(rank1, rank2)
                class_id = 47 - single_rank
            return class_id

        public_card = public.copy()
        all_card = hand.copy()
        all_card.extend(public_card)
        while len(all_card) < 7:
            card = self.deal_card(all_card)
            all_card.append(card_reverse(card))

        all_card = self.sort_cards(all_card)
        self.value_estimator.all_cards = all_card
        self.value_estimator.evaluateHand()
        level = self.value_estimator.category

        if level >= 4:
            class_id = 0
        elif level == 3:
            class_id = 1
        elif level == 2:  # class_id in [2, 14]
            tmp_dict = {}
            pair_rank = -1
            for card in all_card:
                if card[1] not in tmp_dict:
                    tmp_dict[card[1]] = 1
                else:
                    pair_rank = card[1]
                    pair_rank = self.value_estimator.STRING_TO_RANK[pair_rank]
            class_id = 16 - pair_rank
        elif level == 1:  # class_id in [15, 19]
            largest_rank = self.value_estimator.all_cards[-1][1]
            largest_rank = self.value_estimator.STRING_TO_RANK[largest_rank]
            class_id = 28 - largest_rank

        return class_id

    def sort_cards(self, cards):
        '''
        Sort all the seven cards ascendingly according to RANK_LOOKUP
        '''
        cards = sorted(
            cards, key=lambda card: self.RANK_LOOKUP.index(card[1]))
        return cards


class CheatingSheet(object):
    def __init__(self, env=None, class_num=45, action_num=4, load_path='', scale_factor=1.0, normalize=True,
                 lr=1e-4):
        self.class_num = class_num
        self.action_num = action_num
        self.value_table = np.zeros((class_num, action_num))
        self.state_process = StateClassification(env=env)
        self.lr = lr
        if len(load_path) != 0:
            self.value_table = np.load(load_path)
            print(f'Load Value Table from {load_path}')
            if normalize:
                self.value_table = (self.value_table - np.mean(self.value_table, axis=1).reshape(-1, 1)) / (
                            np.std(self.value_table, axis=1).reshape(-1, 1) + 1e-8)
            self.value_table *= scale_factor

    def update(self, trajectory):
        if len(trajectory) == 0:
            return
        action_list = []
        state_class_list = []
        for ts in trajectory:
            (state, action, reward, next_state, done) = tuple(ts)
            action_list.append(action)
            hand, public = state['raw_obs']['hand'], state['raw_obs']['public_cards']
            state_class_list.append(self.state_process.State2Class(hand, public))
        reward = trajectory[-1][2]
        random_num = np.random.randint(0, len(trajectory))
        target_action = action_list.pop(random_num)
        target_class = state_class_list.pop(random_num)
        old_target_value = self.value_table[target_class][target_action]
        new_target_value = reward
        for i in range(len(action_list)):
            new_target_value -= self.value_table[state_class_list[i]][action_list[i]]
        self.value_table[target_class][target_action] += (new_target_value - old_target_value) * self.lr

    def print_table(self, ):
        print(self.value_table)

    def get_value(self, state, action):
        hand, public = state['raw_obs']['hand'], state['raw_obs']['public_cards']
        # If Only hand card, read from table
        if len(public) == 0:
            class_id = self.state_process.State2Class(hand, public)
            value = self.value_table[class_id][action]
            return value
        # else, use Monte Carlo
        value_list = []
        for _ in range(100):
            class_id = self.state_process.State2Class(hand, public)
            value_list.append(self.value_table[class_id][action])
        return np.mean(value_list)

    def save_model(self, save_path='table.npy'):
        np.save(save_path, self.value_table)