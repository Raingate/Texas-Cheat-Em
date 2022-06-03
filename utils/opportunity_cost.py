def card_reverse(card):
    str_card = str(card)
    r, s = str_card[0], str_card[1]
    return s + r


def get_cost(env, my_idx=0):
    public_card = env.get_perfect_information()['public_card']
    if public_card == None:
        public_card = []
    while len(public_card) < 5:
        public_card.append(card_reverse(env.game.dealer.deal_card()))

    player1_hand_card = env.get_perfect_information()['hand_cards'][my_idx]
    player2_hand_card = env.get_perfect_information()['hand_cards'][1 - my_idx]

    player1_hand_card.extend(public_card)
    player2_hand_card.extend(public_card)

    cost = get_start_value(player1_hand_card) - get_start_value(player2_hand_card)

    cost = cost if cost < 0 else 0.0  # cost cliping, only for bad case
    return cost


def get_start_value(all_card):
    """
    input: all_card(), card for one player, include the hand and public card
    return: value(float), the estimate value for game when starting
    """
    category2count = {
        '9': 40, '8': 624, '7': 3744, '6': 5108, '5': 10200, '4': 54912, '3': 123552, '2': 1098240, '1': 1302540,
        '0': 2598960
    }
    category2AccumCount = category2count.copy()

    for i in range(2, 10):
        category2AccumCount[str(i)] += category2AccumCount[str(i - 1)]

    # print(category2AccumCount)
    SCALE = 1.25

    value_estimator.all_cards = all_card
    value_estimator.evaluateHand()
    level = value_estimator.category
    if level >= 3:
        value = category2AccumCount[str(level)]
    elif level == 2:
        tmp_dict = {}
        pair_rank = -1
        for card in all_card:
            if card[1] not in tmp_dict:
                tmp_dict[card[1]] = 1
            else:
                pair_rank = card[1]
                pair_rank = value_estimator.STRING_TO_RANK[pair_rank] - 1
        value = category2AccumCount['1'] + category2count['2'] * pair_rank / 13.0
    elif level == 1:
        largest_rank = value_estimator.all_cards[-1][1]
        largest_rank = value_estimator.STRING_TO_RANK[largest_rank] - 8
        value = largest_rank * category2count['1'] / 5.0

    value = value / category2count['0'] * SCALE
    # print(all_card,value)
    return value



