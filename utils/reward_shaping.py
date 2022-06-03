import copy


def reward_shaping(trajectories, big_blind=2, change_id=0):
    """

    :param change_id:
    :param trajectories: trajectories of games
    :param big_blind: big blind of the game
    :return: trajectories after reward shaping
    """

    winner_id = winner(trajectories)
    loser_id = 1 - winner_id
    first_id = trajectories[loser_id][0][0]['action_record'][0][0]

    neg_sum_reward, pos_sum_reward = 0, 0

    for t in range(len(trajectories[loser_id])):
        # not the last state
        if t != len(trajectories[loser_id]) - 1:
            action = trajectories[loser_id][t][1]

            if action == 0:  # call
                trajectories[loser_id][t][2] = -1

            elif action == 1:  # raise
                all_chips = trajectories[loser_id][t][0]['raw_obs']['all_chips']  # list of chips
                my_chips = trajectories[loser_id][t][0]['raw_obs']['my_chips']
                trajectories[loser_id][t][2] = -(max(all_chips) - my_chips + big_blind)/big_blind

            elif action == 3:  # check
                trajectories[loser_id][t][2] = 0

            neg_sum_reward += trajectories[loser_id][t][2]

        # last state
        else:
            trajectories[loser_id][t][2] -= neg_sum_reward

    for t in range(len(trajectories[winner_id])):
        # not the last state
        if t != len(trajectories[winner_id]) - 1:
            if first_id == winner_id:
                next_t = t
            else:
                next_t = t + 1

            trajectories[winner_id][t][2] = -trajectories[loser_id][next_t][2]
            pos_sum_reward += trajector