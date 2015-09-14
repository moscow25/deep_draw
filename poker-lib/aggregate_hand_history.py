import sys
import csv
import numpy as np
from poker_lib import *
from poker_util import *

"""
Author: Nikolai Yakovenko

Simply take in CSV of hand results.
Read line by line. Skip any bad lines.
Track results per player, by examining margin_result from the blinds.
Print result to screen.
"""

TRIPLE_DRAW_EVENT_HEADER = ['hand', 'draws_left', 'best_draw', 'hand_after',
                            'bet_model', 'value_heuristic', 'position',  'num_cards_kept', 'num_opponent_kept',
                            'action', 'pot_size', 'bet_size', 'pot_odds', 'bet_this_hand',
                            'actions_this_round', 'actions_full_hand', 
                            'total_bet', 'result', 'margin_bet', 'margin_result',
                            'current_margin_result', 'future_margin_result',
                            'oppn_hand', 'current_hand_win',
                            'hand_num', 'running_average', 'bet_val_vector', 'act_val_vector', 'num_draw_vector']
csv_key_map = CreateMapFromCSVKey(TRIPLE_DRAW_EVENT_HEADER)

if __name__ == '__main__':
    filename = sys.argv[1]	

    players = set([])
    player_results_map = {}

    # Read filename
    csv_reader = csv.reader(open(filename, 'rb'), lineterminator='\n')
    for line in csv_reader:
        # print(line)
        try:
            name = line[csv_key_map['bet_model']]
            action = line[csv_key_map['action']]
            margin_result = float(line[csv_key_map['margin_result']])

            if action in set(['pos_BB', 'pos_SB']):
                #print('------------')
                #print([name, action, margin_result])
                #print('------------')

                if not(name in players):
                    players.add(name)
                    player_results_map[name] = [];
                player_results_map[name].append(margin_result)
                    
        except (_csv.Error, TypeError, IndexError, ValueError, KeyError, AssertionError):
            print('bad line: %s' % line);

    # Now iterate over players, and print results
    for player in players:
        results = np.array(player_results_map[player])
        print('------------')
        print('%d\tResults for player |%s|' % (results.size, player))
        print('\tnum: %d\tave: %.2f\tstdv: %.2f' % (results.size, np.mean(results), np.std(results)))
