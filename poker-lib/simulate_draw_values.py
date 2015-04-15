import sys
import csv
import logging
import time
import math
import re
import random
import numpy as np
import scipy.stats as ss
from poker_lib import *
from poker_util import *

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing basic video poker, and training/generating trianing data, for a neural network AI to learn how to play such games.

Hat tip to Erik of DeepPink, which implements a Theano system for chess.
https://github.com/erikbern/deep-pink

Basic components:
- Card, Deck, PokerHand classes
- shuffle, deal, draw operations
- evaulate a poker hand [imports C-lang methods]
- assign rewards to hands

I/O components:
- In this system, we compute "truth" for a given hand, based on all 32 possible draw combinations.
- Of course we could look up the "right move," but that's not what we are doing here.
- Instead, we compute an average return for every possible draw. 
...which is what we want the AI to learn.

Because you see.. it's not about making the best move with each draw. It's about optimizing value, and learning important patterns that create value, within a system. 
""" 

# All the data to store, from a hand simulation
POKER_FULL_SIM_HEADER = ['hand', 'best_value', 'best_draw', 'sample_size', 'pay_scheme']
for draw_pattern in all_draw_patterns:
    draw_to_string = '[%s]' % ','.join([str(i) for i in list(draw_pattern)])
    POKER_FULL_SIM_HEADER.append('%s_value' % draw_to_string)
    POKER_FULL_SIM_HEADER.append('%s_draw' % draw_to_string)

print POKER_FULL_SIM_HEADER

# Save a fully simulated hand, in the above format!
def output_full_sim_csv(poker_hand, header_map, sample_size):
    # Collect all values we may want to output
    output_map = {}
    output_map['hand'] = hand_string(poker_hand.dealt_cards)
    output_map['sample_size'] = sample_size
    output_map['pay_scheme'] = 'jacks_or_better_976_9_6'
    output_map['best_value'] = poker_hand.best_result.average_value
    output_map['best_draw'] = hand_string(poker_hand.best_result.draw_cards)

    # Each draw output
    for i in range(len(all_draw_patterns)):
        draw_pattern = all_draw_patterns[i]
        draw_result = poker_hand.sim_results[i]
        draw_to_string = '[%s]' % ','.join([str(i) for i in list(draw_pattern)])
        output_map['%s_value' % draw_to_string] = draw_result.average_value
        output_map['%s_draw' % draw_to_string] = hand_string(draw_result.draw_cards)
        
    # Place values in correct order
    output_row = VectorFromKeysAndSparseMap(keys=header_map, sparse_data_map=output_map, default_value = '')
    return output_row


# Try every possible draw combination... X times. Save averages.
# Output... average value of the best move.
#
# A. Create a deck, shuffle & deal hand.
# B. Compute average values for each discard arrangement
#   - for each discard arrangement, try X times
#       - draw cards, evaluate hand
#       - return cards, draw again
#       - save values & averages
# C. Output value of best average
def game_full_sim(round, tries_per_draw):

    print '\n-- New Round %d --\n' % round

    deck = PokerDeck(shuffle=True)
    #print deck
    draw_hand = PokerHand()
    deal_cards = deck.deal(5)
    draw_hand.deal(deal_cards)
    print draw_hand

    # Now, have the hand simulate simulate every possible draw, and record results.
    # NOTE: Don't copy the deck!
    cashier = JacksOrBetter() # "976-9-6" Jacks or Better -- with 100% long-term payout.
    draw_hand.simulate_all_draws(deck=deck, tries=tries_per_draw, payout_table=cashier)

    print draw_hand

    # What's the average payout, for the best move?
    pay_him = draw_hand.best_result.average_value

    return (draw_hand, pay_him)


# Play a number of hands. For each hand, try every possible draw X times, save as output
def generated_cases(sample_size, tries_per_draw, output_file_name):
    round = 0
    start_time = time.time()
    short_results = []
    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(POKER_FULL_SIM_HEADER)
        csv_header_map = CreateMapFromCSVKey(POKER_FULL_SIM_HEADER)
    else:
        csv_writer = None

    while round < sample_size:
        hand, payout = game_full_sim(round, tries_per_draw)
        short_results.append([hand_string(hand.best_result.draw_cards), payout])

        # Save hand to CSV, if output supplied.
        if csv_writer:
            hand_csv_row = output_full_sim_csv(poker_hand=hand, header_map=csv_header_map, sample_size=tries_per_draw)
            csv_writer.writerow(hand_csv_row)

            # Hack, to show matrix for final hand.
            #print hand_to_matrix(hand.final_hand)
            pretty_print_hand_matrix(hand.final_hand)

        round += 1
        end_round_time = time.time()
        #sys.exit(-3)

        print '%d rounds took %.1f seconds' % (round, end_round_time - start_time)

    if csv_writer:
        print '\nwrote %d rows' % len(results)
        output_file.close()

    print short_results
    result_values = [r[1] for r in short_results]
    print '\naverage return: %.2f\tmax return: %.1f' % (np.mean(result_values), max(result_values))

if __name__ == '__main__':
    samples = 20000
    tries_per_draw = 1000 
    output_file_name = '%d_full_sim_samples.csv' % samples
    # TODO: Take num samples from command line.
    generated_cases(sample_size=samples, tries_per_draw=tries_per_draw, output_file_name=output_file_name)
