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
from holdem_lib import *
from poker_util import *

"""
Simulate Hold'em values. Idea is to collect random hands, and [0.0, 1.0] values for these hands, preflop, on the flop, on turn, and river.

Heuristic mapping for Hold'em is simple: win rate vs a random hand.

As much as possible, borrow from and combine with draw hands. In some cases, this will be silly, or not be possible.

For poker hand evaluation, keep simulation simple: deal to the end. For hand evaluation, keep it simple: try send to a function that automates trying every legal combination of cards. This will be different for Omaha Hold'em, and Omaha/8, but also simple to adjust at that point.
"""



# All the data to store, from a hand simulation
POKER_FULL_SIM_HEADER = ['game', 'hand', 'flop', 'turn', 'river', 'best_value', 'sample_size']
POKER_FULL_SIM_HEADER += [categoryName[category] for category in HIGH_HAND_CATEGORIES]
print POKER_FULL_SIM_HEADER

# Save a fully simulated hand, in the above format!
def output_full_sim_csv(poker_hand, result, category_values, header_map, sample_size):
    # Collect all values we may want to output
    output_map = {}
    output_map['game'] = 'texas'
    output_map['hand'] = hand_string(poker_hand.dealt_cards)
    output_map['flop'] = hand_string(poker_hand.community.flop)
    output_map['turn'] = hand_string(poker_hand.community.turn) 
    output_map['river'] = hand_string(poker_hand.community.river)
    output_map['sample_size'] = sample_size
    output_map['best_value'] = result

    # Averages for hand categories
    for category in HIGH_HAND_CATEGORIES:
        output_map[categoryName[category]] = category_values[high_hand_categories_index[category]][1]
        
    # Place values in correct order
    output_row = VectorFromKeysAndSparseMap(keys=header_map, sparse_data_map=output_map, default_value = '')
    return output_row


# Try a hand, and certain depth of dealing the community cards.
# 
# A. Create a deck, shuffle & deal hand.
# B. Deal a flop, turn or river if that depth is required.
# C. Deal the rest of the hands X times, including random opponent hand, and collect the average.
# D. Output value of best average.
def game_full_sim(round, tries_per_draw, dealer_round=random.choice(list(HOLDEM_ROUNDS_SET))): #PREFLOP_ROUND):

    print '\n-- New Round %d --\n' % round

    deck = PokerDeck(shuffle=True)
    #print deck
    community_hand = HoldemCommunityHand()
    holdem_hand = HoldemHand(community = community_hand)
    deal_cards = deck.deal(2)
    holdem_hand.deal(deal_cards)

    # Deal the hand, then rewind to the correct position.
    community_hand.deal(deck=deck, runway=True)
    community_hand.rewind(deck=deck, round=dealer_round)
    print holdem_hand

    # Our phantom opponent. Don't give him cards yet.
    opponent_hand = HoldemHand(community = community_hand)

    hand_results = []
    category_results = [0.0 for category in HIGH_HAND_CATEGORIES]
    for i in range(tries_per_draw):
        #print '\nrewind (%s)...\n' % i
        community_hand.rewind(deck=deck, round=dealer_round)
        deck.return_cards(opponent_hand.dealt_cards, shuffle=False)
        deck.shuffle()
        
        # Deal to the end of the hand, and evaluate.
        community_hand.deal(deck=deck, runway=True)
        holdem_hand.evaluate()

        # Deal a random opponent hand. See how we compare
        opponent_hand = HoldemHand(cards = deck.deal(2), community = community_hand)
        opponent_hand.evaluate()

        # TODO: Count winners & losers...
        if holdem_hand.rank > opponent_hand.rank:
            #print holdem_hand
            #print('\tloser')
            #print opponent_hand
            result = 0.0
        elif holdem_hand.rank < opponent_hand.rank:
            #print holdem_hand
            #print('\twinner')
            #print opponent_hand
            result = 1.0
        else:
            #print holdem_hand
            #print('\ttie-die')
            #print opponent_hand
            result = 0.5
        hand_results.append(result)
        #print('ave result: %.3f' % np.mean(hand_results))

        # Record the category of our hand
        category = holdem_hand.category
        category_index = high_hand_categories_index[category]
        category_results[category_index] += 1.0

    print '\nrewind (%s)...\n' % i
    community_hand.rewind(deck=deck, round=dealer_round)
    print holdem_hand

    print('final results vs random hand %.4f' % np.mean(hand_results))
    print[[categoryName[category], category_results[high_hand_categories_index[category]] / tries_per_draw] for category in HIGH_HAND_CATEGORIES] 
    #print [cat_result / tries_per_draw for cat_result in category_results]

    #sys.exit(-1)

    # Return the hand (including community cards link), average value against random hand, and average results for all categories...
    return (holdem_hand, np.mean(hand_results), [[category, category_results[high_hand_categories_index[category]] / tries_per_draw] for category in HIGH_HAND_CATEGORIES])

    """
    # Now, have the hand simulate simulate every possible draw, and record results.
    # NOTE: Don't copy the deck!
    cashier = JacksOrBetter() # "976-9-6" Jacks or Better -- with 100% long-term payout.
    draw_hand.simulate_all_draws(deck=deck, tries=tries_per_draw, payout_table=cashier, debug=False)

    #print draw_hand

    # What's the average payout, for the best move?
    pay_him = draw_hand.best_result.average_value

    return (draw_hand, pay_him)
    """


# Play a number of hands. For each hand, try every possible draw X times, save as output
def generated_cases(sample_size, tries_per_draw, output_file_name):
    round = 0
    start_time = time.time()
    short_results = []
    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
	# Don't print header
        csv_writer.writerow(POKER_FULL_SIM_HEADER)
        csv_header_map = CreateMapFromCSVKey(POKER_FULL_SIM_HEADER)
    else:
        csv_writer = None

    while round < sample_size:
        hand, average_result, category_values = game_full_sim(round, tries_per_draw, dealer_round=random.choice(list(HOLDEM_ROUNDS_SET)))
        short_results.append([hand_string(hand.dealt_cards), average_result])

        # Save hand to CSV, if output supplied.
        if csv_writer:
            hand_csv_row = output_full_sim_csv(poker_hand=hand, result=average_result, category_values=category_values, 
                                               header_map=csv_header_map, sample_size=tries_per_draw)
            csv_writer.writerow(hand_csv_row)

            # Hack, to show matrix for final hand.
            #print hand_to_matrix(hand.final_hand)
            #pretty_print_hand_matrix(hand.final_hand)

        round += 1
        end_round_time = time.time()
        #sys.exit(-3)

        print '%d rounds took %.1f seconds' % (round, end_round_time - start_time)

    if csv_writer:
        #print '\nwrote %d rows' % round
        output_file.close()

    #print short_results
    result_values = [r[1] for r in short_results]
    print '\naverage return: %.4f\tmax return: %.4f' % (np.mean(result_values), max(result_values))

if __name__ == '__main__':
    # TODO: Set from command line
    samples = 10000
    tries_per_draw = 2000

    # default 
    output_file_name = '%d_holdm_full_sim_samples.csv' % samples

    # Output filename if given
    # TODO: set via command line flagz
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
        output_file_name = filename

    print 'will save %d lines to %s' % (samples, output_file_name)

    # TODO: Take num samples from command line.
    generated_cases(sample_size=samples, tries_per_draw=tries_per_draw, output_file_name=output_file_name)



