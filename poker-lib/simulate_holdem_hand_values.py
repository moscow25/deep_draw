import sys
import csv
import logging
import time
import math
import re
import random
import numpy as np
from matplotlib import pyplot as plt # for plotting bar graphs, etc
import scipy.stats as ss
from poker_lib import *
from holdem_lib import *
from poker_util import *

"""
Similar to simulate_holdem_values, hit "river values" for a given hand. 

Related to the CFR work, we simply want to count/estimate how hand performs against random opponents. 

Example: 44 preflop: how many rivers are we 90%? 20%? 40% against random hand.
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

# Hard-coded for preflop hands (simplest to understand)
# A. Create deck 
# B. Fix preflop cards we need
# C. Empty board, preflop hand
# D. 1000x times: deal a board, simulate allin value, save, backup & shuffle
# NOTE: Simulation takes place on the river. We want a histogram of how (preflop) hand performs over different board runouts.
def collect_river_values(sample_size=1000, tries_per_draw=1000, csv_writer=None):
    
    print('\n-- Dealing out fixed hand %s --\n' % 'JsTs')

    deck = PokerDeck(shuffle=True)
    # Now fix the top of the deck, for cards we need...
    c1 = deck.remove_card(Card(suit=SPADE, value=Jack))
    c2 = deck.remove_card(Card(suit=SPADE, value=Ten))
    dealer_round = RIVER_ROUND # Always deal to the river before evaluation
    #print deck
    community_hand = HoldemCommunityHand()
    holdem_hand = HoldemHand(community = community_hand)
    #deal_cards = deck.deal(2)
    deal_cards = [c1, c2]            
    holdem_hand.deal(deal_cards)

    # Our phantom opponent. Don't give him cards yet.
    opponent_hand = HoldemHand(community = community_hand)
    
    # OK, now we have the hand, and the deck.
    # Deal to river, evaluate, record, rewind, repeat.
    river_averages = []
    for round in range(sample_size):

        print('\n-- River Round %d --\n' % round)

        # Deal the hand, then rewind to the correct position.
        deck.shuffle()
        community_hand.deal(deck=deck, runway=True)
        community_hand.rewind(deck=deck, round=dealer_round)
        holdem_hand.evaluate()
        print holdem_hand

        hand_results = []
        category_results = [0.0 for category in HIGH_HAND_CATEGORIES]
        for i in range(tries_per_draw):
            #print '\nrewind (%s)...\n' % i

            # Return opponent cards, and river/turn cards if needed (default is full boards)
            community_hand.rewind(deck=deck, round=dealer_round)
            deck.return_cards(opponent_hand.dealt_cards, shuffle=False)
            deck.shuffle()

            # Deal to the end of the hand, and evaluate.
            community_hand.deal(deck=deck, runway=True)
            holdem_hand.evaluate()

            # Deal a random opponent hand. See how we compare
            opponent_hand = HoldemHand(cards = deck.deal(2), community = community_hand)
            opponent_hand.evaluate()
        
            #print holdem_hand
            #print('\tvs')
            #print opponent_hand

            if holdem_hand.rank > opponent_hand.rank:
                result = 0.0
            elif holdem_hand.rank < opponent_hand.rank:
                result = 1.0
            else:
                result = 0.5
            hand_results.append(result)

            # Record the category of our hand
            category = holdem_hand.category
            category_index = high_hand_categories_index[category]
            category_results[category_index] += 1.0
        
        # We finished given flop. Display, save.
        print('--> Average values: %.4f %s' % (np.mean(hand_results), hand_results[0:10]))

        #print (str(holdem_hand), np.mean(hand_results), [[categoryName[category], category_results[high_hand_categories_index[category]] / tries_per_draw] for category in HIGH_HAND_CATEGORIES])
        average_result = np.mean(hand_results)
        category_values = [[category, category_results[high_hand_categories_index[category]] / tries_per_draw] for category in HIGH_HAND_CATEGORIES]

        # TODO: Save to disk! 
        if csv_writer:
            hand_csv_row = output_full_sim_csv(poker_hand=holdem_hand, result=average_result, category_values=category_values, 
                                               header_map=csv_header_map, sample_size=tries_per_draw)
            csv_writer.writerow(hand_csv_row)
        

        # TODO: Save average value to graph in a matplotlib...
        river_averages.append(average_result)

        # Finish by rewinding the deck & shuffle.
        community_hand.rewind(deck=deck, round=PREFLOP_ROUND)
        deck.shuffle()

    print('finished %d rivers... results already saved.' % sample_size)
    print('\tmean: %.4f\tmedian: %.4f\tstdev: %.4f' % (np.mean(river_averages), np.median(river_averages), np.std(river_averages)))
    river_averages.sort()
    #print(river_averages)


    # Now, turn into bar graphs [0.0, 0.05, 0.1 ... 1.0]
    # x_buckets = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    x_buckets = [(x+1)/20.0 for x in xrange(20)]
    
    # Numpy operation for "nearest integer"
    river_averages_buckets = np.rint(np.array(river_averages) * 20.0) / 20.0
    #print(river_averages_buckets)
    
    # Turn these into bucket counts.
    y_bucket_counts = [0 for x in x_buckets]
    for bucket_val in river_averages_buckets:
        if bucket_val == 0.0:
            bucket_val = 1/20.0
        y_bucket_counts[x_buckets.index(bucket_val)] += 1
    y_bucket_averages = [float(x)/sample_size for x in y_bucket_counts]

    # Now, do a matplotlib
    allin_bars = plt.figure()
    ax = plt.subplot(111)
    #ax.bar( , , width=100)

    # Graphs all points. As true values, or corresponding buckets
    #ax.bar(range(len(river_averages)), river_averages, align='center')
    #ax.bar(range(len(river_averages_buckets)), river_averages_buckets, align='center')

    # Graph bucket counts, instead
    #ax.bar(range(len(x_buckets)), y_bucket_counts, align='center')
    #ax.bar(range(20), y_bucket_averages, align='center')

    N = len(y_bucket_averages)
    ind = np.arange(N)  # the x locations for the groups
    #width = 0.35       # the width of the bars
    rects1 = ax.bar(ind, y_bucket_averages, color='r')
    ax.set_ylim([0.0,0.20])

    #ax.set_xticks(ind+width)
    ax.set_xticklabels( ('', 0.25, 0.5, 0.75, 1.0) ) # no 0.0 label

    # Text labels...
    allin_bars.suptitle('Probability Distribution: JsTs', fontsize=20)
    plt.xlabel('Hand Strength', fontsize=20)
    plt.ylabel('Probability', fontsize=20)

    plt.show()

# TODO: Declare a hand (in code), run allin river values, save to CSV, and graph to matplotlib
if __name__ == '__main__':
    # TODO: Set from command line
    samples = 1000
    tries_per_draw = 200
    output_file_name = '%d-full-boards.csv' % samples

    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
	# Don't print header
        csv_writer.writerow(POKER_FULL_SIM_HEADER)
        csv_header_map = CreateMapFromCSVKey(POKER_FULL_SIM_HEADER)
    else:
        csv_writer = None

    collect_river_values(sample_size=samples, tries_per_draw=tries_per_draw, csv_writer=csv_writer)

    
