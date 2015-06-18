"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing basic video poker, and training/generating trianing data, for a neural network AI to learn how to play such games.

Hat tip to Erik of DeepPink, which implements a Theano system for chess.
https://github.com/erikbern/deep-pink

Here, we use Lasagne, a set of tools & examples for setting up a neural net with Theano, to train the network to predict values of video poker draws.
https://github.com/Lasagne/Lasagne

We based the first version of this code, on the Lasagne MNIST example.

Rather than trying to predict the best move of a hand, say [Kh,4s,5s,Qd,5c], we output the average value, for a given (hidden) payout table, for each draw option:

All 32 sim results for hand [Kh,4s,5s,Qd,5c]:
	[]:	1000 sample:	0.34 average	9.00 maximum
	[Kh]:	10000 sample:	0.47 average	25.00 maximum
        ...
	[Kh,5s,Qd,5c]:	1000 sample:	0.38 average	3.00 maximum
	[4s,5s,Qd,5c]:	1000 sample:	0.41 average	3.00 maximum
	[Kh,4s,5s,Qd,5c]:	1000 sample:	0.00 average	0.00 maximum

best result:
	[5s,5c]:	1000 sample:	0.76 average	25.00 maximum
""" 

from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import theano.printing as tp # theano.printing.pprint(variable) [tp.pprint(var)]
import time
import math
import csv
from poker_lib import *
from poker_util import *

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = '' # 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = '../data/250k_full_sim_combined.csv' # full dataset, with preference to better data (more samples per point)
#'../data/100k-super_sim_full_vector.csv' # more data, still 5k samples per point; see if improvement on more inexact data?  
#'../data/40k-super_sim_full_vector.csv' # smaller data set, 5k samples per hand point 
#'../data/300k_full_sim_samples.csv' # big data set, 1k samples per generated hand point
# '../data/100k_full_sim_samples.csv' # '../data/20000_full_sim_samples.csv' # '../data/100k_full_sim_samples.csv' #'../data/40000_full_sim_samples.csv'
# Not too much accuracy gain... in doubling the training data. And more than 2x as slow.
# '../data/20000_full_sim_samples.csv'

# Default, if not specified elsewhere...
MAX_INPUT_SIZE = 250000 #100000 # 50000 # 200000 # 150000 # 1000000 #40000 # 10000000 # Remove this constraint, as needed
VALIDATION_SIZE = 2000
TEST_SIZE = 2000
NUM_EPOCHS = 100 # 20 # 100
BATCH_SIZE = 100 # 50 #100
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.1 # Try faster learning # 0.01
MOMENTUM = 0.9

# Do we include all data? Sure... but maybe not.
# If we keep everything, almost 60% of data falls into "keep 2" nodes. 
DATA_SAMPLING_KEEP_ALL = [1.0 for i in range(32)]
# Choose 50% of "keep two cards" moves, and 50% of "keep one card" moves
DATA_SAMPLING_REDUCE_KEEP_TWO = [1.0] + [0.5] * 5 + [0.5] * 10 + [1.0] * 10 + [1.0] * 5 + [1.0] 
DATA_SAMPLING_REDUCE_KEEP_TWO_W_EQUIVALENCES = [0.25] + [0.10] * 5 + [0.10] * 10 + [0.50] * 10 + [0.25] * 5 + [1.0]

# Focus on straights, flushes, draws and pat hands. [vast majority of cards option to 2-card draw anyway...]
DATA_SAMPLING_REDUCE_KEEP_TWO_FOCUS_FLUSHES = [0.50] + [0.25] * 5 + [0.25] * 10 + [0.7] * 10 + [1.0] * 5 + [1.0] 

# Pull levers, to zero-out some inputs... while keeping same shape.
# TODO: Move these into a "Default" class in separate file. Like here: https://github.com/spragunr/deep_q_rl/blob/lasagne_nature/deep_q_rl/run_nature.py
NUM_DRAWS_ALL_ZERO = False # True # Set true, to add "num_draws" to input shape... but always zero. For initialization, etc.
PAD_INPUT = True # False # Set False to handle 4x13 input. *Many* things need to change for that, including shape.
CONTEXT_LENGTH = 2 + 5 + 5 + 5 # Fast way to see how many zero's to add, if needed. [xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept]
CONTEXT_ALL_ZERO = False # True # Set true, to map "xPos, xPot, ..." to x0 of same length. Useful for getting baseline for hands, values first
CARDS_INPUT_ALL_ZERO = False # True # Set true, to map xCards to x0. Why? To train a decision baseline that has to focus on betting & pot odds for a minute.

# Bias training data? It's not just for single draw video poker..
# hold value == [0.0, 1.0] value of keep-all-five hand.
# Anything >= 0.5 is a really good pat hand
# Anything <= 0.075 is a pair, straight or flush...
SAMPLE_BY_HOLD_VALUE = True # Default == true, for all 32-length draws. As it focuses on cases that matter.

# If we load the input arrays without refactoring... might save memory.
TRAINING_INPUT_TYPE = theano.config.floatX # np.int32

# Value of a "zero event." Why baseline? Model doesn't really handle negative numbers!
EVENTS_VALUE_BASELINE = 2.000

# Keep less than 100% of deuce events, to cover more hands, etc. Currently events from hands are in order.
# TODO: Pre-compute numpy arrays, and train on more data. Not all in "shared," etc.
SAMPLE_RATE_DEUCE_EVENTS = 1.0 # 0.50 # 0.33

# Use this to train only on results of intelligent players, if different versions available
PLAYERS_INCLUDE_DEUCE_EVENTS = set(['CNN', 'man', 'sim']) # Incude 'sim' and ''?

# returns numpy array 5x4x13, for card hand string like '[Js,6c,Ac,4h,5c]' or 'Tc,6h,Kh,Qc,3s'
# if pad_to_fit... pass along to card input creator, to create 14x14 array instead of 4x13
def cards_inputs_from_string(hand_string, pad_to_fit = PAD_INPUT, max_inputs=50,
                             include_num_draws=False, num_draws=None, include_full_hand = False, include_hand_context = False):
    hand_array = hand_string_to_array(hand_string)

    # Now turn the array of Card abbreviations into numpy array of of input
    cards_array_original = [card_from_string(card_str) for card_str in hand_array]
    assert(len(cards_array_original) == 5)

    # If we also for "full hand", also include a 6th "card" that's matrix of the entire hand.
    # NOTE: We do *not* support permutations. If we want permutations... should permute this also.
    if include_full_hand:
        full_hand_to_matrix = hand_to_matrix(cards_array_original, pad_to_fit=pad_to_fit)
        #print(hand_string)
        #print(full_hand_to_matrix)

    # If we also adding "number of draws" as input, do so here.
    if include_num_draws:
        # Returns [card_3, card_2, card_1] as three fake cards, matching input format for actual cards.
        num_draws_input = num_draws_input_from_string(num_draws, pad_to_fit=pad_to_fit)
        assert len(num_draws_input) == 3, 'Incorrect length of input for number of draws %s!' % num_draws_input

    # All 4-24 of these permutations are equivalent data (just move suits)
    # Keep first X if so declared.
    if max_inputs > 1:
        cards_array_permutations = hand_suit_scrambles(cards_array_original)[0:max_inputs]
    else:
        # Shortcut if only one output...
        cards_array_permutations = [cards_array_original]

    cards_inputs_all = []
    for cards_array in cards_array_permutations:
        if include_num_draws:
            # NOTE: If doing permuations... full hand matrix will not match!
            if include_full_hand:
                cards_input = np.array([card_to_matrix(cards_array[0], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[1], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[2], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[3], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[4], pad_to_fit=pad_to_fit), full_hand_to_matrix, num_draws_input[0], num_draws_input[1], num_draws_input[2]], TRAINING_INPUT_TYPE)
            else:
                cards_input = np.array([card_to_matrix(cards_array[0], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[1], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[2], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[3], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[4], pad_to_fit=pad_to_fit), num_draws_input[0], num_draws_input[1], num_draws_input[2]], TRAINING_INPUT_TYPE)
        else:
            # NOTE: If doing permuations... full hand matrix will not match!
            if include_full_hand:
                cards_input = np.array([card_to_matrix(cards_array[0], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[1], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[2], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[3], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[4], pad_to_fit=pad_to_fit), full_hand_to_matrix], TRAINING_INPUT_TYPE)
            else:
                cards_input = np.array([card_to_matrix(cards_array[0], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[1], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[2], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[3], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[4], pad_to_fit=pad_to_fit)], TRAINING_INPUT_TYPE)

        # If 'context' is required... output 17 bits of zeros.
        if include_hand_context:
            empty_bits_array = np.zeros((CONTEXT_LENGTH, HAND_TO_MATRIX_PAD_SIZE, HAND_TO_MATRIX_PAD_SIZE), dtype=TRAINING_INPUT_TYPE)
            cards_input_with_context = np.concatenate((cards_input, empty_bits_array), axis = 0)
            cards_inputs_all.append(cards_input_with_context)
        else:
            cards_inputs_all.append(cards_input)

    return cards_inputs_all

# Special case, to output first one!
def cards_input_from_string(hand_string, pad_to_fit = PAD_INPUT, include_num_draws=False, num_draws=None, include_full_hand = False, include_hand_context = False):
    return cards_inputs_from_string(hand_string, pad_to_fit, max_inputs=1,
                                    include_num_draws = include_num_draws, num_draws=num_draws,
                                    include_full_hand = include_full_hand, include_hand_context = include_hand_context)[0]

# For encoding number of draws left (0-3), encode 3 "cards", of all 0's, or all 1's
# 3 draws: [1], [1], [1]
# 2 draws: [0], [1], [1]
# 1 draws: [0], [0], [1]
# 0 draws: [0], [0], [0]
# NOTE: This encodes the number back to front. Keep this for backward compatibility with models. 
def num_draws_input_from_string(num_draws_string, pad_to_fit = PAD_INPUT, num_draws_all_zero = NUM_DRAWS_ALL_ZERO):
    num_draws = int(num_draws_string)
    card_1 = card_to_matrix_fill(0, pad_to_fit = pad_to_fit)
    card_2 = card_to_matrix_fill(0, pad_to_fit = pad_to_fit)
    card_3 = card_to_matrix_fill(0, pad_to_fit = pad_to_fit)
    
    # We may want ot zero out this input...
    # Why? To initialize with same shape, but less information.
    if not num_draws_all_zero:
        if num_draws >= 3:
            card_3 = card_to_matrix_fill(1, pad_to_fit = pad_to_fit)
        if num_draws >= 2:
            card_2 = card_to_matrix_fill(1, pad_to_fit = pad_to_fit)
        if num_draws >= 1:
            card_1 = card_to_matrix_fill(1, pad_to_fit = pad_to_fit)

    return [card_3, card_2, card_1]

# More generalized form of the above integer -> bits encoding. But reverse order for better logic. Encode 1/5 as 10000
# TODO: Add Front/Back flag to flip.
def integer_to_card_array(number, max_integer, pad_to_fit = PAD_INPUT):
    output_array = [card_to_matrix_fill(0, pad_to_fit = pad_to_fit) for i in range(max_integer)]
    for index in range(len(output_array)):
        if int(number) > index: # (max_integer - index): Encode 1/5 as 00001
            output_array[index] = card_to_matrix_fill(1, pad_to_fit = pad_to_fit)
    return output_array


# Encode a string like '011' to 5-length array of "cards"
def bets_string_to_array(bets_string, pad_to_fit = PAD_INPUT):
    output_array = [card_to_matrix_fill(0, pad_to_fit = pad_to_fit) for i in range(5)]
    index = 0
    for c in bets_string:
        if index >= len(output_array):
            break
        else:
            if c == '0':
                continue
            elif c == '1':
                output_array[index] = card_to_matrix_fill(1, pad_to_fit = pad_to_fit)
            else:
                assert (c == '1' or c == '0'), 'unknown char |%s| in bets string!' % c
    return output_array

# [xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept]
def hand_input_from_context(position=0, pot_size=0, bets_string='', cards_kept=0, opponent_cards_kept=0, pad_to_fit = PAD_INPUT):

    position_input = card_to_matrix_fill(position, pad_to_fit = pad_to_fit)
    pot_size_input = pot_to_array(pot_size, pad_to_fit = pad_to_fit) # saved to single "card"
    bets_string_input = bets_string_to_array(bets_string, pad_to_fit = pad_to_fit) # length 5
    cards_kept_input = integer_to_card_array(cards_kept, max_integer = 5, pad_to_fit = pad_to_fit)
    opponent_cards_kept_input = integer_to_card_array(opponent_cards_kept, max_integer = 5, pad_to_fit = pad_to_fit)

    # Put it all together...
    # [xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept]
    hand_context_input = np.array([position_input, pot_size_input, 
                                   bets_string_input[0], bets_string_input[1], bets_string_input[2], bets_string_input[3], bets_string_input[4],
                                   cards_kept_input[0], cards_kept_input[1], cards_kept_input[2], cards_kept_input[3], cards_kept_input[4],
                                   opponent_cards_kept_input[0], opponent_cards_kept_input[1], opponent_cards_kept_input[2], opponent_cards_kept_input[3], opponent_cards_kept_input[4]], TRAINING_INPUT_TYPE)
    return hand_context_input

# Encode pot (0 to 3000 or so) into array... by faking a hand.
# Every $50 of pot is another card... so 50 -> [2c], 200 -> [2c, 2d, 2h, 2s]
def pot_to_array(pot_size, pad_to_fit = PAD_INPUT):
    pot_to_cards = []
    for rank in ranksArray:
        for suit in suitsArray:
            card = Card(suit=suit, value=rank)
            if pot_size >= 50:
                pot_to_cards.append(card)
                pot_size -= 50
            else:
                break
        if pot_size < 50:
            break
    pot_size_card = hand_to_matrix(pot_to_cards, pad_to_fit=pad_to_fit)
    return pot_size_card


# a hack, since we aren't able to implement liner loss in Theano...
# To remove, or reduce problems with really big values... map values above 2.0 points to sqrt(remainder)
# 25.00 to 6.80
# 50.00 to 8.93
def adjust_float_value(hand_val, mode = 'video'):
    if mode and mode == 'video':
        # Use small value if mean square error. Big value if linear error...
        base = 4.0 # 50.0 # 2.0
        if hand_val <= base:
            return hand_val
        else:
            remainder = hand_val - base
            remainder_reduce = math.sqrt(remainder)
            total = base + remainder_reduce
            #print('regressing high hand value %.2f to %.2f' % (hand_val, total))
            return total
    elif mode and mode == 'deuce':
        # For Deuce... we get values on 0-1000 scale. Adjust these to [0.0, 1.0] scale. Easier to train.
        assert hand_val >= 0 and hand_val <= 1000, '\'deuce\' value out of range %s' % hand_val
        return hand_val * 0.001
    elif mode and mode == 'deuce_events':
        # For events, we are looking at marginal value of events, in 100-200 limit game.
        # Thus, inputs range from +3000 (win big pot) to -2000 (lose a big one, and pay many future bets)
        # map these values to +3.0 and -2.0

        # Add a (significant) positive baseline to values. Why? A. stands out from noise data B. model not really built to predict negatives.
        return hand_val * 0.001 + EVENTS_VALUE_BASELINE # hack, to test if can learn negative values?
    else:
        # Unknown mode. 
        print('Warning! Unknown mode %s for value %s' % (mode, hand_val))
        return hand_val


# return value [0.0, 1.0] odds to keep a hand, given "hold value"
# Example: Keep all pairs, all straights, etc. Down-sample normal hands (no pair, etc)
def sample_rate_for_hold_value(hold_value):
    # Keep 100% of all pairs, straights & flushes
    # NOTE: These are hands that should not be kept. Especially straights and flushes. Train on this!
    if hold_value <= 0.080:
        return 1.0

    # Keep 100% of all good pat lows (9-low, 8-low, 7-low)
    # NOTE: It's important that we know these values well. For betting evaluation, especially.
    if hold_value >= 0.500:
        return 1.0

    # For the rest of the hands... interpolate a value, keeping best hands with highest likihood.
    # Why? Well.. it's important not to have step-functions. Except, with discrete things like a pair.
    xp = [0.1, 0.50]
    fp = [0.3, 1.0] # Keep as few a 1/3 of most common examples (bad pat hand, no pair). And almost 100% of good pat hands.
    prob_keep = np.interp(hold_value, xp, fp)
    return prob_keep

# Turn each hand into an input (cards array) + output (32-line value)
# if output_best_class==TRUE, instead outputs index 0-32 of the best value (for softmax category output)
# Why? A. Easier to train B. Can re-use MNIST setup.
def read_poker_line(data_array, csv_key_map, adjust_floats='video', include_num_draws = False, include_full_hand = False, include_hand_context = False):
    # array of equivalent inputs (if we choose to create more data by permuting suits)
    # NOTE: Usually... we don't do that.
    # NOTE: We can also, optionally, expand input to include other data, like number of draws made.
    # It might be more proper to append this... but logically, easier to place in the original np.array
    cards_inputs = cards_inputs_from_string(data_array[csv_key_map['hand']], max_inputs=1, 
                                            include_num_draws=include_num_draws, num_draws=data_array[csv_key_map['draws_left']], 
                                            include_full_hand = include_full_hand, include_hand_context = include_hand_context)

    #print(cards_inputs[0])
    #print((cards_inputs[0]).shape)
    #sys.exit(-5)

    # Ok now translate the 32-low output row.
    if csv_key_map.has_key('[]_value'):
        # full 32-output vector.
        # NOTE: We should *not* adjust floats... except when massively skewed data. For example... 800-9-6 video poker payout.
        # NOTE: Different games... will have different adjustment models.
        if adjust_floats:
            output_values = [adjust_float_value(float(data_array[csv_key_map[draw_value_key]]), mode=adjust_floats) for draw_value_key in DRAW_VALUE_KEYS] 
        else:
            output_values = [float(data_array[csv_key_map[draw_value_key]]) for draw_value_key in DRAW_VALUE_KEYS] 
        best_output = max(output_values)
        output_category = output_values.index(best_output)
    else:
        # just "best hand," in which case we need to look it up...
        assert(csv_key_map.has_key('best_draw'))
        
        best_draw_string = data_array[csv_key_map['best_draw']]
        #print('best draw for hands is %s. Need to compute the array index...' % best_draw_string)
        hand_string = data_array[csv_key_map['hand']]
        output_category = get_draw_category_index(hand_string_to_array(hand_string), best_draw_string)
        #print('found it at draw index %d!!' % output_category)

        assert(output_category >= 0)

        output_values = [0]*32 # Hack to just return empty array.
               
    # Output all three things. Cards input, best category (0-32) and 32-row vector, of the weights
    return (cards_inputs, output_category, output_values) 

# Same output format as poker line... but encodes a poker event (bet/check/fold)
# returns (hand_input, output_class, output_array):
# hand_input --> X by 19 x 19 padded cards and bits
# output_class --> index of event actually taken
# output_array --> 32-length output... but only the 'output_class' index matters. The rest are zero
# Thus, we can train on the same model as 3-round draw decisions... resulting good initialization.
def read_poker_event_line(data_array, csv_key_map, adjust_floats = 'deuce_event', pad_to_fit = PAD_INPUT, include_hand_context = False): 
    #print(data_array)

    # If we are told which player agent made this move, skip moves from players except those whom we trust.
    if csv_key_map.has_key('bet_model') and PLAYERS_INCLUDE_DEUCE_EVENTS:
        bet_model = data_array[csv_key_map['bet_model']]
        if not(bet_model in PLAYERS_INCLUDE_DEUCE_EVENTS):
            #if bet_model:
            #    print(data_array)
            #    print('skipping action from missing or undesireable model. |%s|' % bet_model)
            return

    # X x 19 x 19 input, including our hand, & num draws. 
    if not CARDS_INPUT_ALL_ZERO:
        cards_input = cards_input_from_string(data_array[csv_key_map['hand']], 
                                              include_num_draws=True, num_draws=data_array[csv_key_map['draws_left']], 
                                              include_full_hand = True)
    else:
        # If we want to focus on game structure and ignore cards, push cards all 0's... [still keep 3 bits for number of rounds]
        # TODO: "num_draws_input" should return np.array to avoid confusion about order, etc
        empty_bits_cards = np.zeros((5+1, HAND_TO_MATRIX_PAD_SIZE, HAND_TO_MATRIX_PAD_SIZE), dtype=TRAINING_INPUT_TYPE)
        num_draws_input = num_draws_input_from_string(num_draws_string=data_array[csv_key_map['draws_left']], pad_to_fit=pad_to_fit)
        num_draws_array = np.array([num_draws_input[0], num_draws_input[1], num_draws_input[2]], TRAINING_INPUT_TYPE)
        cards_input = np.concatenate((empty_bits_cards, num_draws_array), axis = 0)
    
    # Include this context output... or all xO, depending on what's asked
    if not CONTEXT_ALL_ZERO:
        # Along with xCards and xNumDraws, also encode...
        # xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept
        position = int(data_array[csv_key_map['position']]) # binary
        pot_size = float(data_array[csv_key_map['pot_size']]) # 150 - 3000 or so
        bets_string = data_array[csv_key_map['actions_this_round']] # items like '0111' (5 max)
        cards_kept = int(data_array[csv_key_map['num_cards_kept']]) # 0-5
        opponent_cards_kept = int(data_array[csv_key_map['num_opponent_kept']]) # 0-5

        hand_context_input = hand_input_from_context(position=position, pot_size=pot_size, bets_string=bets_string,
                                                     cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept)

        full_input = np.concatenate((cards_input, hand_context_input), axis = 0)
    else:
        empty_bits_array = np.zeros((CONTEXT_LENGTH, HAND_TO_MATRIX_PAD_SIZE, HAND_TO_MATRIX_PAD_SIZE), dtype=TRAINING_INPUT_TYPE)
        full_input = np.concatenate((cards_input, empty_bits_array), axis = 0)

    #print(full_input)
    #print('fully concatenated input %s, shape %s' % (type(full_input), full_input.shape))

    # Look up the action taken with this event. If it's unknown or not appropriate, skip.
    action_name = data_array[csv_key_map['action']]
    #print('action taken = |%s|' % action_name)
    action_taken = actionNameToAction[action_name]
    #print(action_taken)

    # We are only interesting in bets, raises, calls, checks and folds. All other actions can be ignored...
    if not ((action_taken in ALL_BETS_SET) or (action_taken in ALL_CALLS_SET) or (action_taken == CHECK_HAND) or (action_taken == FOLD_HAND)):
        #print('action not useful for poker event training %s' % action_taken)
        raise AssertionError()
    #else
        #print('acceptable & useful action %s' % action_taken)
        

    # Which place in the 32-vector array does this action map to?
    output_category = category_from_event_action(action_taken)

    # Our results... is an empty 32-vector, with only action's value set
    action_marginal_value = adjust_float_value(float(data_array[csv_key_map['margin_result']]), mode=adjust_floats)
    output_array = np.zeros(32)
    output_array[output_category] = action_marginal_value

    # Also, we need to code FOLD --> 0.0. Why? This helps with calibration. 
    fold_marginal_value = adjust_float_value(0.0, mode=adjust_floats)
    output_array[FOLD_CATEGORY] = fold_marginal_value

    # Do we return just the hand, or also 17 "bits" of context?
    if include_hand_context:
        return (full_input, output_category, output_array)
    else:
        return (cards_input, output_category, output_array)
    
# Read CSV lines, create giant numpy arrays of the input & output values.
def _load_poker_csv(filename=DATA_FILENAME, max_input=MAX_INPUT_SIZE, output_best_class=True, keep_all_data=False, format='video', include_num_draws = False, include_full_hand = False, sample_by_hold_value = SAMPLE_BY_HOLD_VALUE, include_hand_context = False):
    csv_reader = csv.reader(open(filename, 'rb')) # 'rU')) "line contains NULL byte"

    csv_key = None
    csv_key_map = None

    # Can't grow numpy arrays. So instead, be lazy and grow array to be turned into numpy arrays.
    X_train_not_np = [] # all input data (multi-dimension nparrays)
    y_train_not_np = [] # "best category" for each item
    z_train_not_np = [] # 32-length vectors for all weights
    m_train_not_np = [] # 32-length "mask" for which weights matter. 1 = yes 0 = N/A or ??
    hands = 0
    last_hands_print = -1
    lines = 0

    # Sample down even harder, if outputting equivalent hands by permuted suit (fewer examples for flushes)
    # NOTE: This samples by "best class" [32]
    # TODO: Alternatively, sample by the *value* of class[31] (keep all)
    if keep_all_data:
        sampling_policy = DATA_SAMPLING_KEEP_ALL
    else:
        sampling_policy = DATA_SAMPLING_REDUCE_KEEP_TWO 

    # compute histogram of how many hands, output "correct draw" to each of 32 choices
    y_count_by_bucket = [0 for i in range(32)] 
    for line in csv_reader:
        lines += 1
        if lines % 10000 == 0:
            print('Read %d lines' % lines)

        # Read the CSV key, so we look for columns in the data, not fixed positions.
        if not csv_key:
            print('CSV key' + str(line))
            csv_key = line
            csv_key_map = CreateMapFromCSVKey(csv_key)
        else:
            # Skip any mail-formed lines.
            try:
                # hand_inputs represents array of *equivalent* inputs (in case we want to permute inputs, to get more data)
                if format == 'video' or format == 'deuce':
                    hand_inputs_all, output_class, output_array = read_poker_line(line, csv_key_map, adjust_floats = format, include_num_draws=include_num_draws, include_full_hand = include_full_hand, include_hand_context = include_hand_context)
                elif format == 'deuce_events':
                    # For less confusion, if input data is "events" ie bets, checks, etc... 
                    # Data gets zipped into the same training format, trained with same shape, but just initialize it differently
                    # (re-use sub functions for encoding a hand, etc, whereever possible)
                    hand_input, output_class, output_array = read_poker_event_line(line, csv_key_map, adjust_floats = format, include_hand_context = include_hand_context)

                    # Get rid of input shuffling. At least for now
                    hand_inputs_all = [hand_input]
                else:
                    print('unknown input format: %s' % format)
                    sys.exit(-3)
            
            # except (IndexError): # Fewer errors, for debugging
            except (TypeError, IndexError, ValueError, KeyError, AssertionError): # Any reading error
                #print('\nskipping malformed input line:\n|%s|\n' % line)
                continue


            # Assumes that output_array is really just 0-31 for best class.
            # TODO: Just output array and class every time, choose which one to use!
            #output_class = output_array

            # Create an output, for each equivalent input...
            for hand_input in hand_inputs_all:
                # Weight of last item in output_array == value of keeping all 5 cards.
                hold_value = output_array[-1]

                # If requested, sample out some too common cases. 
                # A. Better balance
                # B. Faster training
                if format == 'video' and sampling_policy:
                    output_percent = sampling_policy[output_class]
                    if random.random() > output_percent:
                        continue
                elif format == 'deuce' and sample_by_hold_value:
                    # Alternatively, sample by the *value* of class[31] (keep all)
                    # Example: down-sample all, except dealt pairs, or dealt flushes, etc
                    sample_rate = sample_rate_for_hold_value(hold_value)
                    if random.random() > sample_rate:
                        #if (hands % 5000) == 0 and hands != last_hands_print:
                        #    print(line)
                        #    print('Skipping item with %s hold value!\n' % hold_value)
                        continue
                elif format == 'deuce_events':
                    # For 'deuce_events', randomly down-sample from full set of events. 
                    # NOTE: We do this, to cover more hands, and to not over-train on specific hand situation.
                    # TODO: Control this by flag, or pre-compute data before choosing sample policy
                    # TODO: Sample differently by "sim", "man", "cnn" actors...
                    sample_rate = SAMPLE_RATE_DEUCE_EVENTS
                    if random.random() > sample_rate:
                        continue

                # We can also add output "mask" for which of the "output_array" values matter.
                # As default, for testing, etc, try applying mask for "output_class" == 1
                output_mask = np.zeros((len(output_array)))
                output_mask[output_class] = 1.0 
                # For events, we also encode that fold == 0 value. Let's hope this helps sparse training.
                if format == 'deuce_events':
                    output_mask[FOLD_CATEGORY] = 1.0
                
                if (hands % 5000) == 1 and hands != last_hands_print:
                    print('\nLoaded %d hands...\n' % hands)
                    print(line)
                    #print(hand_input)
                    print(hand_input.shape)

                    # Attempt to show debug of the input... without the padding...
                    if HAND_TO_MATRIX_PAD_SIZE == 17:
                        opt = np.get_printoptions()
                        np.set_printoptions(threshold='nan')

                        # Also print the whole thing...
                        #print(hand_input)
                        #print('-------')

                        # Get all bits for input... excluding padding bits that go to 17x17
                        debug_input = hand_input[:,6:10,2:15]
                        print(debug_input)
                        print(debug_input.shape)

                        # Return options to previous settings...
                        np.set_printoptions(**opt)

                    print(output_class)
                    print(output_mask)
                    if format != 'deuce_events':
                        print('hold value %s' % hold_value)
                    print(output_array)
                    print('------------')
                    last_hands_print = hands

                # count class, if item chosen
                y_count_by_bucket[output_class] += 1

                X_train_not_np.append(hand_input)
                y_train_not_np.append(output_class)
                z_train_not_np.append(output_array)
                m_train_not_np.append(output_mask)

                hands += 1

            if hands >= max_input:
                break

            #sys.exit(-3)

    # Show histogram... of counts by 32 categories.
    print('count ground truth for 32 categories:\n%s\n' % ('\n'.join([str([DRAW_VALUE_KEYS[i],y_count_by_bucket[i],'%.1f%%' % (y_count_by_bucket[i]*100.0/hands)]) for i in range(32)])))

    X_train = np.array(X_train_not_np)
    y_train = np.array(y_train_not_np)
    z_train = np.array(z_train_not_np) # 32-length vectors
    m_train = np.array(m_train_not_np) # 32-length masks

    print('Read %d data points. Shape below:' % len(X_train_not_np))
    #print(X_train)
    #print(y_train)
    print('num_examples (train) %s' % X_train.shape[0])
    print('input_dimensions %s' % X_train.shape[1])
    print('X_train object is type %s of shape %s' % (type(X_train), X_train.shape))
    print('y_train object is type %s of shape %s' % (type(y_train), y_train.shape))
    print('z_train object is type %s of shape %s' % (type(z_train), z_train.shape))
    print('m_train object is type %s of shape %s' % (type(m_train), m_train.shape))

    return (X_train, y_train, z_train, m_train)

def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_poker_csv()

    # We ignore z (all values array)
    X_all, y_all, z_all = data

    # Split into train (remainder), valid (1000), test (1000)
    X_split = np.split(X_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    X_valid = X_split[0]
    X_test = X_split[1]
    X_train = X_split[2]

    print('X_valid %s %s' % (type(X_valid), X_valid.shape))
    print('X_test %s %s' % (type(X_test), X_test.shape))
    print('X_train %s %s' % (type(X_train), X_train.shape))

    # And same for Y
    y_split = np.split(y_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    y_valid = y_split[0]
    y_test = y_split[1]
    y_train = y_split[2]

    print('y_valid %s %s' % (type(y_valid), y_valid.shape))
    print('y_test %s %s' % (type(y_test), y_test.shape))
    print('y_train %s %s' % (type(y_train), y_train.shape))

    #sys.exit(0)

    # We ignore validation & test for now.
    #X_valid, y_valid = data[1]
    #X_test, y_test = data[2]

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1] * X_train.shape[2] * X_train.shape[3], # How much size per input?? 5x4x13 data (cards, suits, ranks)
        output_dim=32, # output cases
    )

def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
    """Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.

    The training function of this model must have a mini-batch size of
    `batch_size`.

    A theano expression which represents such a network is returned.
    """
    print('building model for input_dim: %s, output_dim: %s, batch_size: %d, num_hidden_units: %d' % (input_dim, output_dim,
                                                                                                      batch_size, num_hidden_units))

    print('creating l_in layer')
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    print('creating l_hidden1 layer')
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    print('creating l_hidden1_dropout layer')
    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
    )
    print('creating l_hidden2 layer')
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    print('creating l_hidden2_dropout layer')
    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
    )
    print('creating dense layer')
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out

# X_tensor_type=T.matrix --> for matrix (pixel) input
# need to change it... T.tensor4 [4-d matrix]
# http://deeplearning.net/software/theano/library/tensor/basic.html
def create_iter_functions(dataset, output_layer,
                          X_tensor_type=T.tensor4, # T.matrix,
                          batch_size=BATCH_SIZE,
                          learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    print('creating iter funtions')

    print('input dataset %s' % dataset)

    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x')

    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)

    # Default: Nesterov momentum. Try something else?
    # updates = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

    # "AdaDelta" by Matt Zeiler -- no learning rate or momentum...
    updates = lasagne.updates.adadelta(loss_train, all_params)

    # Add a function.... to evaluate the result of an input!
    #evaluate = theano.function()

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )

# Pass epoch_switch_adapt variable, to switch for adaptive training...
def train(iter_funcs, dataset, batch_size=BATCH_SIZE, epoch_switch_adapt=10000):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []

        # Hack: after X number of runs... switch to adaptive training!
        if epoch <= epoch_switch_adapt:
            print('default train for epoch %d' % epoch)
            sys.stdout.flush()
            for b in range(num_batches_train):
                batch_train_loss = iter_funcs['train'](b)
                batch_train_losses.append(batch_train_loss)
        else:
            print('adaptive training for epock %d' % epoch)
            sys.stdout.flush()
            for b in range(num_batches_train):
                batch_train_loss = iter_funcs['train_ada_delta'](b)
                batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


"""
# Borrowed from DeepPink chess.
# Apply function to (single) input
def get_model_from_pickle(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)
    
    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict
    """

#def get_predict_function(output_layer, 



# Now how do I return theano function to predict, from my given thing? Should be simple.
def predict_model(output_layer, test_batch):
    print('Computing predictions on test_batch: %s %s' % (type(test_batch), test_batch.shape))
    #pred = T.argmax(output_layer.get_output(test_batch, deterministic=True), axis=1)
    pred = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True)
    print('Prediciton: %s' % pred)
    print(tp.pprint(pred))
    softmax_values = pred.eval()
    print(softmax_values)

    #f = theano.function([output_layer.get_output], test_batch)

    #print(f)

    #print('Predition eval: %' % pred.eval

    """
    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)
    """

    pred_max = T.argmax(pred, axis=1)

    print('Maximums %s' % pred_max)
    print(tp.pprint(pred_max))

    softmax_choices = pred_max.eval()
    print(softmax_choices)

    # now debug the softmax choices...
    softmax_debug = [DRAW_VALUE_KEYS[i] for i in softmax_choices]
    print(softmax_debug)
    


def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print(dataset)

    #sys.exit(-3)

    print("Building model and compiling functions...")
    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
    )
    iter_funcs = create_iter_functions(dataset, output_layer)

    print("\n\nStarting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass


    # Can we do something with final output model? Like show a couple of moves...
    #test_batch = dataset['X_test']

    # Test cases
    test_cases = ['As,Ad,4d,3s,2c', 'As,Ks,Qs,Js,Ts', '3h,4s,3d,5c,6d', '2h,3s,4d,6c,5s',
                  '8s,Ad,Kd,8c,Jd', '8s,Ad,2d,7c,Jd', '[8s,9c,8c,Kd,7h]', '[Qh,3h,6c,5s,4s]', '[Jh,Td,9s,Ks,5s]',
                  '[6c,4d,Ts,Jc,6s]', '[4h,8h,2c,7d,3h]', '[2c,Ac,Tc,6d,3d]'] 

    print('looking at some test cases: %s' % test_cases)

    # Fill in test cases to get to batch size?
    #for i in range(BATCH_SIZE - len(test_cases)):
    #    test_cases.append(test_cases[1])
    test_batch = np.array([cards_input_from_string(case) for case in test_cases], TRAINING_INPUT_TYPE)
    predict_model(output_layer=output_layer, test_batch=test_batch)

    print('again, the test cases: \n%s' % test_cases)

    #print(predict(output_layer[x=test_batch]))

    return output_layer


if __name__ == '__main__':
    main()
