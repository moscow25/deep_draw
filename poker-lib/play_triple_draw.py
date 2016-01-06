import sys
import gc
import csv
import logging
import math
import time
import re
import random
import os.path # checking file existence, etc
import argparse # command line arguements parsing
import numpy as np
import scipy.stats
import lasagne
import theano
import theano.tensor as T

from poker_lib import *
from holdem_lib import * # if we want to support Holdem hands and game sim
from poker_util import *
from draw_poker_lib import * 

from draw_poker import cards_input_from_string
from draw_poker import hand_input_from_context
from draw_poker import holdem_cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import build_nopool_model
from triple_draw_poker_full_output import build_fully_connected_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector
from triple_draw_poker_full_output import evaluate_single_event # just give it the 26x17x17 bits... and get a vector back
from triple_draw_poker_full_output import evaluate_single_holdem_hand # for a holdem hand, returns 0-1.0 value vs random, and some odds
from triple_draw_poker_full_output import expand_parameters_input_to_match # expand older models, to work on larger input (just zero-fill layer)
# from triple_draw_poker_full_output import evaluate_batch_hands # much faster to evaluate a batch of hands

print('parsing command line args %s' % sys.argv)
parser = argparse.ArgumentParser(description='Play heads-up triple draw against a convolutional network. Or see two networks battle it out.')
# ---- Added for benefit of ACPC server ------
# To connect with the server!
parser.add_argument('-address', '--address', default='localhost', help='ACPC server/dealer address')
parser.add_argument('-port', '--port', default='48777', help='ACPC server/dealer PORT')
# ------------------------------------------------------------------------------------
parser.add_argument('-draw_model', '--draw_model', default=None, help='neural net model for draws, or simulate betting if no bet model') # draws, from 32-length array
parser.add_argument('-holdem_model', '--holdem_model', default=None, help='neural net model for Holdem hands, with first value 0-1.0 value vs random hand, other values the odds of making specific hand types. Baseline value for any valid hand and flop, turn or river')
parser.add_argument('-output', '--output', help='output CSV') # CSV output file, in append mode.

parser.add_argument('-CNN_model', '--CNN_model', default=None, help='neural net model for betting') # Optional CNN model. If not supplied, uses draw model to "sim" decent play
parser.add_argument('-CNN_model_tag', '--CNN_model_tag', default=None, help='name for CNN_model (compare_models format)') 
parser.add_argument('--human_player', action='store_true', help='pass for p2 = human player') # Declare if we want a human competitor? (as player_2)
parser.add_argument('-CNN_old_model', '--CNN_old_model', default=None, help='pass for p2 = old model (or second model)') # useful, if we want to test two CNN models against each other.
parser.add_argument('-CNN_old_model_tag', '--CNN_old_model_tag', default=None, help='name for CNN_old_model (compare_models format)') 
parser.add_argument('-CNN_other_old_model', '--CNN_other_old_model', default=None, help='pass for p2 = other old model (or 3rd model)') # and a third model, 
parser.add_argument('-compare_models', '--compare_models', action='store_true', help="pass for model A vs model B. Needs to input exactly two models") # Useful for A/B testing. Should auto-detect when a model is DNN or CNN. Leave model_2 empty for comp with heuristic. Crashes if 3 models given.
parser.add_argument('-hand_history', '--hand_history', default=None, help='shortcut to generate CSV from ACPC file (line per hand). NLH only') # Instead of fresh hands, give hand history, and generate CSV
args = parser.parse_args()

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing heads-up triple-draw poker, with both players simulated by AI.

As such, the game is run by a dealer, who controls actions, a deck, which can be drawn from,
player hands, and player agents... which make decisions, when propted, ultimately by querying an AI system.

The game is hard-coded to implement triple draw. But should be possible, to switch rules,
models and final hand evaluations, to accomodate any other draw game.
"""

# Lame, but effective way of splitting games.
# TODO: Do this from command line... but then need to ensure that all done correctly.
# Better yet, outside global constants class (modified from command line, etc)
# [default format is 'deuce']
CARDS_CANONICAL_FORM = True # if available. NOTE: Older models my work better w/o canonical form, new models require it
FORMAT = 'nlh' # 'deuce' # 'holdem' # 'deuce'
# TODO: allow 'holdem' (limit) vs 'nlh' from command line
if args.holdem_model and FORMAT != 'nlh':
    FORMAT = 'holdem'

USE_ACTION_PERCENTAGE = True # For CNN7+, use action percentage directly from the model? Otherwise, take action with highest value (some noise added)
USE_ACTION_PERCENTAGE_BOTH_PLAYERS = True # Try both players action percentage... can be useful for Hold'em. Good for exploratory moves
# Disable action% for holdem... until it's ready.
#if FORMAT == 'holdem':
#    USE_ACTION_PERCENTAGE = False 
if FORMAT == 'deuce':
    ACTION_PERCENTAGE_CHOICE_RATE = 0.5 # 0.7 # How often do we use the choice %?? (value with noise the rest of the time)
elif FORMAT == 'holdem':
    ACTION_PERCENTAGE_CHOICE_RATE = 0.3 # 0.5 # 0.7 # Reduce for holdem, as values get better. Otherwise, very primitive bets model... with not enough explore
elif FORMAT == 'nlh':
    ACTION_PERCENTAGE_CHOICE_RATE = 0.0 # Make the highest value play for NLH. And/or separate take on optimal bet size. 
else:
    ACTION_PERCENTAGE_CHOICE_RATE = 0.0

# Build up a CSV, of all information we might want for CNN training
# TODO: Replace with more logical header for 'Holdem', and other games...
TRIPLE_DRAW_EVENT_HEADER = ['hand', 'draws_left', 'best_draw', 'hand_after',
                            'bet_model', 'value_heuristic', 'position',  'num_cards_kept', 'num_opponent_kept',
                            'action', 'pot_size', 'bet_size', 'pot_odds', 
                            'bet_faced', 'stack_size', 'bet_this_street', # Added for NLH
                            'bet_this_hand', # total for all rounds
                            'actions_this_round', 'actions_full_hand', 
                            'total_bet', 'result', 'margin_bet', 'margin_result',
                            'current_margin_result', 'future_margin_result',
                            'oppn_hand', 'current_hand_win', # what oppn has, and are we winning?
                            'hand_num', 'running_average', 'bet_val_vector', 'act_val_vector', 'num_draw_vector', # model's internal predictions
                            'allin_vs_oppn', 'stdev_vs_oppn', 'allin_vs_random', 'stdev_vs_random', 'allin_categories_vector' # odds from Monte Carlo simulation
                            ] 

BATCH_SIZE = 100 # Across all cases

RE_CHOOSE_FOLD_DELTA = 0.50 # If "random action" chooses a FOLD... re-consider %% of the time.
# Tweak values by up to X (100*chips). Seems like a lot... but not really. 
# NOTE: Does *not* apply to folds. Don't tweak thos.
# NOTE: Lets us break out of a rut of similar actions, etc.
PREDICTION_VALUE_NOISE_HIGH = 0.06
PREDICTION_VALUE_NOISE_LOW = -0.04 # Do decrease it sometimes... so that we don't massively inflate value of actions
PREDICTION_VALUE_NOISE_AVERAGE = (PREDICTION_VALUE_NOISE_HIGH + PREDICTION_VALUE_NOISE_LOW)/2.0 

# Don't add any prediction noise for NLH [until we understand sizing better]
if FORMAT == 'nlh':
    PREDICTION_VALUE_NOISE_HIGH = 0.0

# Alternatively, use a more sophisticated "tail distribution" from Gumbel
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gumbel.html
# mean = mu + 0.58821 * beta (centered around mu). So match above
PREDICTION_VALUE_NOISE_BETA = 0.02 # 0.03 # 0.04 # 0.06 # vast majority of change within +- 0.05 value, but can stray quite a bit further. Helps make random-ish moves
PREDICTION_VALUE_NOISE_MU = PREDICTION_VALUE_NOISE_AVERAGE - 0.58821 * PREDICTION_VALUE_NOISE_BETA

# Don't boost aggressive actions so much... we want to see more calls, check, especially checks, attempted in spots that might be close.
AGGRESSIVE_ACTION_NOISE_FACTOR = 1.0 # 1.0 # 0.5
BOOST_AGGRESSIVE_ACTION_NOISE = True # training only(?), increase value of aggressive actions (don't let them go negative)
MULTIPLE_MODELS_NOISE_FACTOR = 1.0 # 0.5 # Reduce noise... if using multiple models already (noise that way)
BOOST_PAT_BET_ACTION_NOISE = True # Do we explicitly support betting after standing pat? Yes. Even if hand is weak (snowed, etc).

# Enable, to use 0-5 num_draw model. Recommends when to snow, and when to break, depending on context.
USE_NUM_DRAW_MODEL = True
# Use a num_draw model... and tend to do so more later in the hand. For example, 30% first draw, 60% 2nd draw, 90% 3rd draw.
NUM_DRAW_MODEL_RATE = 1.0 # 0.7 # how often do we use num_draw model? Just use context-free 0-32 output much/most of the time...
NUM_DRAW_MODEL_RATE_REDUCE_BY_DRAW = 0.4 # 0.3 # Use model on the final draw, but perhaps less on previous draws...
NUM_DRAW_MODEL_NOISE_FACTOR = 0.2 # Add noise to predictions... but just a little. 
FAVOR_DEFAULT_NUM_DRAW_MODEL = True # Enable, to boost # of draw cards preferred by 0-32 model. Else, too noisy... but strong preference for other # of cards still matters.

INCLUDE_HAND_CONTEXT = True # False 17 or so extra "bits" of context. Could be set, could be zero'ed out.

SHOW_HUMAN_DEBUG = True # Show debug, based on human player...
SHOW_MACHINE_DEBUG_AGAINST_HUMAN = False # True, to see machine logic when playing (will reveal hand)
USE_MIXED_MODEL_WHEN_AVAILABLE = True # When possible, including against human, use 2 or 3 models, and choose randomly which one decides actions.
RETRY_FOLD_ACTION = True # If we get a model that says fold preflop... try again. But just once. We should avoid raise/fold pre
ADJUST_VALUES_TO_FIX_IMPOSSIBILITY = True # Do we fix impossile values? Like check < 0.0, or calling on river > pot + bet
# TODO: Cap adjustment from RAW_FOLD_VALUE? Or apply it with a discount...
ADJUST_TO_RAW_FOLD_VALUE = True # Subtract out "fold value." Can get rather positive, on a good board, and rather negative preflop, and other times.

# Turn this on... only if we are not vulnerable to super-aggro patting machine. In the end... should be off. Gotta pay off sometimes.
USE_NEGATIVE_RIVER_CALL_VALUE = False #  True # Prevent action% model, when river negative value to call? (still allowed to tweak values and call)
NEGATIVE_RIVER_CALL_CUTOFF = -0.05 # try to reduce *really bad* calls, like straights, big pairs, other hopeless hands.  
#if FORMAT == 'holdem':
#    NEGATIVE_RIVER_CALL_CUTOFF = -0.01 # Holdem is tighter. Don't auto-bet with bad hands...

# Block anything noise-wise, that may not work well for NLH.
if FORMAT == 'nlh':
    RETRY_FOLD_ACTION = False # Don't re-consider preflop folds in NLH
    ADJUST_VALUES_TO_FIX_IMPOSSIBILITY = True # Fixed/added NLH impossibility logic (FOLD == 0.0, CHECK >= 0.0, etc)
    USE_NEGATIVE_RIVER_CALL_VALUE = True
    # Hands with good allin value should not be folded. Especially preflop
    # NOTE: This compares to model's internal value.
    # NOTE: Can be super-conservative and set this to 0.80, etc. But need to give a value
    # TODO: Different value for preflop? Or for small pots in general??
    NEVERFOLD_ALLIN_VALUE_ESTIMATE = 0.505 # 0.485 # 0.510 # Use larger value as we move into production. Good to keep 48-49% for self-play!

# For NLH model trained on good data, consider bet/nonbet choice, based on "aggressive%" learned from data (similar context)
USE_AGGRO_CUTOFFS = True # Do we prevent NL model from betting, even if high +EV, but learned "aggressive%" basically zero
MINIMUM_AGGRO_RATE_TO_BET = 0.03 # With leaky ReLU, no-bet cases will usually be negative. To be safe, use a very small (-) number
MAXIMUM_AGGRO_RATE_TO_NOT_BET = 1.0 - MINIMUM_AGGRO_RATE_TO_BET # If model says 100% betting here, reduce value of check/call, as long as bet value is positive (don't fold by accident)
# As sanity check, don't discount call values over X, not matter what (even allin pot, negative FOLD value)
MAXIMUM_MIN_CALL_CHIP_VALUE = 300.0

# Do we ask the player to make highest-value play, or to follow CFR as close as possible?
IMITATE_CFR_AGGRO_STRATEGY = True 
IMITATE_CFR_FOLD_STRATEGY = True # Parallel adjustment, for fold vs call [if EV close, use the percentage]
IMITATE_CFR_BETTING_PERCENTAGE = 0.95 # 0.8 # 0.7 # How often to imitate CFR, and how often to choose best-value action? (per-hand basis)
CLOSE_BET_RATIO_CUTOFF_IMITATE_CFR = 0.3 # pretty liberal ratio, in terms of pot size, to choose bet based on CFR aggro%

# If we are given FOLD%... use it to
# A. Stop aggressive pruning on FOLD, when EV close to flat (especially preflop)
# B. If close, and IMITATE_CFR_AGGRO_STRATEGY, use it to actually determine with which rate to fold.
# NOTE: It's all about close calls! Don't use CFR to over-rule a strong call, or a strong fold.
# It's about being unpredictable when it's close, and about reducing model sensitively.
# TODO: A true Bayesian approach would be excellent. 
USE_FOLD_PERCENTAGE = True
MINIMUM_FOLD_RATE_TO_FOLD = 0.03
MAXIMUM_FOLD_RATE_TO_CALL = 1.0 - MINIMUM_FOLD_RATE_TO_FOLD
# The *best* hand we'd ever fold, based on CFR strategy?
# TODO: Some sort of percentage cutoff?? What if 2k value, but facing 10k bet? 
MAXIMUM_CALL_TO_FOLD_WITH_CFR = 2 * 300.0 
MINIMUM_CALL_VALUE_TO_CALL_WITH_CFR = -1.0 * MAXIMUM_CALL_TO_FOLD_WITH_CFR # The *worst* had we'll ever call, based on CFR strategy?

# Do we choose bet sizing from smoothed bet-values?
# NOTE: Gaussian smoothing, in both x (bet size) and y (value).
# For now, just take best value. Could (easily) make it a little stochastic.
# NOTE: Order of inputs matters (for smoothing), even if multiple x == same. Why? min-bet has meaning. Also, easier.
BET_SIZE_FROM_SMOOTHED = True

# Given a check/bet number in range [not 0.0 or 1.0], do we need to add in a bias?
# Why? Experimentally, the system imitates CFR, but plays too passively. 
# Read 160000 lines
# ['bet', 11567, '15.6%']
# ['raise', 9700, '13.1%']
# ['check', 30047, '40.6%']
# ['call', 13712, '18.5%']
# ['fold', 8974, '12.1%']
# 
# compare to CFR:
# Read 1700000 lines
# ['bet', 125962, '17.0%']
# ['raise', 123090, '16.6%']
# ['check', 243369, '32.9%']
# ['call', 136800, '18.5%']
# ['fold', 110779, '15.0%']
# 
# The biggest difference is [40.6% check, 15.6% bet] vs [32.9% check, 17.0% bet]
# The CFR is also significantly more likely to raise, when bet into.
#
# Solution: twist aggro% from model, by X%. Now obviously it's not (aggro + X) --> aggro.
# So largest difference is applied at the 50/50 mark. Take that +20%: 50% -> 60% aggressive.
AGGRESSION_PERCENTAGE_TWIST_FACTOR = 0.30

# From experiments & guesses... what contitutes an 'average hand' (for our opponen), at this point?
# TODO: Consider action so far (# of bets made this round)
# TODO: Consider # of cards drawn by opponent
# TODO: Consider other action so far...
def baseline_heuristic_value(round, bets_this_round = 0):
    baseline = RANDOM_HAND_HEURISTIC_BASELINE
    if round == PRE_DRAW_BET_ROUND:
        baseline = RANDOM_HAND_HEURISTIC_BASELINE
    elif round == DRAW_1_BET_ROUND:
        baseline = RANDOM_HAND_HEURISTIC_BASELINE + 0.10
    elif round == DRAW_2_BET_ROUND:
        baseline = RANDOM_HAND_HEURISTIC_BASELINE + 0.150
    elif round == DRAW_3_BET_ROUND:
        baseline = RANDOM_HAND_HEURISTIC_BASELINE + 0.200
        
    # Increase the baseline... especially as we get into 3-bet and 4-bet territory.
    # NOTE: For NLH, "bets this round" is in bet size, not number of bets.
    if bets_this_round >= 1:
        baseline += 0.05 * (bets_this_round) 
        # baseline += 0.05 * (bets_this_round - 1)

    return min(baseline, 0.90)

# Return indices for [0, 5] cards kept (best x-card draw)
def best_five_draws(hand_draws_vector):
    best_draws = []
    offset = 0
    # 0-card draw
    best_draws.append(0)
    offset += 1
    # 1-card draw
    sub_array = hand_draws_vector[offset:offset+5]
    best_draw = np.argmax(sub_array)
    best_draws.append(best_draw + offset)
    offset += 5
    # 2-card draw
    sub_array = hand_draws_vector[offset:offset+10]
    best_draw = np.argmax(sub_array)
    best_draws.append(best_draw + offset)
    offset += 10
    # 3-card draw
    sub_array = hand_draws_vector[offset:offset+10]
    best_draw = np.argmax(sub_array)
    best_draws.append(best_draw + offset)
    offset += 10
    # 4-card draw
    sub_array = hand_draws_vector[offset:offset+5]
    best_draw = np.argmax(sub_array)
    best_draws.append(best_draw + offset)
    # 5-card draw
    best_draws.append(31)
    return best_draws

# Stochastic, but always positive boost, for "best_draw" from 0-32 model, in num_draw model.
def best_draw_value_boost():
    # 0.5 * (5 x max(0, noise))
    noise = 0.5 * (max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) )
    # 2 x noise_average
    noise += PREDICTION_VALUE_NOISE_AVERAGE * 2.0
    return noise

# Actually a demotion. Suppress 'pat' draw, as model learns to suggest it way too often.
# NOTE: Especially useful to let other alternatives to 'best_draw' thrive.
def pat_draw_value_boost():
    return -0.5 * best_draw_value_boost()

# Similarly, demote 3 & 4 card draws, late in the hand. Reason is the same.
# These trigger when normal draw is bad. But still... taking 3 or 4 cards late isn't the answer. Consider alternatives.
def draw_many_value_boost():
    return -1.0 * best_draw_value_boost()
def draw_very_many_value_boost():
    return -2.0 * best_draw_value_boost()

# Should inherit from more general player... when we need one. (For example, manual player who chooses his own moves and own draws)
class TripleDrawAIPlayer():
    # TODO: Initialize model to use, etc.
    def __init__(self):
        self.draw_hand = None
        self.name = '' # backward-compatible, not really used any more...
        self.tag = '' # Useful for outside tagging (re-create ACPC logs, etc)

        # TODO: Name, and track, multiple models. 
        # This is the draw model. Also, outputs the heuristic (value) of a hand, given # of draws left.
        self.output_layer = None # for draws
        self.input_layer = None
        self.holdem_output_layer = None # simimilary for holdem (if applies)
        self.holdem_input_layer = None
        self.bets_output_layer = None 
        self.bets_input_layer = None
        self.use_learning_action_model = False # should we use the trained model, to make betting decisions?
        self.old_bets_output_model = False # "old" model used for NIPS... should be set from the outside
        self.other_old_bets_output_model = False # the "other" old bets model [CNN3 vs CNN5, etc]
        self.bets_output_array = [] # if we use *multiple* models, and choose randomly between them
        self.use_action_percent_model = False # Make moves from CNN action-values (with noise added), or from action percentages from CNN?
        self.is_dense_model = False # neural network is DNN?
        self.is_human = False

        # Special cases for NLH
        self.imitate_CFR_betting = False # Try to check/bet at ration learned from CFR training??

        # Current 0-1000 value, based on cards held, and approximation of value from draw model.
        # For example, if no more draws... heuristic is actual hand.
        self.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 
        self.num_cards_kept = 0 # how many cards did we keep... with out last draw?
        self.cards = [] # Display purposes only... easy way to see current hand as last evaluated

        # For further debug... latest debug vectors for various thing values we get back from AI model
        self.bet_val_vector = []
        self.act_val_vector = []
        self.num_draw_vector = []

        # TODO: Use this to track number of cards discarded, etc. Obviously, don't look at opponent's cards.
        self.opponent = None

    def player_tag(self):
        if self.tag:
            return self.tag # set from outside, as for ACPC re-create, etc
        elif self.is_human:
            return 'man'
        elif self.use_learning_action_model and self.bets_output_layer:
            # Backward-compatibility hack, to allow support for "old" model used for NIPS paper (CNN)
            # NOTE: Will be deprecated...
            if self.is_dense_model:
                name = 'DNN_2'
            elif self.bets_output_array and len(self.bets_output_array) > 0:
                name = 'CNN_87' # we sample from multiple models!
            elif self.other_old_bets_output_model:
                name = 'CNN_6'
            elif self.old_bets_output_model:
                name = 'CNN_1' # Hack here if we have AI1 vs AI2
                # Another hack, to turn off bet-action for CNN_1 [unless we want it on to explore]
                self.use_action_percent_model = False;
            else:
                name = 'CNN_8'

            # To help remember
            if self.use_action_percent_model:
                name = name + '_per'
            return name
        else:
            if FORMAT == 'holdem':
                return 'holdem_sim'
            elif FORMAT == 'nlh':
                return 'nlh_sim'
            else:
                return 'sim'

    # Takes action on the hand. But first... get Theano output...
    def draw_move(self, deck, num_draws = 1, debug = True, draw_recommendations = None):
        # Reduce debug, if opponent is human, and could see.
        if self.opponent and self.opponent.is_human and (not SHOW_MACHINE_DEBUG_AGAINST_HUMAN):
            debug = False

        hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
        if debug:
            print('dealt %s for draw %s' % (hand_string_dealt, num_draws))

        # Get 32-length vector for each possible draw, from the model.
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = num_draws, input_layer=self.input_layer) #, test_batch=self.test_batch)
        if debug:
            print('All 32 values: %s' % str(hand_draws_vector))
        # For further debug, and to use outside suggestion of #draw... select best draw for each 0-5 cards kept
        best_draws_by_num_kept = best_five_draws(hand_draws_vector)
        if debug:
            print('best draws: %s' % best_draws_by_num_kept)
            for i in xrange(len(best_draws_by_num_kept)):
                best_draw = best_draws_by_num_kept[i]
                print('\tBest draw %d: %d [value %.2f] (%s)' % (5-i, best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        # Just choose best draw, ignoring recommendations of # cards (break, snow, etc)
        best_draw = np.argmax(hand_draws_vector)
        best_draw_no_model = best_draw # for debub & comparison
        best_draw_num_kept = len(all_draw_patterns[best_draw])
        if debug:
            print('Best draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        
        # Alternatively, use the recommendation from num_draw model, if available
        # [do this X% of the time]
        draw_model_rate = NUM_DRAW_MODEL_RATE
        if NUM_DRAW_MODEL_RATE_REDUCE_BY_DRAW:
            # reduce frequency of num_draws model use... as we go earlier in the hand.
            draw_model_rate -= (num_draws - 1) * NUM_DRAW_MODEL_RATE_REDUCE_BY_DRAW

        # Another hack. Don't use draw model with CNN_1 (remove if we want to build from it)
        if draw_recommendations and USE_NUM_DRAW_MODEL and self.player_tag() != 'CNN_1' and self.player_tag() != 'CNN_1_per' and random.random() <= draw_model_rate:
            # The point isn't to over-rule 0-32 model in close cases. The point is to look for *clear advantages* to 
            # snowing a hand, or breaking a hand. Therefore, add a bonus to the move already preferred by 0-32 model.
            if FAVOR_DEFAULT_NUM_DRAW_MODEL:
                default_num_kept = best_draw_num_kept # 0-5
                for prediction in draw_recommendations:
                    action = prediction[1]
                    # Boost the value by fixed positive but noisy amount (noise only on the upside)
                    # This will reduce randomly choosing an inferior draw. But not every time. And *much better* draw action wins easily.
                    # NOTE: Boost the pat draw... but less... if model recommends breaking.
                    if drawCategoryNumCardsKept[action] == default_num_kept:
                        if action == KEEP_5_CARDS:
                            noise = abs(pat_draw_value_boost())
                        else:
                            noise = best_draw_value_boost()
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted %d-card draw by %.3f' % (5-default_num_kept, noise))
                    # Demote pat draw somewhat. This is especially useful if we need alternatives. As pat represents itself well.
                    # NOTE: DO *not* demote pat... if it's the recommended move
                    # TODO: Also do not demote pat... if it's close in value to the recommended move
                    if action == KEEP_5_CARDS and drawCategoryNumCardsKept[action] != default_num_kept:
                        noise = pat_draw_value_boost()
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted pat draw by %.3f' % (noise))
                    # Also demote 3,4,5 card draws (and 2-card draw on final round). Look for other alteratives, even in a tough spot. Especially late in the hand.
                    if (num_draws < 3) and (action == KEEP_2_CARDS) and drawCategoryNumCardsKept[action] != default_num_kept:
                        noise = draw_many_value_boost()
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted %d-draw by %.3f' % (5-drawCategoryNumCardsKept[action], noise))
                    elif (num_draws < 3) and (action == KEEP_0_CARDS or action == KEEP_1_CARDS) and drawCategoryNumCardsKept[action] != default_num_kept:
                        # Almost no reason at all to draw 4+ cards. Strongly consider any other option
                        noise = draw_very_many_value_boost()
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted %d-draw by %.3f' % (5-drawCategoryNumCardsKept[action], noise))
                    elif (num_draws == 1) and (action == KEEP_3_CARDS) and drawCategoryNumCardsKept[action] != default_num_kept:
                        # Demote 2-card draw on the final round. If close... better to stand pat or take 1 card. 
                        noise = pat_draw_value_boost()
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted %d-draw by %.3f' % (5-drawCategoryNumCardsKept[action], noise))
                # Re-sort results, of course.
                draw_recommendations.sort(reverse=True)

            action = draw_recommendations[0][1]
            cards_kept = drawCategoryNumCardsKept[action]
            best_draw = best_draws_by_num_kept[cards_kept]
            if debug:
                print('\tchosen to use num_draw model with %.1f%%' % (draw_model_rate * 100.0))
                print(draw_recommendations)
                print('\trecommend keeping %d cards' % cards_kept)
                print('\tBest draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))

                if best_draw != best_draw_no_model:
                    print('\t->model changed the draw!')
                else:
                    print('\t->model recommends same draw!')

        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        draw_string = ''
        for i in range(0,5):
            if not (i in all_draw_patterns[best_draw]):
                draw_string += '%d' % i

        if debug:
            print('Draw string from AI! |%s|' % draw_string)
            print('\nDraw %d cards.\n' % len(draw_string))
        else:
            print('\nDraw %d cards.\n' % len(draw_string))

        discards = self.draw_hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        self.draw_hand.deal(new_cards, final_hand=True)

        # Record current setting of these values...
        # NOTE: heuristic value... is before we got our cards.
        self.num_cards_kept = len(all_draw_patterns[best_draw])
        self.heuristic_value = expected_payout
        
        return expected_payout

    # Baseline the hand value. CNN lookup, but with differences depending on the game.
    # TODO: Change logic to where we pass game type, etc. For now, easier to use a global.
    def update_hand_value(self, num_draws=0, debug = True):
        # print('calling update_hand_value for format %s' % FORMAT)
        if FORMAT == 'holdem' or FORMAT == 'nlh':
            self.update_holdem_hand_value(debug=debug)
        else:
            self.update_draw_hand_value(num_draws=num_draws, debug=debug)

    # Get CNN output for a hand, flop, turn, river, or any legal subset. Will return 0.0-1.0 value for odds vs random hand.
    # Also other odds that we can use or ignore.
    # NOTE: It would be easy to just give it cards here... but even easier for the AI to track its own hand.
    def update_holdem_hand_value(self, debug = True):
        if not self.holdem_output_layer:
            #assert self.holdem_output_layer, 'Need holdem_output_layer CNN model, if we want to value holdem hands!'
            print('Need holdem_output_layer CNN model, if we want to value holdem hands!')
            return

        # Hack: Save a lot of time by not outputting these.
        # TODO: We could fill in the data from bets model, since it also outputs these... if we like.
        return 

        # Reduce debug, if opponent is human (don't give away our hand)
        if self.opponent and self.opponent.is_human:
            debug = False

        # The inputs we need to CNN is the cards, flop, turn, river [also save for debug and training cases]
        self.cards = self.holdem_hand.dealt_cards
        self.flop = self.holdem_hand.community.flop
        self.turn = self.holdem_hand.community.turn
        self.river = self.holdem_hand.community.river

        if debug:
            print('dealt %s' % ([hand_string(self.cards), hand_string(self.flop), hand_string(self.turn), hand_string(self.river)]))
            
        # Quickly exit, and save a lookup, for human player!
        if self.is_human:
            return

        # Lookup in the Holdem value model...
        hand_draws_vector = evaluate_single_holdem_hand(output_layer=self.holdem_output_layer, input_layer=self.holdem_input_layer,
                                                        cards = self.cards, flop = self.flop, turn = self.turn, river = self.river)
        # Always just first value. The rest predict odds of a flush, etc.
        best_draw = 0
        if debug:
            # TODO: would be better to print: "flush = 15%", etc (for all non-zero values!)
            print('All Holdem values: %s' % str(np.around(hand_draws_vector[0:len(HOLDEM_VALUE_KEYS)], decimals=6)))
            print('Holdem value: %.3f\n' % (hand_draws_vector[best_draw]))
        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        self.heuristic_value = expected_payout

        return expected_payout

    # Apply the CNN... to get "value" of the current hand. best draw for hands with draws left; current hand for no more draws.
    # NOTE: Similar to draw_move() but we don't make any actual draws with the hand.
    def update_draw_hand_value(self, num_draws=0, debug = True):
        # Reduce debug, if opponent is human (don't give away our hand)
        if self.opponent and self.opponent.is_human:
            debug = False

        # For no more draws... use "final hand." Otherwise we run into issues with showdown, etc
        if (num_draws >= 1):
            hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
            self.cards = self.draw_hand.dealt_cards
        else:
            hand_string_dealt = hand_string(self.draw_hand.final_hand)
            self.cards = self.draw_hand.final_hand

        if debug:
            print('dealt %s for draw %s' % (hand_string_dealt, num_draws))
            
        # Quickly exit, and save a lookup, for human player!
        if self.is_human:
            return

        # Get 32-length vector for each possible draw, from the model.
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = max(num_draws, 1), input_layer=self.input_layer) #, test_batch=self.test_batch)

        # Except for num_draws == 0, value is value of the best draw...
        if num_draws >= 1:
            if debug:
                print('All 32 values: %s' % str(hand_draws_vector))
            best_draw = np.argmax(hand_draws_vector)
        else:
            print('With no draws left, heurstic value is for pat hand.')
            best_draw = 31
        
        if debug:
            print('Best draw: %d [value %.2f] (%s)\n' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        self.heuristic_value = expected_payout

        return expected_payout

    # Apply current model, based on known information, to draw 0-5 cards from the deck.
    def draw(self, deck, num_draws = 1, bets_this_round = 0, 
             has_button = True, pot_size=0, actions_this_round=[], actions_whole_hand=[],
             cards_kept=0, opponent_cards_kept=0, 
             debug = True, retry = False):
        # Reduce debug, if opponent is human, and could see.
        if (self.opponent and self.opponent.is_human and (not SHOW_MACHINE_DEBUG_AGAINST_HUMAN)) or self.is_human:
            debug = False
        
        """
        # If we have context and bets model that also outputs draws... at least give it a look.
        bets_layer = self.bets_output_layer # use latest "bets" layer, even if multiple available.
        bets_input_layer = self.bets_input_layer
        """
        # Either use bets model... or if given several... choose one at random to apply
        # NOTE: We do not want to average the *values* in models, but the actual choices. Thus we'll be stochastic & unpredicatable.
        # NOTE: We want to use all possible draws models... since otherwise not enough exploration if single model over or under-does something.
        bets_layer = None
        bets_input_layer = None
        if self.bets_output_array:
            # TODO: Also track input layer, for faster evaluation...
            index = random.randrange(0, len(self.bets_output_array))
            bets_layer, bets_input_layer = self.bets_output_array[index]
            if debug:
                print('chose draws model %d in %d-length models index...' % (index, len(self.bets_output_array)))
        else:
            bets_layer = self.bets_output_layer
            bets_input_layer = self.bets_input_layer
        value_predictions = None
        if bets_layer and self.use_learning_action_model and self.player_tag() != 'CNN_1' and self.player_tag() != 'CNN_1_per':
            if debug:
                print('trying draw model in debug mode...')
            num_draws_left = num_draws
            if (num_draws_left >= 1):
                hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
            else:
                hand_string_dealt = hand_string(self.draw_hand.final_hand)

            # Input related to the hand
            cards_input = cards_input_from_string(hand_string_dealt, include_num_draws=True, 
                                                  num_draws=num_draws_left, include_full_hand = True, 
                                                  include_hand_context = False)

            # TODO: This should be a util function.
            bets_string = ''
            for action in actions_this_round:
                if action.type in ALL_BETS_SET:
                    bets_string += '1'
                elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                    bets_string += '0'
                else:
                    # Don't encode non-bets
                    continue

            # Repeat the same for 'whole game' bets string
            # TODO: Use a util function!
            all_rounds_bets_string = ''
            for action in actions_whole_hand:
                if action.type in ALL_BETS_SET:
                    all_rounds_bets_string += '1'
                elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                    all_rounds_bets_string += '0'
                else:
                    # Don't encode non-bets
                    continue
            
            # Now hand context
            if debug:
                if self.is_dense_model:
                    print('~ DNN model ~')
                print('context %s' % ([hand_string_dealt, num_draws_left, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept, all_rounds_bets_string]))
            hand_context_input = hand_input_from_context(position=has_button, pot_size=pot_size, bets_string=bets_string,
                                                         cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept,
                                                         all_rounds_bets_string=all_rounds_bets_string)
            full_input = np.concatenate((cards_input, hand_context_input), axis = 0)
            
            """
            # What do input bits look like?
            if debug:
                opt = np.get_printoptions()
                np.set_printoptions(threshold='nan')

                # Get all bits for input... excluding padding bits that go to 17x17
                debug_input = full_input[:,6:10,2:15]
                print(debug_input)
                print(debug_input.shape)

                # Return options to previous settings...
                np.set_printoptions(**opt)
                """

            # TODO: Rewrite with input and output layer... so that we can avoid putting all this data into Theano.shared()
            bets_vector = evaluate_single_event(bets_layer, full_input, input_layer = bets_input_layer)

            # Show the values for all draws [0, 5] cards kept.
            # For action %... normalize the vector to show action %%.
            if debug:
                print('vals\t%s' % ([(val - 2.0) for val in bets_vector[:5]]))
                print('acts\t%s' % ([val for val in bets_vector[5:10]]))
                print('drws\t%s' % ([(val - 2.0) for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]))
            #value_predictions = [[(bets_vector[category_from_event_action(action)] - 2.0), action, '%s: %.3f' % (actionName[action], bets_vector[category_from_event_action(action)] - 2.0)] for action in actions]
            value_predictions = [[(bets_vector[action] - 2.0), action, '%s: %.3f' % (drawCategoryName[action], bets_vector[action] - 2.0)] for action in DRAW_CATEGORY_SET]
            value_predictions.sort(reverse=True)
            
            # Save for debug
            self.bet_val_vector = [int(x * 100000) / 100000.0 for x in [(val - 2.0) for val in bets_vector[:5]]]
            self.act_val_vector = [int(x * 100000) / 100000.0 for x in [val/max(np.sum([val for val in bets_vector[5:10]]), 0.01) for val in bets_vector[5:10]]]
            self.num_draw_vector = [int(x * 100000) / 100000.0 for x in [(val - 2.0) for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]]

            if debug:
                print(value_predictions)

            # Here is where we add noise to predictions, if so inclined...
            # (add very little noise, as many values are correctly, close together)
            if NUM_DRAW_MODEL_NOISE_FACTOR:
                for prediction in value_predictions:
                    action = prediction[1]
                    noise = 0.0
                    noise = np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)
                    noise *= NUM_DRAW_MODEL_NOISE_FACTOR
                    # TODO: Do we want to exclude some draws from noise? Like 4, 5 card draws that we prefer to discourage...
                    prediction[0] += noise
                # re-sort the values, after noise added
                value_predictions.sort(reverse=True)

                if debug:
                    print(value_predictions)

        # Get and apply action... from 0-32 actions layer.
        print('Requesting draw_move() for format %s' % FORMAT)
        self.draw_move(deck, num_draws, draw_recommendations = value_predictions)

        # TODO: We should log this, and output information useful for tracking.
        #self.draw_random(deck)

    # Total placeholder. Draw random cards.
    def draw_random(self, deck=None):
        draw_string = ''
        for i in range(0,5):
            if random.random() > 0.50:
                draw_string += '%d' % i

        discards = self.draw_hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        self.draw_hand.deal(new_cards, final_hand=True)

    # This should choose an action policy... based on things we know, randomness, CNN output, RL, etc
    def choose_action(self, actions, round, bets_this_round = 0, bets_sequence = [], chip_stack = 0,
                      has_button = True, pot_size=0, actions_this_round=[], actions_whole_hand=[],
                      cards_kept=0, opponent_cards_kept=0, 
                      debug = True, retry = False):
        # Reduce debug, if opponent is human, and could see.
        if self.opponent and self.opponent.is_human and (not SHOW_MACHINE_DEBUG_AGAINST_HUMAN):
            debug = False

        # print('Choosing among actions %s for round %s' % (actions, round))
        # self.choose_random_action(actions, round)

        # Either use bets model... or if given several... choose one at random to apply
        # NOTE: We do not want to average the *values* in models, but the actual choices. Thus we'll be stochastic & unpredicatable.
        bets_layer = None
        bets_input_layer = None
        if self.bets_output_array:
            # TODO: Also track input layer, for faster evaluation...
            index = random.randrange(0, len(self.bets_output_array))
            bets_layer, bets_input_layer = self.bets_output_array[index]
            if debug:
                print('chose bets model %d in %d-length models index...' % (index, len(self.bets_output_array)))
        else:
            bets_layer = self.bets_output_layer
            bets_input_layer = self.bets_input_layer

        if bets_layer and self.use_learning_action_model:
            #print('We has a *bets* output model. Use it!')
            if FORMAT == 'holdem' or FORMAT == 'nlh':
                cards_string = hand_string(self.holdem_hand.dealt_cards)
                flop_string = hand_string(self.holdem_hand.community.flop)
                turn_string = hand_string(self.holdem_hand.community.turn)
                river_string = hand_string(self.holdem_hand.community.river)
                cards_input = holdem_cards_input_from_string(cards_string, flop_string, turn_string, river_string, include_hand_context = False, use_canonical_form = CARDS_CANONICAL_FORM)
            else:
                num_draws_left = 3
                if round == PRE_DRAW_BET_ROUND:
                    num_draws_left = 3
                elif round == DRAW_1_BET_ROUND:
                    num_draws_left = 2
                elif round == DRAW_2_BET_ROUND:
                    num_draws_left = 1
                elif round == DRAW_3_BET_ROUND:
                    num_draws_left = 0

                if (num_draws_left >= 1):
                    hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
                else:
                    hand_string_dealt = hand_string(self.draw_hand.final_hand)

                # Input related to the hand
                cards_input = cards_input_from_string(hand_string_dealt, include_num_draws=True, 
                                                      num_draws=num_draws_left, include_full_hand = True, 
                                                      include_hand_context = False)

            # TODO: This should be a util function.
            # NOTE: 'Actions' can be objects, with "type", or a string...
            bets_string = ''
            #print('actions_this_round: |%s|' % actions_this_round)
            if FORMAT == 'nlh':
                print('encoding NLH actions_this_round: |%s|' % actions_this_round)
                if actions_this_round and isinstance(actions_this_round, basestring):
                    bets_string = actions_this_round
                else:
                    bets_string = encode_big_bets_string(actions_this_round)
            elif actions_this_round and isinstance(actions_this_round, basestring):
                print('detected actions_this_round is string: %s' % actions_this_round)
                for action in actions_this_round:
                    if action == 'r':
                        bets_string += '1'
                    elif action == 'c':
                        bets_string += '0'
                    else:
                        # Don't encode non-bets
                        continue
                print(bets_string)
            else:
                bets_string = encode_limit_bets_string(actions_this_round)
            #print('bets_string |%s|' % bets_string)

            # Repeat the same for 'whole game' bets string
            # TODO: Use a util function!
            # NOTE: 'Actions' can be objects, with "type", or a string...
            all_rounds_bets_string = ''
            #print('actions_whole_hand %s' % actions_whole_hand)
            if FORMAT == 'nlh':
                print('encoding NLH actions_whole_hand: |%s|' % actions_whole_hand)
                if actions_whole_hand and isinstance(actions_whole_hand, basestring):
                    all_rounds_bets_string = actions_whole_hand
                else:
                    all_rounds_bets_string = encode_big_bets_string(actions_whole_hand)
            elif actions_whole_hand and isinstance(actions_whole_hand, basestring):
                print('detected actions_whole_hand is string: %s' % actions_whole_hand)
                for action in actions_whole_hand:
                    if action == 'r':
                        all_rounds_bets_string += '1'
                    elif action == 'c':
                        all_rounds_bets_string += '0'
                    else:
                        # Don't encode non-bets
                        continue
                print(all_rounds_bets_string)
            else:
                all_rounds_bets_string = encode_limit_bets_string(actions_whole_hand)
            #print('all_rounds_bets_string |%s|' % all_rounds_bets_string)
            
            # Now hand context
            if debug:
                if self.is_dense_model:
                    print('~ DNN model ~')
                if FORMAT == 'holdem' or FORMAT == 'nlh':
                    # print('bets this street: %s' % bets_this_street)
                    print('context %s' % ([cards_string, flop_string, turn_string, river_string, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept,  all_rounds_bets_string]))
                else:
                    print('context %s' % ([hand_string_dealt, num_draws_left, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept,  all_rounds_bets_string]))

            # TODO: Clean up this hack. Need "x_events" format for model loading.
            format = 'deuce_events'
            if FORMAT == 'holdem':
                format = 'holdem_events'
            elif FORMAT == 'nlh':
                format = 'nlh_events'
            hand_context_input = hand_input_from_context(position=has_button, pot_size=pot_size, bets_string=bets_string,
                                                         cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept,
                                                         all_rounds_bets_string=all_rounds_bets_string, format=format)
            full_input = np.concatenate((cards_input, hand_context_input), axis = 0)

            ###########################
            """
            #print(full_input)
            print('fully concatenated input %s, shape %s' % (type(full_input), full_input.shape))
            opt = np.get_printoptions()
            np.set_printoptions(threshold='nan')

            # Get all bits for input... excluding padding bits that go to 17x17
            if (format=='nlh_events' and DOUBLE_ROW_BET_MATRIX) or DOUBLE_ROW_HAND_MATRIX:
                debug_input = full_input[:,4:12,2:15]
            else:
                debug_input = full_input[:,6:10,2:15]
            print(debug_input)
            print(debug_input.shape)

            # Return options to previous settings...
            np.set_printoptions(**opt)
            """
            
            #############################

            # TODO: Rewrite with input and output layer... so that we can avoid putting all this data into Theano.shared()
            bets_vector = evaluate_single_event(bets_layer, full_input, input_layer = bets_input_layer)

            # Show all the raw returns from training. [0:5] -> values of actions [5:10] -> probabilities recommended
            # NOTE: Actions are good, and useful, and disconnected from values... but starting with CNN7 only!
            # For action %... normalize the vector to show action %%.
            if debug:
                if FORMAT == 'nlh':
                    print(('vals\t%s' % (['%.5f' % (val - 2.0) for val in bets_vector[:5]])).replace("'", ''))
                    print(('betsize\t%s' % (['%.2f' % ((val - 2.0) * 10000) for val in bets_vector[5:13]])).replace("'", '')) # If betting, value by size
                    print(('chips\t%s' % (['%.2f' % ((val) * 10000) for val in bets_vector[13:17]])).replace("'", '')) # chips already in the pot
                    print(('allins\t%s' % (['%.4f' % (val) for val in bets_vector[17:]])).replace("'", '')) # Odds and chances to make a hand
                else:
                    print('vals\t%s' % ([(val - 2.0) for val in bets_vector[:5]]))
                    print('acts\t%s' % ([val for val in bets_vector[5:10]]))
                    print('drws\t%s' % ([(val - 2.0) for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]))

            # Values of 5 basic actions.
            chip_bet_ratio = 1000.0
            if FORMAT == 'nlh':
                chip_bet_ratio = 10000.0
            value_predictions = [[(bets_vector[category_from_event_action(action)] - 2.0), action, '%s: %.3f' % (actionName[action], bets_vector[category_from_event_action(action)] - 2.0)] for action in actions]

            # Anything else worth saving?
            # Look at bet sizes. Easiest: recommended bet size.
            recommended_bet_size = bets_vector[BET_SIZE_MADE] * chip_bet_ratio

            # Now look at values for various bet sizes.
            # TODO: Split this into functions, whenever possible!
            # A. From pot size and bet pattern, compute min_bet, max_bet
            # TODO: Compute this much more accurately! (See heuristic value function)
            #bet_faced = bets_vector[BET_FACED] * chip_bet_ratio
            stack_size = chip_stack
            #min_bet = min(max(bet_faced, SMALL_BET_SIZE), stack_size)

            # If NLH, bets_sequence -> use this for bet size faced
            bet_faced = 0.0
            raise_amount = 0.0
            min_bet = 0.0
            max_bet = 0.0
            # Bet faced is the previous bet, if any. Minimum raise requires us to bet (bet_faced + minimum raise amount)
            if bets_sequence:
                bet_faced = max(bets_sequence[-1], 0.0)
                raise_amount = bet_faced
                if len(bets_sequence) >= 2:
                    raise_amount = max(bets_sequence[-1] - bets_sequence[-2], SMALL_BET_SIZE)
            min_bet = bet_faced + raise_amount
            min_bet = max(min_bet, SMALL_BET_SIZE)
            max_bet = chip_stack
            min_bet = min(min_bet, max_bet)

            # With min bets, where are we going wrong?
            print('min_bet: %d\tbet_faced: %d\traise_amount: %d\tmax_bet:%d\t(sequence %s)' % (min_bet, bet_faced, raise_amount, max_bet, bets_sequence))

            # Since model is trained on high-quality ACPC games, we also learn "aggro%."
            # How often does a good player bet or raise here?
            fold_rate = bets_vector[FOLD_PERCENT]
            aggro_rate = bets_vector[AGGRESSION_PERCENT]
            if debug:
                print('\tfrom CFR:\tfold: %.3f\taggro: %.3f' % (fold_rate, aggro_rate))
            
            # We can use this in two ways:
            # A. Remove the "bet" option completely, if aggro_rate is very low (near 0%). 
            # B. Create a player that follows the aggro% rate precisely. So if raising 30% of the time... that's what we do.
            # Both players have value.
            # [A] is needed, because until and unless we explore all states, value of bets in unknown cases is unreliable.
            # (and the model is heavily biased to toward betting, since good CFR players bet with an advantage)
            # [B] is useful, at the very least for training. Our best chance to create a CFR-like player.
            # (that said, we should also mix in [B] in production at some rate, to make model stochastic)
            # TODO: Best solution is probably a mix of [A] and [B]. Sometimes we exploit, given that state is understood,
            # Other cases we mimick the CFR ratio. 
            # TODO: Lastly, we could *also* mix in some rare bets outside of [A], using an epsilon, and non-trivial model advantage bet > call.

            # Pre-compute some values, for easy comparison.
            # Are we allowed to bet? Is the value of betting positive or negative (before adjustment)?
            all_bets_set = set([prediction[1] for prediction in value_predictions])
            raw_bet_value = 0.0
            raw_fold_value = 0.0 # look for model over-valuing fold (should also adjust call value, perhaps)
            raw_check_call_value = 0.0
            for prediction in value_predictions:
                action = prediction[1]
                value = prediction[0]
                if action in ALL_BETS_SET:
                    raw_bet_value = value
                elif action == FOLD_HAND:
                    raw_fold_value = value
                elif action == CHECK_HAND or action == CALL_NO_LIMIT:
                    raw_check_call_value = value

            # Since we do comparisons, (might be) better to subract the raw fold from CALL and RAISE.
            # NOTE: Especially with CFR model, we tend to see significant +EV for fold, in big-board cases
            # NOTE: While preflop, significantly -EV for fold. 
            # Obviously FOLD === 0.0, and we reset that. So need to adjust the others.
            # TODO: We could ajust the FOLD at a discount, or cap it, etc.
            if ADJUST_TO_RAW_FOLD_VALUE and raw_fold_value != 0.0:
                if debug and abs(raw_fold_value) >= SMALL_BET_SIZE / chip_bet_ratio:
                    print('--> non-trivial fold value %.5f, to be added to checks and calls' % raw_fold_value)
                for prediction in value_predictions:
                    action = prediction[1]
                    value = prediction[0]
                    if action in ALL_BETS_SET:
                        prediction[0] -= raw_fold_value
                        raw_bet_value = prediction[0]
                    elif action == CHECK_HAND or action == CALL_NO_LIMIT:
                        prediction[0] -= raw_fold_value
                        raw_check_call_value = prediction[0]

            # A. Remove the "bet" option completely, if aggro_rate is very low (near 0%). 
            # TODO: Add logic for epsilon...
            if USE_AGGRO_CUTOFFS:
                # Now iterate over actions, and make adjustments based on "aggro%"
                for prediction in value_predictions:
                    action = prediction[1]
                    value = prediction[0]
                    # TODO: Consider if bet/raise, but all bet-sizes are negative (or only allin is positive).
                    # In that case, consider not betting.
                    if action in ALL_BETS_SET and aggro_rate <= MINIMUM_AGGRO_RATE_TO_BET and value >= 0.0:
                        # If bet value > 0.0 but aggro_rate effectively zero, Fix bet/raise value at small negative value.
                        # TODO: Consider an epsilon. Especially with very high expected +EV value
                        prediction[0] = -1.0 * SMALL_BET_SIZE / chip_bet_ratio
                        if debug:
                            print('--> With aggro %.3f, supress the +EV bet action %.5f.' % (aggro_rate, value))
                    elif ((action in ALL_CALLS_SET) or (action == CHECK_HAND)) and (all_bets_set.intersection(ALL_BETS_SET)) and (raw_bet_value >= 1.0 * SMALL_BET_SIZE / chip_bet_ratio):
                        #if debug:
                        #    print('Considering check/call adjustment toward raise/bet. Since bet exists, and has +EV raw value: %.4f' % raw_bet_value)
                        if aggro_rate >= MAXIMUM_AGGRO_RATE_TO_NOT_BET and value > 0.0001:
                            # Adjusting check/call value to one min bet below raw bet value.
                            prediction[0] = max(0.0001, raw_bet_value - (1.0 * SMALL_BET_SIZE / chip_bet_ratio))
                            if debug:
                                print('--> Aggro percent very high and bet value +EV, so adjusting check/call value down %.4f -> %.4f' % (value, prediction[0]))
                            value = prediction[0] # in case we need to adjust again
                    
                    # Separate loop, to push down up the value of calling, if fold% very very low (and call-value is not terrible)
                    if (action == CALL_NO_LIMIT) and (raw_check_call_value >= MINIMUM_CALL_VALUE_TO_CALL_WITH_CFR / chip_bet_ratio):
                        #if debug:
                        #    print('Considering fold-hand adjustment toward a call. Since call exist, and has non-bad raw value: %.4f' % raw_check_call_value)
                        if fold_rate <= MINIMUM_FOLD_RATE_TO_FOLD:
                            # If fold rate is less than 2% and call is non-terrible, just lower the fold rate, below a call. 
                            prediction[0] = max(value + (0.1 * SMALL_BET_SIZE / chip_bet_ratio), 0.0 + (0.5 * SMALL_BET_SIZE / chip_bet_ratio))
                            if debug:
                                print('--> Fold percent very low and call value OK, so incrase CALL value up %.4f --> %.4f' % (value, prediction[0]))
                            value = prediction[0] # in case we need to adjust again



                    # This one is tricky. Look for cases that are 100% folds in the data. Rule somewhat nuanced, to idea is clear.
                    # Basically, if there is strong evidence that call_value close to 0%, hand is bad (in this context), and model always folds, then fold.
                    # Tricky, because a few things to look at:
                    # - call_value below $50
                    # - call_value below 10% of the pot
                    # - call_value below 10% of the pot (or below $50) if we subtract positive-fold from model
                    # [for large pots, no unusual for fold value to be > $0. This noise spreads to the call, also]
                    # NOTE: This only applies if aggro% == 0.0, since we are looking for auto-fold spots.
                    if action == CALL_NO_LIMIT and aggro_rate <= MINIMUM_AGGRO_RATE_TO_BET and value >= 0.0 and not (USE_FOLD_PERCENTAGE and fold_rate <= MAXIMUM_FOLD_RATE_TO_CALL) :
                        # What cutoff is appropriate for this hand?
                        minimum_bet_value = max(0.0, (0.5 * SMALL_BET_SIZE / chip_bet_ratio)) # 1/2 of bet, or $50
                        minimum_bet_value = max(minimum_bet_value, 0.05 * (pot_size) / chip_bet_ratio) # or 5% of the pot
                        # Add "raw_bet_value" if FOLD  > 0.0... up to 10% of the current pot (minus a min bet)
                        minimum_bet_value = min(minimum_bet_value, 0.10 * (pot_size - SMALL_BET_SIZE) / chip_bet_ratio)
                        # Avoid folding for very small bets, if still slightly +EV
                        minimum_bet_value = min(minimum_bet_value, 0.3 * bet_faced / chip_bet_ratio) 
                        # Final sanity check... don't block call values above a certain threshold.
                        minimum_bet_value = min(minimum_bet_value, MAXIMUM_MIN_CALL_CHIP_VALUE / chip_bet_ratio)
                        
                        if debug:
                            print('Looking at call_value [bet_faced %.2f], since aggro low (%.3f), call_value %.4f. Make sure not auto-fold spot. raw_fold %.4f' % (bet_faced, aggro_rate, value, raw_fold_value))
                            print('--> Minimum chips for this hand to call: %.2f' % (minimum_bet_value * chip_bet_ratio))
    
                        if value < minimum_bet_value:
                            print('--> Minimum chips to call %.2f > %.2f chips call. Turning call into fold...' % (minimum_bet_value * chip_bet_ratio, 
                                                                                                                   value * chip_bet_ratio))
                            prediction[0] = -1.0 * SMALL_BET_SIZE / chip_bet_ratio
                            value = prediction[0] # in case we need to adjust again
                                
                                
            # Now consider the case where we want to imitate CFR, rather than choose highest-value bet.
            # A. Still apply cutoffs above.
            # B. If aggro% between min and max, we are allowed to bet/raise, and bet/raise val "reasonably close" to "check/call" value
            # --> also make sure that it's not all negative values (for example, preflop bad hand)
            # C. Flip a weighted coin, and promote the lower value, just above the higher value
            #bet_aggression_strategy = IMITATE_CFR_AGGRO_STRATEGY # TODO: Assign this per-hand, or some % of the time.
            imitate_CFR_strategy = self.imitate_CFR_betting
            if imitate_CFR_strategy and IMITATE_CFR_AGGRO_STRATEGY:
                # Use a "twist" to bias the model in favor of more aggressive actions.
                if AGGRESSION_PERCENTAGE_TWIST_FACTOR and aggro_rate < 1.0 and aggro_rate > 0.0:
                    aggro_rate_twist = min(aggro_rate, abs(aggro_rate - 1.0)) * AGGRESSION_PERCENTAGE_TWIST_FACTOR
                    if debug:
                        print('\ttwisting aggro_rate:\t%.4f -> %.4f (%.3f factor)' % (aggro_rate, aggro_rate + aggro_rate_twist, AGGRESSION_PERCENTAGE_TWIST_FACTOR))
                    aggro_rate += aggro_rate_twist

                agressive_action_difference = abs(raw_bet_value - raw_check_call_value)
                if aggro_rate <= MAXIMUM_AGGRO_RATE_TO_NOT_BET and aggro_rate >= MINIMUM_AGGRO_RATE_TO_BET and (all_bets_set.intersection(ALL_BETS_SET)) and (raw_bet_value > 0.0 or raw_check_call_value > 0.0):
                    close_bet_ratio = (agressive_action_difference - ( 0.5 * SMALL_BET_SIZE) / chip_bet_ratio) / max((pot_size + 2.0 * SMALL_BET_SIZE) / chip_bet_ratio, raw_bet_value)
                    #print('\nConsidering imitating CFR for active/passive choice. B/R: %.5f C/K: %.5f. Aggro: %.3f\nClose bet ratio: %.3f\n' % (raw_bet_value, raw_check_call_value, aggro_rate, close_bet_ratio)) 
                    if close_bet_ratio > CLOSE_BET_RATIO_CUTOFF_IMITATE_CFR:
                        if debug:
                            print('Close_bet ratio %.3f too *wide*. Take highest value play' % close_bet_ratio)
                    else:
                        # NOTE: Dumb, but if paranoid, see the coin.
                        coin = np.random.random()
                        if debug:
                            print('Close_bet ratio %.3f close enough. Flip coin %.4f to aggro or not.' % (close_bet_ratio, coin))
                        if coin <= aggro_rate:
                            if debug:
                                print('Choose aggressive action! B/R: %.5f' % (raw_bet_value))
                                if raw_bet_value < raw_check_call_value:
                                    print('\t--> making a change to the best-move in favor of aggression')
                            for prediction in value_predictions:
                                action = prediction[1]
                                value = prediction[0]
                                if action in ALL_BETS_SET:
                                    prediction[0] = max(value, raw_check_call_value + (SMALL_BET_SIZE / chip_bet_ratio))
                                    prediction[0] = max(prediction[0], 1.5 * SMALL_BET_SIZE / chip_bet_ratio) # in case we boost the call higher...
                        else:
                            if debug:
                                print('Choose passive action! C/K: %.5f' % (raw_check_call_value))
                                if raw_bet_value > raw_check_call_value:
                                    print('\t--> making a change to the best-move in favor of passivity')
                            for prediction in value_predictions:
                                action = prediction[1]
                                value = prediction[0]
                                if action == CHECK_HAND or action == CALL_NO_LIMIT:
                                    prediction[0] = max(value, raw_bet_value + (SMALL_BET_SIZE) / chip_bet_ratio)
            
            # Now repeat the imitate_CFR... for call/FOLD
            if imitate_CFR_strategy and IMITATE_CFR_FOLD_STRATEGY:
                fold_action_difference = abs(raw_check_call_value) # Should be abstracted out for FOLD already
                # MINIMUM_FOLD_RATE_TO_FOLD = 0.03 -- MAXIMUM_FOLD_RATE_TO_CALL = 0.97
                if fold_rate <= MAXIMUM_FOLD_RATE_TO_CALL and fold_rate >= MINIMUM_FOLD_RATE_TO_FOLD and (CALL_NO_LIMIT in all_bets_set) and (raw_check_call_value >= MINIMUM_CALL_VALUE_TO_CALL_WITH_CFR / chip_bet_ratio):
                    close_bet_ratio = (fold_action_difference - ( 0.5 * SMALL_BET_SIZE) / chip_bet_ratio) / ((pot_size + 2.0 * SMALL_BET_SIZE) / chip_bet_ratio)
                    #print('\nConsidering imitating CFR for call/fold choice. Call: %.5f FOLD: %.5f. FOLD: %.3f\nClose bet ratio: %.3f\n' % (raw_check_call_value, raw_fold_value, fold_rate, close_bet_ratio)) 
                    if close_bet_ratio > CLOSE_BET_RATIO_CUTOFF_IMITATE_CFR:
                        if debug:
                            print('Close_bet ratio %.3f too *wide*. Take highest value play' % close_bet_ratio)
                    else:
                        # NOTE: Dumb, but if paranoid, see the coin.
                        coin = np.random.random()
                        if debug:
                            print('Close_bet ratio %.3f close enough. Flip coin %.4f to call or fold.' % (close_bet_ratio, coin))
                        if coin >= fold_rate:
                            if debug:
                                print('Choose passive action! C/K: %.5f' % (raw_check_call_value))
                                if raw_check_call_value < 0.0:
                                    print('\t--> making a change to the best-move in favor of calling')
                            for prediction in value_predictions:
                                action = prediction[1]
                                value = prediction[0]
                                if action == CALL_NO_LIMIT:
                                    prediction[0] = max(value, 0.0 + (SMALL_BET_SIZE) / chip_bet_ratio)
                        else:
                            if debug:
                                print('Choose FOLD action! C/K: %.5f' % (raw_check_call_value))
                                if raw_check_call_value > 0.0:
                                    print('\t--> making a change to the call-value in favor of folding')
                            for prediction in value_predictions:
                                action = prediction[1]
                                value = prediction[0]
                                if action == CALL_NO_LIMIT:
                                    # Don't over-rule call value that's actually pretty good.
                                    if value <= (MAXIMUM_CALL_TO_FOLD_WITH_CFR / chip_bet_ratio):
                                        prediction[0] = min(value, 0.0 - (1.0 * SMALL_BET_SIZE) / chip_bet_ratio)
                                    elif debug:
                                        print('Chosen to FOLD by CFR, but cant overrule call value %.4f > %.4f' % (value,  (MAXIMUM_CALL_TO_FOLD_WITH_CFR / chip_bet_ratio)))


            # B. All bet sizes, corresponding to bet sizes array
            bet_sizes_vector = np.clip([pot_size * percent for percent in NL_BET_BUCKET_SIZES], min_bet, stack_size)
            if debug:
                print('bet buckets (pot %.0f)\t%s' % (pot_size, bet_sizes_vector))
            # C. Smooth out -- especially at the edges (same value getting different recommendations)
            # D. Save best bet value
            bet_sizes_values = (bets_vector[5:13] - 2.0) * chip_bet_ratio
            if debug:
                print('bet values (for buckets)\t%s' %  bet_sizes_values)
            
            """
            # Attempt crude value-at-risk adjustment. [1.0 if negative value]
            value_at_risk_factor = [np.sqrt(pot_size + min_bet)/np.sqrt(value + pot_size) for value in bet_sizes_vector]
            value_at_risk_factor = [(factor if value >= 0 else 1.0) for (factor, value) in zip(value_at_risk_factor, bet_sizes_values)]
            #if debug:
            #    print('VARisk factors\t%s' % value_at_risk_factor)
            bet_sizes_values = np.multiply(bet_sizes_values, value_at_risk_factor)
            #if debug:
            #    print('bet values (for buckets)\t%s' %  bet_sizes_values)

            # E. Now... try to map gaussian mixture distribution from the data
            # F. Choose a bet size... from the Gaussian mixture.
            # G. [Option to adjust risk-adjusted model?] Ok to bet big... but bias against large bets.
            """
            max_value_index = np.argmax(bet_sizes_values)
            best_bet_size = bet_sizes_vector[max_value_index]
            #if debug:
            #    print('best bet: %s' % best_bet_size)
            recommended_bet_size = best_bet_size
            # --> bounds checking to make sure nothing crazy

            
            # Do we (instead) choose bet size from the smoothing of these values?
            # TODO: Skip this (slightly) expensive step, if we are not making a bet/raise.
            if BET_SIZE_FROM_SMOOTHED:
                # Internal model parameter. Estimate of own odds to win allin vs opponent, given the action.
                allin_vs_oppn = bets_vector[ALLIN_VS_OPPONENT]
                best_bet_size, value = best_bet_with_smoothing(bets = bet_sizes_vector, values = bet_sizes_values,
                                                               min_bet = min_bet, pot_size = pot_size, allin_win = allin_vs_oppn)
                if debug:
                    print('\tsmoothing recommends:\tbet: %d\tvalue: %.2f' % (best_bet_size, value))
                recommended_bet_size = np.clip(best_bet_size, min_bet, stack_size)


            # Use this space to fix bad logic in the values. Things that could not *possibly* be true.
            # A. Check has value < 0.0. 
            # B. Check has value > pot, and we are closing the action (checking behind)
            # C. Call has value > pot, and we are closing the action (calling final bet)
            adjusted_values_to_fix_impossibility = False
            if ADJUST_VALUES_TO_FIX_IMPOSSIBILITY:
                for prediction in value_predictions:
                    action = prediction[1]
                    value = prediction[0]
                    if action == FOLD_HAND:
                        # A1. Fold == 0.0 
                        if debug:
                            print('--> reset fold-hand value to 0.0. It can not be negative.')
                        prediction[0] = 0.0
                        adjusted_values_to_fix_impossibility = True
                    elif action == CHECK_HAND:
                        #print('examining check_hand action, value %.2f' % value)
                        #print('round = %s, has_button = %s' % (round, has_button))
                        # A. Check has value < 0.0. 
                        if value < 0.0:
                            if debug:
                                print('--> reset check-hand value to 0.0. It can not be negative.')
                            prediction[0] = 0.0
                            adjusted_values_to_fix_impossibility = True
                        # B. Check has value > pot, if we are closing the action.
                        if value > pot_size / chip_bet_ratio and (round == DRAW_3_BET_ROUND) and has_button:
                            if debug:
                                print('--> reset check-down value to pot size, if we closing the action.')
                            prediction[0] = pot_size / chip_bet_ratio
                            adjusted_values_to_fix_impossibility = True
                    elif action == CALL_NO_LIMIT:
                        # Internal model parameter. Estimate of own odds to win allin vs opponent, given the action.
                        allin_vs_oppn = bets_vector[ALLIN_VS_OPPONENT]
                        # C2. Call has negative value, but model says our hand value above good-hand threshold (allin vs oppn)
                        # NOTE: This value can be 48%, it can be 88%. At some point, we should not be folding.
                        if value <= 0.0 and allin_vs_oppn >= NEVERFOLD_ALLIN_VALUE_ESTIMATE:
                            if debug:
                                print('--> estimated allin_vs_oppn %.5f > %.5f, so adjusting call value above a fold.' % (allin_vs_oppn, NEVERFOLD_ALLIN_VALUE_ESTIMATE))
                            prediction[0] = allin_vs_oppn * SMALL_BET_SIZE / chip_bet_ratio
                            adjusted_values_to_fix_impossibility = True
                    elif action == CALL_BIG_STREET:
                        # C. Call has value > pot, and we are closing the action (calling final bet)
                        if value > (pot_size + BIG_BET_SIZE) / chip_bet_ratio * 0.95 and round == DRAW_3_BET_ROUND:
                            if debug:
                                print('--> reset call-down value to pot size plus bet, if we calling on the river')
                            prediction[0] = (pot_size + BIG_BET_SIZE) / chip_bet_ratio * 0.95
                            adjusted_values_to_fix_impossibility = True
                    elif (action == BET_BIG_STREET or action == RAISE_BIG_STREET):
                        # D. If bet/raise action is > 90% pot... use values, not action %. Until act% improves, just take greedy action.
                        if value > (pot_size + BIG_BET_SIZE) / chip_bet_ratio * 0.90 and round == DRAW_3_BET_ROUND:
                            if debug:
                                print('--> bet/raise value is > 90%% pot size (%.3f vs %.1f). So use greedy bet action' % (value, pot_size))
                            adjusted_values_to_fix_impossibility = True
            value_predictions.sort(reverse=True)

            # If we are on the river and calling has negative value... notice, so we don't use action% model (still ok to tweak & call)
            negative_river_call_value = False
            if USE_NEGATIVE_RIVER_CALL_VALUE and (round == DRAW_3_BET_ROUND):
                for prediction in value_predictions:
                    action = prediction[1]
                    value = prediction[0]
                    # NOTE: Use figure slightly less than 0.0 as cutoff... since it's really about clear lossses, not really marginal calls at 0.0
                    if action == CALL_BIG_STREET and value < NEGATIVE_RIVER_CALL_CUTOFF:
                        negative_river_call_value = True

            #if adjusted_values_to_fix_impossibility:
            #    sys.exit(-1)

            # Now apply noise to action values, if requested.
            # Why? If actions are close, don't be vulnerable to small differences that lock us into action.
            # NOTE: We do *not* want more folds, so only increase values of non-fold actions. Clear folds still fold.
            # Also, boost betting out if we're drawing ahead.
            best_action_no_noise = value_predictions[0][1]
            adjust_bet_better_draw = False
            if PREDICTION_VALUE_NOISE_HIGH:
                for prediction in value_predictions:
                    action = prediction[1]
                    noise = 0.0

                    # Don't boost FOLD at all.
                    if action == FOLD_HAND:
                        noise = 0.0
                    else:
                        # Naive approach: random value between x and y
                        # noise = np.random.uniform(PREDICTION_VALUE_NOISE_LOW,PREDICTION_VALUE_NOISE_HIGH)

                        # Better, "tail" approach is the Gumbel distribution
                        # http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gumbel.html
                        noise = np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)

                    # And boost bet/raise somewhat more or less, than the other actions.
                    if action in ALL_BETS_SET:
                        noise *= AGGRESSIVE_ACTION_NOISE_FACTOR

                    # Also, boost a *bet* value, if we stood pat last round. Why? Helps with snowing, etc.
                    # Boost only *bet* (not raise). And only if we are pat, and opponent is not.
                    if (action in ALL_BETS_SET) and not (action in ALL_RAISES_SET):
                        # Make sure that boost to bets is non-negative.
                        if BOOST_AGGRESSIVE_ACTION_NOISE:
                            noise = max(PREDICTION_VALUE_NOISE_AVERAGE, noise) 

                        # If we are pat, consider boosting the bet value. Too many checks after patting (not only on snow)
                        # (similarly, though less, tend to bet when we took fewer cards than apponent)
                        draws_ahead_boost = 3.0 * PREDICTION_VALUE_NOISE_AVERAGE +  max([0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA), np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA), np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)])
                        if BOOST_PAT_BET_ACTION_NOISE and cards_kept == 5 and opponent_cards_kept != 5:
                            if debug:
                                print('considering boosting the *bet* for pat hand. Opponent won\'t bet hand for us!')
                            noise += draws_ahead_boost * 2.0 * (cards_kept - opponent_cards_kept)
                            adjust_bet_better_draw = True
                        elif BOOST_PAT_BET_ACTION_NOISE and cards_kept > 2 and opponent_cards_kept < cards_kept:
                            if debug:
                                print('considering boosting the *bet* for drawing hand that is ahead. Opponent won\'t bet hand for us!')
                            noise += draws_ahead_boost * (cards_kept - opponent_cards_kept)
                            adjust_bet_better_draw = True

                    # NOTE: Do *not* apply noise, if already using a mixed model. That is noise enough.
                    if self.bets_output_array:
                        noise *= MULTIPLE_MODELS_NOISE_FACTOR

                    prediction[0] += noise
                value_predictions.sort(reverse=True)
                if debug:
                    print(value_predictions)

            """
            # Special case: if we can bet or check... and both values are negative... then boost the check.
            # Not uncommon, especially early in game training, to have negative expectation for all actions. 
            # If we can't fold... we should favor checking in these spots. Why bet, if betting has a known negative value?
            if len(value_predictions) == 2 and (CHECK_HAND in actions) and not (FOLD_HAND in actions):
                #print('Only bet and check options. Boost check, if both are negative...')
                if value_predictions[0][0] < 0.025 and value_predictions[1][0] < 0.025:
                    #print('Yep, both negative.')
                    for prediction in value_predictions:
                        action = prediction[1]
                        if action == CHECK_HAND:
                            noise = abs(prediction[0]) * 0.66666
                            #print('boost negative check value by %.3f' % noise)
                            prediction[0] += noise
            value_predictions.sort(reverse=True)
            """

            # Same for action%
            action_percentge = [[bets_vector[category_from_event_action(action) + 5] / max(np.sum(bets_vector[5:10]), 0.01), action, '%s: (%.1f%%)' % (actionName[action], bets_vector[category_from_event_action(action) + 5] / max(np.sum(bets_vector[5:10]), 0.01) * 100.0)] for action in actions]
            action_percentge.sort(reverse=True)
            if debug and FORMAT != 'nlh':
                print(action_percentge)
            
            if debug:
                print(value_predictions)

            # Save for debug
            self.bet_val_vector = [int(x * 100000) / 100000.0 for x in [(val - 2.0) for val in bets_vector[:5]]]
            if FORMAT == 'nlh':
                # Values for bet sizes
                self.act_val_vector = [int(x * 100000) / 100000.0 for x in [(val - 2.0) for val in bets_vector[5:13]]] 
                # Chips bet, odds, etc
                self.num_draw_vector = [int(x * 100000) / 100000.0 for x in [val for val in bets_vector[13:21]]]
            else:
                # Limit games, save act%
                self.act_val_vector = [int(x * 100000) / 100000.0 for x in [val/max(np.sum([val for val in bets_vector[5:10]]), 0.01) for val in bets_vector[5:10]]]
                # Save draw hand... for non-holdem limit
                if FORMAT == 'holdem':
                    self.num_draw_vector = []
                else:
                    self.num_draw_vector = [int(x * 100000) / 100000.0 for x in [(val - 2.0) for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]]

            # Now we have a choice.
            # A. Add noise to action values, and take action with highest post-noise value. This is a good option. Directly from RL
            # B. If good action% model... just take the action based on %
            best_action = None
            if (not adjusted_values_to_fix_impossibility) and (not adjust_bet_better_draw) and (not negative_river_call_value) and USE_ACTION_PERCENTAGE and self.use_action_percent_model and np.random.rand() <= ACTION_PERCENTAGE_CHOICE_RATE:
                # Sampled choice, from available actions, based on action% from neural net output.
                # NOTE: Need to explicitly round... to avoid annoying numpy/float32 issues
                probabilities = np.array([max(int(action_tuple[0] * 100000), 0) / 100000.0 for action_tuple in action_percentge])
                remainder = 1.0 - probabilities.sum()
                probabilities[0] += remainder
                choice = np.random.choice([action_tuple[1] for action_tuple in action_percentge], 
                                          p=probabilities)
                best_action = choice
                if debug:
                    print('~ using percent action choice ~')

                # Final Hack! We need should aim to avoid close folds pre-draw.
                # Thus, if close, try again.
                if round == PRE_DRAW_BET_ROUND and best_action == FOLD_HAND and retry == False and RETRY_FOLD_ACTION:
                    if debug:
                        'Retrying preflop fold action...'
                    return self.choose_action(actions=actions, round=round, bets_this_round = bets_this_round, 
                                              has_button = has_button, pot_size=pot_size, actions_this_round=actions_this_round,
                                              cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept, 
                                              debug = debug, retry = True, actions_whole_hand=actions_whole_hand)

                print('\n%s\n' % actionName[best_action])
                
                # TODO: We need action size, for NLH. Pure hack, to keep it moving!
                return (best_action, recommended_bet_size)
            
            # Highest-value action, after all possible adjustments.
            best_action = value_predictions[0][1]

            # Final Hack! We need should aim to avoid close folds pre-draw.
            # Thus, if close, try again.
            if round == PRE_DRAW_BET_ROUND and best_action == FOLD_HAND and retry == False and RETRY_FOLD_ACTION:
                if debug:
                    'Retrying preflop fold action...'
                return self.choose_action(actions=actions, round=round, bets_this_round = bets_this_round, 
                                          has_button = has_button, pot_size=pot_size, actions_this_round=actions_this_round,
                                          cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept, 
                                          debug = debug, retry = True, actions_whole_hand=actions_whole_hand)

            # Purely for debug
            #if debug:
            #    self.create_heuristic_action_distribution(round, bets_this_round = bets_this_round, has_button = has_button)
            if (best_action_no_noise != best_action) and debug:
                print('--> changed best action %s -> %s from tweaking!' % (actionName[best_action_no_noise], actionName[best_action]))

            #print(best_action)
            print('\n%s\n' % actionName[best_action])
            
            # Internal variable, for easy switch between learning model, and heuristic model below.
            if self.use_learning_action_model:
                # TODO: We need action size, for NLH. Pure hack, to keep it moving!
                return (best_action, recommended_bet_size)
        else:
            print('No *bets* output model specified (or not used) for player %s' % self.name)
        heuristic_action, bet_amount= self.choose_heuristic_action(allowed_actions = list(actions), 
                                                                   round = round, 
                                                                   bets_this_round = bets_this_round, 
                                                                   bets_sequence = bets_sequence,
                                                                   chip_stack = chip_stack,
                                                                   pot_size = pot_size,
                                                                   has_button = has_button, debug=debug)
        # NOTE: For backward compatible, can skip "bet_amount" if game not big-bet game
        return (heuristic_action, bet_amount)

    # Use known game information, and especially hand heuristic... to output probability preference for actions.
    # (bet_raise, check_call, fold, bet_amount)
    # TODO: Pass along other important hand aspects here... # of bets made, hand history, opponent draw #, etc
    def create_heuristic_action_distribution(self, round, bets_this_round = 0, bets_sequence = [], chip_stack = 0, pot_size = 0, has_button = True, debug = True):
        # Baseline is 2/2/0.5 bet/check/fold
        bet_raise = 2.0
        check_call = 2.0
        fold = 0.5 # 1.0
        bet_amount = 0.0 # optional output, only relevant for big-bet games

        # If player is on the button, he should check less, and bet more. 
        # TODO: This change a lot, once we look at previous draw & bet patterns.
        # NOTE: What we want to avoid is player constantly betting into a much stronger hand.
        if has_button:
            bet_raise = 2.0
            check_call = 1.0
            fold = 0.5

        # If NLH, bets_sequence -> use this for bet size faced
        bet_faced = 0.0
        raise_amount = 0.0
        min_bet = 0.0
        max_bet = 0.0
        reasonable_bet = 0.0 
        if FORMAT == 'nlh':
            print('heuristic action for nlh hand... bets_sequence %s [pot size %s]' % (bets_sequence, pot_size))

            # Bet faced is the previous bet, if any. Minimum raise requires us to bet (bet_faced + minimum raise amount)
            if bets_sequence:
                bet_faced = max(bets_sequence[-1], SMALL_BET_SIZE)
                raise_amount = bet_faced
                if len(bets_sequence) >= 2:
                    raise_amount = max(bets_sequence[-1] - bets_sequence[-2], SMALL_BET_SIZE)
            min_bet = bet_faced + raise_amount
            if round <= FLOP_ROUND:
                min_bet = max(min_bet, SMALL_BET_SIZE)
            else:
                min_bet = max(min_bet, BIG_BET_SIZE)
            max_bet = chip_stack
            min_bet = min(min_bet, max_bet)

            # Reasonable bet... is a normal big-bet game bet this will form the median bet size,
            # for a median hand [obviously, a pretty ridiculous concept]
            # If we bet (remember, we also check sometimes), reasonable bet --> 2x min bet, or 2/3 pot. 
            # NOTE: Beta distribution means we could bet more, or bet less.
            reasonable_bet = max(min(max(BIG_BET_SIZE * 2.0, pot_size * 0.66), max_bet), min_bet)

            print('--> if betting, need to decide bet size between (min %.1f, max %.1f, reasonable %.1f)' % (min_bet, max_bet, reasonable_bet))

        # See our value, and typical opponent hand value... adjust our betting pattern.
        # TODO: If NLH... we might want to adjust baseline based on amount bet. Or not!
        hand_value = self.heuristic_value
        baseline_value = baseline_heuristic_value(round, bets_this_round)
        print('hand_value: %.3f\tbaseline: %.3f (%.2f bets this round)' % (hand_value, baseline_value, bets_this_round))

        # TODO: If NLH... use bets sequence (total bet this round) and current bet faced... to tweak baseline
        # TODO: Separate mechanism for calling with pot odds to win, obviously (to increase call, decrease fold)
        # TODO: Third system to guess a Beta distribution for bets...
        # A. beta baseline
        # B. bet more with good hands, generally speaking
        # C. bet more as pot increases --> most important

        # For bet sizing, generate beta distribution fit from (sum of)
        # A. Narrow beta around 'reasonable_bet'
        # B. Sample of bets from minimum to allin.
        if FORMAT == 'nlh' and min_bet == max_bet:
            # If we are close to allin, no real choice, and skip ahead.
            print('only bet is allin!')
            bet_amount = max_bet
            print('choice: %s' % bet_amount)
        elif FORMAT == 'nlh':
            print('pot_size %s\tmin_bet %s\tmax_bet %s\treasonable_bet %s' % (pot_size, min_bet, max_bet, reasonable_bet))

            # Calculate beta-distribution parameters directly from mean and deviation
            reasonable_beta = generate_beta(mean=reasonable_bet, stdev=min(2.0 * min_bet, math.sqrt(max_bet - min_bet)), scale=MAX_STACK_SIZE)
            alpha, beta, scale, loc = reasonable_beta
            print(reasonable_beta)

            # Grab 50-size sample from our desired bet size
            sample_reasonable_beta = scipy.stats.beta.rvs(alpha, beta, scale=scale, size=50) + loc
            sample_reasonable_beta = np.unique(np.clip(sample_reasonable_beta, min_bet, max_bet)) # snip out-of-range bets
            #print('%s reasonablechoices from beta %s' % (len(sample_reasonable_beta), sample_reasonable_beta))

            # Fixed number of samples from the full bets amount (~10 samples)
            sample_all_bets = sample_bets_range(pot_size=pot_size, min_bet=min_bet, max_bet=max_bet) 
            #print('sample all bets: %s' % sample_all_bets)

            # OK... now we have a sample of possible bets, (heavily) weighed toward the "reasonable bet"
            # We can just sample from this set, or fit a beta distribution.
            weighted_bets_sample = np.sort(np.append(sample_reasonable_beta, sample_all_bets))

            print('With reasonable bets and all_bets, sample size %s: %s' % (weighted_bets_sample.shape, weighted_bets_sample))
            bet_amount = np.random.choice(weighted_bets_sample, 1)
            print('choice: %s' % bet_amount)


        if FORMAT == 'holdem' or FORMAT == 'nlh':
            if debug:
                print('Player %s (button %d) value %.2f, vs baseline %.2f %s + %s' % (self.name, has_button, hand_value, baseline_value, 
                                                                                  hand_string(self.cards), 
                                                                                  [hand_string(self.holdem_hand.community.flop), hand_string(self.holdem_hand.community.turn), hand_string(self.holdem_hand.community.river)]))
        else:
            if debug:
                print('Player %s (button %d) value %.2f, vs baseline %.2f %s' % (self.name, has_button, hand_value, baseline_value, hand_string(self.cards)))
        
        if hand_value > baseline_value:
            # Dramatically increase our bet/raise frequency, if hand is better than baseline.
            bet_increase = 3.0 / 0.10 * (hand_value - baseline_value)
            #print('increasing bet/raise by %.2f' % bet_increase)
            bet_raise += bet_increase
            
            fold_decrease = 0.5 / 0.10 * (hand_value - baseline_value)
            #print('decreasing fold by %.2f' % fold_decrease)
            fold -= fold_decrease
        elif hand_value < baseline_value:
            # Quickly stop raising, if our hand is below expectation
            bet_decrease = 1.0 / 0.10 * (hand_value - baseline_value)
            #print('decreasing bet/raise by %.2f' % bet_decrease)
            bet_raise += bet_decrease
            
            # Start to fold more, especially if we are 0.20 or more behind expect opponent hand (2-card draw vs pat hand, etc)
            # Fold more if NLH. 
            fold_increase = 0.5 / 0.10 * (hand_value - baseline_value)
            if FORMAT == 'nlh':
                fold_increase *= 2.0 
            #print('increasing fold by %.2f' % fold_increase)
            fold -= fold_increase      

        # Decrease folding as the pot grows... shift these to calls.
        # NOTE: Also balances us out, in terms of not raising and folding too much. We won't over-fold to 4bet, etc
        if bets_this_round > 1 and FORMAT != 'nlh':
            fold_decrease = 0.5 / 2 * (max(bets_this_round, 5) - 1) 
            #print('decreaseing fold by %.2f since much action this street.' % fold_decrease)
            fold -= fold_decrease

            #print('incrase call by similar amount %.2f' % (2 * fold_decrease))
            check_call += 2 * fold_decrease

        # We should often consider some chance at a really aggressive action, even in bad spots.
        # For example, on busted river... or betting second, when first player has been weak...
        raise_minimum = 0.0
        if bets_this_round == 0:
            raise_minimum += 0.5
            if has_button and round >= DRAW_2_BET_ROUND:
                raise_minimum += 0.5
        if has_button and bets_this_round < 2:
            raise_minimum += 0.5

        #if raise_minimum and raise_minimum > bet_raise:
        #    print('resetting mimimum raise to %.2f. Fortune favors the bold!' % max(bet_raise + raise_minimum, raise_minimum))

        return (max(bet_raise + raise_minimum, raise_minimum), check_call, max(fold, 0.0), math.floor(bet_amount))
        

    # Computes a distribution over actions, based on (hand_value, round, other info)
    # Then, probabilistically chooses a single action, from the distribution.
    # NOTE: allowed_actions needs to be a list... so that we can match probabilities for each.
    def choose_heuristic_action(self, allowed_actions, round, bets_this_round = 0, bets_sequence = [], chip_stack = 0, pot_size = 0, has_button = True, debug=True):
        print('Allowed actions %s' % ([actionName[action] for action in allowed_actions]))

        # First, create a distribution over actions.
        # NOTE: Resulting distribution is *not* normalized. Could return (3, 2, 0.5, 425.7)
        (bet_raise, check_call, fold, bet_amount) = self.create_heuristic_action_distribution(round, 
                                                                                              bets_this_round = bets_this_round,
                                                                                              bets_sequence = bets_sequence,
                                                                                              chip_stack = chip_stack,
                                                                                              pot_size = pot_size,
                                                                                              has_button = has_button, debug=debug)

        # Normalize so sum adds to 1.0
        action_sum = bet_raise + check_call + fold
        assert action_sum > 0.0, 'actions sum to impossible number %s' % [bet_raise, check_call, fold]

        bet_raise /= action_sum
        check_call /= action_sum
        fold /= action_sum

        # Match outputs above to actual game actions. Assign values directly to action.probability
        if debug:
            print('(bet/raise %.2f, check/call %.2f, fold %.2f, amount %.1f)' % (bet_raise, check_call, fold, bet_amount))
        
        # Good for easy lookup of "are we allowed to bet here"?
        all_actions_set = set(allowed_actions)
        print('allowed actions: %s' % all_actions_set)

        action_probs = []
        for action in allowed_actions:
            probability = 0.0
            if action in ALL_CALLS_SET:
                #print('CALL take all of the check/call credit: %s' % check_call)
                probability += check_call
                
                # if we are not allowed to bet or raise... take that credit also. [betting capped, etc]
                if not(set(ALL_BETS_SET) & all_actions_set):
                    #print('since no BET/RAISE, CALL takes all bet/raise credit: %s' % bet_raise)
                    probability += bet_raise
            elif action in ALL_BETS_SET:
                #print('RAISE take all of the bet/raise credit: %s' % bet_raise)
                probability += bet_raise
            elif action == FOLD_HAND:
                #print('FOLD take all of the fold credit: %s' % fold)
                probability += fold
            elif action == CHECK_HAND:
                #print('CHECK take all of the check/call credit: %s' % check_call)
                probability += check_call

                # If we can't fold... credit goes here.
                if not(FOLD_HAND in all_actions_set):
                    #print('Since no FOLD, CHECK takes all fold credit: %s' % fold)
                    probability += fold

                # If we can't bet (already allin, etc)... credit goes here.
                if not(set(ALL_BETS_SET) & all_actions_set):
                    #print('since no BET/RAISE (maybe allin already), CHECK takes all bet/raise credit: %s' % bet_raise)
                    probability += bet_raise
            else:
                assert False, 'Unknown possible action %s' % actionName[action]
                
            action_probs.append(probability)
                
        # Probabilities should add up to 1.0...
        action_distribution = action_probs

        # Then sample a single action, from this distribution.
        choice_action = np.random.choice(len(allowed_actions), 1, p = action_distribution)
        #print('choice: %s' % allowed_actions[choice_action[0]])

        # Bet amount only applies to bets and raises...
        return (allowed_actions[choice_action[0]], bet_amount)

    # Nobody said that... some actions can't be more random than others!
    def choose_random_action(self, actions, round):
        if actions:
            random_choice = random.sample(actions, 1)
            # Act here, if we re-sample, to fold less, etc
            if random_choice[0] == FOLD_HAND and random.random() <= RE_CHOOSE_FOLD_DELTA:
                print('re-considering FOLD')
                return self.choose_random_action(actions, round)
            return random_choice[0]
        return None

# Over-writes bet & draw selections decision, with human prompt
class TripleDrawHumanPlayer(TripleDrawAIPlayer):
    # Make sure that automatically know that agent is human!
    def __init__(self):
        TripleDrawAIPlayer.__init__(self)
        self.is_human = True

    #def choose_action(self, actions, round, bets_this_round = 0, 
    #                  has_button = True, pot_size=0, actions_this_round=[], actions_whole_hand=[],
    #                  cards_kept=0, opponent_cards_kept=0, ):
    def choose_action(self, actions, round, bets_this_round = 0, bets_sequence = [], chip_stack = 0,
                      has_button = True, pot_size=0, actions_this_round=[], actions_whole_hand=[],
                      cards_kept=0, opponent_cards_kept=0, 
                      debug = True, retry = False):
        # Write up bets made so far, to (optionally) show the player
        bets_string = ''
        # print('actions_this_round: |%s|' % actions_this_round)
        if FORMAT == 'nlh':
            bets_string = encode_big_bets_string(actions_this_round)
        elif actions_this_round and isinstance(actions_this_round, basestring):
            print('detected actions_this_round is string: %s' % actions_this_round)
            for action in actions_this_round:
                if action == 'r':
                    bets_string += '1'
                elif action == 'c':
                    bets_string += '0'
                else:
                    # Don't encode non-bets
                    continue
            print(bets_string)
        else:
            bets_string = encode_limit_bets_string(actions_this_round)

        # And for all bets made this hand
        all_rounds_bets_string = ''
        # print('actions_whole_hand %s' % actions_whole_hand)
        if FORMAT == 'nlh':
            all_rounds_bets_string = encode_big_bets_string(actions_whole_hand)
        elif actions_whole_hand and isinstance(actions_whole_hand, basestring):
            print('detected actions_whole_hand is string: %s' % actions_whole_hand)
            for action in actions_whole_hand:
                if action == 'r':
                    all_rounds_bets_string += '1'
                elif action == 'c':
                    all_rounds_bets_string += '0'
                else:
                    # Don't encode non-bets
                    continue
            print(all_rounds_bets_string)
        else:
            all_rounds_bets_string = encode_limit_bets_string(actions_whole_hand)

        print('Choosing among actions %s for round %s' % ([actionName[action] for action in actions], round))
        if SHOW_HUMAN_DEBUG and not (FORMAT == 'holdem' or FORMAT == 'nlh'):
            # First show the context for this hand.
            num_draws_left = 3
            if round == PRE_DRAW_BET_ROUND:
                num_draws_left = 3
            elif round == DRAW_1_BET_ROUND:
                num_draws_left = 2
            elif round == DRAW_2_BET_ROUND:
                num_draws_left = 1
            elif round == DRAW_3_BET_ROUND:
                 num_draws_left = 0

            if (num_draws_left >= 1):
                hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
            else:
                hand_string_dealt = hand_string(self.draw_hand.final_hand)

            # Input related to the hand
            cards_input = cards_input_from_string(hand_string_dealt, include_num_draws=True, 
                                                  num_draws=num_draws_left, include_full_hand = True, 
                                                  include_hand_context = False)

            # TODO: This should be a util function.
            bets_string = ''
            for action in actions_this_round:
                if action.type in ALL_BETS_SET:
                    bets_string += '1'
                elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                    bets_string += '0'
                else:
                    # Don't encode non-bets
                    continue
            
            # Now hand context
            if bets_string == '':
                print('\nfirst to act\n')

            print('context %s' % ([hand_string_dealt, num_draws_left, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept]))
            #print(actions_this_round)
            # Hand baseline, purely for debug
            self.create_heuristic_action_distribution(round, bets_this_round = bets_this_round, has_button = has_button)
        elif FORMAT == 'holdem' or FORMAT == 'nlh':
            # Save these, in case needed for recording data.
            self.cards = self.holdem_hand.dealt_cards
            self.flop = self.holdem_hand.community.flop
            self.turn = self.holdem_hand.community.turn
            self.river = self.holdem_hand.community.river
            
            # show basic debug for human hand...
            print('context %s' % ([hand_string(self.cards), hand_string(self.flop), hand_string(self.turn), hand_string(self.river)]))
            if FORMAT == 'nlh':
                print('context %s' % ([has_button, pot_size, bets_string, cards_kept, opponent_cards_kept,  all_rounds_bets_string]))
            
        # Prompt user for action, and see if it parses...
        # Usually only considers the first character. Unless NLH then expects bet size after the "b/r"
        user_action = None
        while not user_action:
            user_move_string = raw_input("Please select action %s -->   " % [actionName[action] for action in actions])
            print('User action: |%s|' % user_move_string)
            if len(user_move_string) > 0:
                user_char = user_move_string.lower()[0]
                bet_size = 0 # placeholder
                if user_char == 'c':
                    print('action check/call')
                    allowed_actions = set(list(ALL_CALLS_SET) + [CHECK_HAND])
                elif user_char == 'b' or user_char == 'r':
                    print('action bet/raise')
                    allowed_actions = ALL_BETS_SET

                    # If we are capped out, and can't raise but only call... parse that too.
                    if len(list(set.intersection(set(actions), ALL_BETS_SET))) == 0:
                        allowed_actions = set(list(ALL_CALLS_SET))
                    elif FORMAT == 'nlh':
                        # If NLH, expect to match a number, after the b/r
                        bet_size_match = re.match('\S*([0-9]*)+', user_move_string[1:])
                        try: 
                            if bet_size_match:
                                bet_size = int(bet_size_match.group(0))
                                print('bet size %s' % bet_size)
                        except ValueError:
                            bet_size = 0
                elif user_char == 'f':
                    print('action FOLD')
                    allowed_actions = set([FOLD_HAND])
                else:
                    print('unparseable action... try again')
                    continue

                user_actions = list(set.intersection(set(actions), allowed_actions))
                if (not user_actions) or len(user_actions) != 1 or (FORMAT == 'nlh' and user_actions[0] in ALL_BETS_SET and not bet_size):
                    print('unparseable action... try again')
                    continue
                user_action = (user_actions[0], bet_size)

        # Unless we check what AI does here, don't put in the vector values
        self.bet_val_vector = []
        self.act_val_vector = []
        self.num_draw_vector = []

        # Single move, that user selected!
        return user_action
        
    # Human must draw his cards, also! [skips draw recommendations]
    def draw_move(self, deck, num_draws = 1, debug = True,  draw_recommendations = None):
        # Reduce debug, if opponent is human, and could see.
        if self.opponent and self.opponent.is_human:
            debug = False

        hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
        if debug:
            print('dealt %s for draw %s' % (hand_string_dealt, num_draws))

        draw_string = ''
        while True:
            # For player convenience, tag draw cards 1-5 instead of 0-4
            user_move_string = raw_input("Choose cards to draw\n%s -->   " % ['%d: %s' % (i+1, self.draw_hand.dealt_cards[i]) for i in range(5)])
            if user_move_string == '' or user_move_string == 'p':
                draw_string = ''
                break
            for char in user_move_string:
                if char.isdigit():
                    # From player readability, now need to map 2->1, 1->0, etc
                    #draw_string += char
                    implied_integer = int(char) - 1
                    if implied_integer >= 0 and implied_integer <= 4:
                        draw_string += str(implied_integer)
                    else:
                        continue
                else:
                    continue
            if draw_string:
                break

        print('Got usable draw string |%s|' % draw_string)

        print('\nDrawing %d cards.\n' % len(draw_string))

        discards = self.draw_hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        self.draw_hand.deal(new_cards, final_hand=True)

        # Record current setting of these values...
        # NOTE: heuristic value... is before we got our cards.
        self.num_cards_kept = 5 - len(discards)

# Given a partial hand string, parse cards, and set in the deck provided, at given offset.
# NOTE: Pass '0' to expected_length if no requirement
# NOTE: _OFFSET counts in deck counting forward. In real deck, we pop backwards... make sure to reverse deck *once* after setup.
# TODO: Move this to utils.
BLIND_HAND_OFFSET = 0
BUTTON_HAND_OFFSET = 2
BOARD_HAND_OFFSET = 4
def set_deck_with_cards_string(deck, card_string, expected_length, deck_offset):
    # Strip '/' for flop/turn/river split.
    # TODO: Include characters to strip in params
    card_array = card_array_from_string(card_string.replace('/', '').strip())
    if expected_length:
        assert len(card_array) == expected_length, 'incorrect parsing as expected length %d |%s|' % (expected_length, blind_hand_string)
    # Set deck cards!
    for index in range(len(card_array)):
        card = card_array[index]
        deck.set_card(card, pos=deck_offset + index)


# As simply as possible, simulate a full round of triple draw. Have as much logic as possible, contained in the objects
# Actors:
# cashier -- evaluates final hands
# deck -- dumb deck, shuffled once, asked for next cards
# dealer -- runs the game. Tracks pot, propts players for actions. Decides when hand ends.
# players -- acts directly on a poker hand. Makes draw and betting decisions... when propted by the dealer
# NOTE: We can also optionally supply players' hands, the board, and bets_string (or some subset of these)
# NOTE: We do no error checking on validity of player cards given. Will fail later, or if double cards, some will be invalid.
def game_round(round, cashier, player_button=None, player_blind=None,
               button_hand_string = None, blind_hand_string = None,
               board_string = None, bets_string = None,
               csv_writer=None, csv_header_map=None,
               player_button_average=0.0, player_blind_average=0.0):
    print '\n-- New Round %d --\n' % round
    # Performance suffers... a lot, over time. Can we improve this with garbage collection?
    # NOTE: Not really, but doesn't hurt. If you want to improve performance, need to purge the theano-cache
    # "theano-cache purge" --> on GPU
    # "theano-cache clear" --> enough on CPU
    # TODO: How to do this, in-code??
    if round > 0 and round % 100 == 1:
        now = time.time()
        print('--> wait for manual garbage collection...')
        gc.collect()
        print ('--> gc %d took %.1f seconds...\n' % (round, time.time() - now))

    # Shuffle deck *before* any hand setup. 
    deck = PokerDeck(shuffle=True)

    # If we are passed strings for p1_hand, p2_hand, and/or the board...
    # ...parse those cards. And set them in the correct deck order.
    if blind_hand_string:
        set_deck_with_cards_string(deck, blind_hand_string, 2, BLIND_HAND_OFFSET)
    if button_hand_string:
        set_deck_with_cards_string(deck, button_hand_string, 2, BUTTON_HAND_OFFSET)
    if board_string:
        set_deck_with_cards_string(deck, board_string, 0, BOARD_HAND_OFFSET)

    
    # Set BB's hand for testing.    
    #deck.set_card(Card(suit=SPADE, value=Jack), pos=2)
    #deck.set_card(Card(suit=SPADE, value=Ten), pos=3)
    #deck.set_card(Card(suit=SPADE, value=Jack), pos=2) # set it twice. Why? pop/push problem if Ax at position 0 or 1!

    """
    # NOTE: This is the spot to insert a deck setup, if needed for testing
    deck.set_card(Card(suit=DIAMOND, value=Five), pos=0)
    deck.set_card(Card(suit=DIAMOND, value=Jack), pos=1)
    deck.set_card(Card(suit=SPADE, value=Nine), pos=2)
    deck.set_card(Card(suit=CLUB, value=Ace), pos=3)
    deck.set_card(Card(suit=CLUB, value=Deuce), pos=4)
    deck.set_card(Card(suit=CLUB, value=Ten), pos=5)
    deck.set_card(Card(suit=CLUB, value=Trey), pos=6)
    deck.set_card(Card(suit=CLUB, value=Eight), pos=7)
    deck.cards.reverse()
    """

    # NOTE: Cards are popped from the back of the deck... but just set them in the front, then reverse()
    deck.cards.reverse()

    # HACK: Set players to "imitate CFR ratio" some X% of the time
    for player in [player_blind, player_button]:
        if IMITATE_CFR_AGGRO_STRATEGY and np.random.random() < IMITATE_CFR_BETTING_PERCENTAGE: 
            player.imitate_CFR_betting = True
            print('Setting player %s to use IMITATE_CFR_BETTING for next hand.' % (player.player_tag()))
        elif IMITATE_CFR_AGGRO_STRATEGY:
            # Make sure to reset the value to false, if we are flipping coins and lost.
            player.imitate_CFR_betting = False
            print('Setting player %s with IMITATE_CFR_BETTING *off* for next hand.' % (player.player_tag()))

    dealer = TripleDrawDealer(deck=deck, player_button=player_button, player_blind=player_blind, format=FORMAT)
    dealer.play_single_hand(bets_string=bets_string) # Pass empty bets string, to allow players to make actual choices

    winners = dealer.get_hand_result(cashier)
    final_bets = {player_button.name: player_button.bet_this_hand, player_blind.name: player_blind.bet_this_hand}

    # Print out all moves in the hand. And save them to CSV.
    # NOTE: This CSV data should be useful to train a move-action model
    # NOTE: This means, over time, working to include sufficient information for:
    # conv(xCards + xNumDraws + xButton + xPot + xActions + xHistory)
    # xButton = are we button? 
    # xPot = pot size (going into the action)
    # xActions = 011 -> check, bet, raise
    # This information is the most important. Number of draws by opponent matters also, as well as previous bets...
    print('\nFull hand history...')
    # Shared object, so that if we generate "allin value" simulation for actions... don't recompute exact same
    now = time.time() 
    allin_values_cache = HoldemValuesCache()
    for event in dealer.hand_history:
        # Also, pass running average, to the update (average is per-player). 
        # NOTE: It's a hack, but good to see running stats for that player so far.
        if event.position:
            running_average = player_button_average
        else:
            running_average = player_blind_average
        running_average

        # Back-update hand result, and decision result for all moves made
        event.update_result(winners, final_bets, hand_num=round, running_average=running_average)
        print(event)
        if csv_header_map:
            event_line = event.csv_output(csv_header_map, allin_cache=allin_values_cache)
            print(event_line)

        # Write events, for training.
        # TODO: Include draw events.
        if csv_writer:
            csv_writer.writerow(event_line)
    # TODO: Flush buffer here?

    # How long did it take to calculate & print everything?

    # Game log, in ACPC format.
    # TODO: If desired bets supplied via "bet_string", did we get the same thing back??
    game_log = encode_bets_string(dealer.hand_history, format=FORMAT)
    print('game log:\n%s' % game_log) 
    if bets_string:
        if bets_string != game_log:
            print('compare to ACPC bets_string:\n%s' % bets_string)
            print('\n--> Bets strings do not match! (issue with allins?)\n')
        # assert game_log == bets_string, 'Bets strings do not match! (issue with allins?)'
    print('%.2fs to write CSV (simulate allin values, etc)' % (time.time() - now))

    # If we are tracking results... return results (wins/losses for player by order
    bb_result = dealer.hand_history[0].margin_result
    sb_result = dealer.hand_history[1].margin_result
    return (bb_result, sb_result)

# Generate neural net models, for players:
# return (player_one, player_two)
def generate_player_models(draw_model_filename=None, holdem_model_filename=None,
                           bets_model_filename=None, old_bets_model_filename=None, other_old_bets_model_filename=None,
                           human_player=None, compare_models=None):
    # We initialize deck, and dealer, every round. But players kept constant, and reset for each trial.
    # NOTE: This can, and will change, if we do repetative simulation, etc.
    player_one = TripleDrawAIPlayer()
    player_two = TripleDrawAIPlayer()

    # Optionally, compete against human opponent.
    if human_player:
        player_two = TripleDrawHumanPlayer()

    # For easy looking of 'is_human', etc
    player_one.opponent = player_two
    player_two.opponent = player_one

    # Test the model, by giving it dummy inputs
    # Test cases -- it keeps the two aces. But can it recognize a straight? A flush? Trips? Draw?? Two pair??
    test_cases_draw = [['As,Ad,4d,3s,2c', 1], ['As,Ks,Qs,Js,Ts', 2], ['3h,3s,3d,5c,6d', 3],
                       ['3h,4s,3d,5c,6d', 2], ['2h,3s,4d,6c,5s', 1], ['3s,2h,4d,8c,5s', 3],
                       ['8s,Ad,Kd,8c,Jd', 3], ['8s,Ad,2d,7c,Jd', 2], ['2d,7d,8d,9d,4d', 1]] 
    test_cases_holdem = [['Ad,Ac', '[]', '', ''], # AA preflop
                         ['[8d,5h]','[Qh,9d,3d]','[Ad]','[7c]'], # missed draw
                         ['4d,5d', '[6d,7d,3c]', 'Ad', ''], # made flush
                         ['7c,9h', '[8s,6c,Qh]', '', ''], # open ended straight draw
                         ['Ad,Qd', '[Kd,Td,2s]', '3s', ''], # big draw
                         ['7s,2h', '', '', ''], # weak hand preflop
                         ['Ts,Th', '', '', ''], # good hand preflop
                         ['9s,Qh', '', '', ''], # average hand preflop
                         ]
    if FORMAT == 'holdem' or FORMAT == 'nlh':
        test_cases = test_cases_holdem
    else:
        test_cases = test_cases_draw

    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[1])

    # NOTE: Num_draws and full_hand must match trained model.
    # TODO: Use shared environemnt variables...
    if FORMAT == 'holdem' or FORMAT == 'nlh':
        test_batch = np.array([holdem_cards_input_from_string(case[0], case[1], case[2], case[3]) for case in test_cases], np.int32)
    else:
        test_batch = np.array([cards_input_from_string(hand_string=case[0], 
                                                       include_num_draws=True, num_draws=case[1],
                                                       include_full_hand = True, 
                                                       include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], np.int32)
    print('Test batch dimension')
    print(test_batch.shape)

    # If model file provided, unpack model, and create intelligent agent.
    # TODO: 0.0-fill parameters for model, if size doesn't match (26 vs 31, etc)
    # TODO: model loading should be a function!
    output_layer = None
    input_layer = None
    if draw_model_filename and os.path.isfile(draw_model_filename):
        print('\nExisting model in file %s. Attempt to load it!\n' % draw_model_filename)
        all_param_values_from_file = np.load(draw_model_filename)
        print('loaded %s params' % len(all_param_values_from_file))
        expand_parameters_input_to_match(all_param_values_from_file, zero_fill = True)

        # Size must match exactly!
        output_layer, input_layer, layers  = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )

        #print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file)
        predict_model(output_layer=output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled model...')
    else:
        print('No model provided or loaded. Expect error if model required. %s', draw_model_filename)

    # Similarly, unpack model for Holdem, if provided.
    holdem_output_layer = None
    holdem_input_layer = None
    if holdem_model_filename and os.path.isfile(holdem_model_filename):
        print('\nExisting holdem model in file %s. Attempt to load it!\n' % holdem_model_filename)
        all_param_values_from_file = np.load(holdem_model_filename)
        expand_parameters_input_to_match(all_param_values_from_file, zero_fill = True)

        for layer_param in all_param_values_from_file:
            print(layer_param)
            print(layer_param.shape)
            print('---------------')

        # Size must match exactly!
        holdem_output_layer, holdem_input_layer, holdem_layers  = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )

        #print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(holdem_output_layer, all_param_values_from_file)

        predict_model(output_layer=holdem_output_layer, test_batch=test_batch, format = FORMAT)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled model...')
    else:
        print('No model provided or loaded. Expect error if model required. %s', holdem_model_filename)

    # If supplied, also load the bets model. conv(xCards + xNumDraws + xContext) --> values for all betting actions
    bets_output_layer = None
    bets_input_layer = None
    bets_layers = None
    if bets_model_filename and os.path.isfile(bets_model_filename):
        print('\nExisting *bets* model in file %s. Attempt to load it!\n' % bets_model_filename)
        bets_all_param_values_from_file = np.load(bets_model_filename)
        print('loaded %s params' % len(bets_all_param_values_from_file))
        expand_parameters_input_to_match(bets_all_param_values_from_file, zero_fill = True)

        # Size must match exactly!
        # HACK: Automatically switch to Dense (DNN) model, if size of input...
        # TODO: Not that this is dense model!
        if len(bets_all_param_values_from_file) == 6:
             bets_output_layer, bets_input_layer, bets_layers  = build_fully_connected_model(
                 HAND_TO_MATRIX_PAD_SIZE, 
                 HAND_TO_MATRIX_PAD_SIZE,
                 32,
                 )
        elif len(bets_all_param_values_from_file) == 16:
            bets_output_layer, bets_input_layer, bets_layers  = build_nopool_model(
                 HAND_TO_MATRIX_PAD_SIZE, 
                 HAND_TO_MATRIX_PAD_SIZE,
                 32,
                 num_filters=64,
                 )
        else:
            bets_output_layer, bets_input_layer, bets_layers  = build_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )

        #print('filling model with shape %s, with %d params' % (str(bets_output_layer.get_output_shape()), len(bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(bets_output_layer, bets_all_param_values_from_file)
        predict_model(output_layer=bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *bets* model provided or loaded. Expect error if model required. %s', bets_model_filename)

    # If supplied "old" bets layer supplied... then load it as well.
    old_bets_output_layer = None
    old_bets_input_layer = None
    old_bets_layers = None
    if old_bets_model_filename and os.path.isfile(old_bets_model_filename):
        print('\nExisting *old bets* model in file %s. Attempt to load it!\n' % old_bets_model_filename)
        old_bets_all_param_values_from_file = np.load(old_bets_model_filename)
        expand_parameters_input_to_match(old_bets_all_param_values_from_file, zero_fill = True)

        # Size must match exactly!
        # HACK: Automatically switch to Dense (DNN) model, if size of input...
        # TODO: Not that this is dense model!
        if len(old_bets_all_param_values_from_file) == 6:
             old_bets_output_layer, old_bets_input_layer, old_bets_layers  = build_fully_connected_model(
                 HAND_TO_MATRIX_PAD_SIZE, 
                 HAND_TO_MATRIX_PAD_SIZE,
                 32,
                 )
        else:
            old_bets_output_layer, old_bets_input_layer, old_bets_layers = build_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )
        #print('filling model with shape %s, with %d params' % (str(old_bets_output_layer.get_output_shape()), len(old_bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(old_bets_output_layer, old_bets_all_param_values_from_file)
        predict_model(output_layer=old_bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *old bets* model provided or loaded. Expect error if model required. %s', old_bets_model_filename)

    # Lastly, load a third model...
    other_old_bets_output_layer = None
    other_old_bets_input_layer = None
    other_old_bets_layers = None
    if other_old_bets_model_filename and os.path.isfile(other_old_bets_model_filename):
        print('\nExisting *old bets* model in file %s. Attempt to load it!\n' % other_old_bets_model_filename)
        other_old_bets_all_param_values_from_file = np.load(other_old_bets_model_filename)
        expand_parameters_input_to_match(other_old_bets_all_param_values_from_file, zero_fill = True)

        # Size must match exactly!
        if len(other_old_bets_all_param_values_from_file) == 6:
             other_old_bets_output_layer, other_old_bets_input_layer, other_old_bets_layers = build_fully_connected_model(
                 HAND_TO_MATRIX_PAD_SIZE, 
                 HAND_TO_MATRIX_PAD_SIZE,
                 32,
                 )
        else:
            other_old_bets_output_layer, other_old_bets_input_layer, other_old_bets_layers = build_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )
        #print('filling model with shape %s, with %d params' % (str(other_old_bets_output_layer.get_output_shape()), len(other_old_bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(other_old_bets_output_layer, other_old_bets_all_param_values_from_file)
        predict_model(output_layer=other_old_bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *old bets* model provided or loaded. Expect error if model required. %s', other_old_bets_model_filename)

    # Add models, to players.

    # Player 1 plays the latest model... or a mixed bag of models, if provided. (unless we do A/B testing, in which case latest model only)
    # TODO: Perform massive cleanup, removing passing input, output layers for everything... just layers[0] and layers[-1] should suffice.
    player_one.output_layer = output_layer
    player_one.input_layer = input_layer
    player_one.holdem_output_layer = holdem_output_layer
    player_one.holdem_input_layer = holdem_input_layer
    player_one.bets_output_layer = bets_output_layer
    player_one.bets_input_layer = bets_input_layer

    # HACK: Are we using a dense model, for A vs B comparison (or A vs human)
    if bets_layers and len(bets_layers) <= 6:
        player_one.is_dense_model = True

    # enable, to make betting decisions with learned model (instead of heurstics)
    player_one.use_learning_action_model = True

    # Use action% for opponent... but only against human player. Confusing... but not really.
    if (human_player) and USE_ACTION_PERCENTAGE:
        player_one.use_action_percent_model = True
    elif USE_ACTION_PERCENTAGE and USE_ACTION_PERCENTAGE_BOTH_PLAYERS:
        # HACK: What if we want both players to use action percentage model?
        # TODO: Fix this better, or from command line.
        player_one.use_action_percent_model = True

    # Use array of all three models... unless we are comparing A vs B
    if compare_models:
        print('In compare_models mode, player_one uses the bets output layer only.')
        assert not human_player, 'Can not have both compare_models and human player.'
        # If given a tag via args, use it for player_one.
        if args.CNN_model_tag:
            player_one.tag = args.CNN_model_tag

    elif USE_MIXED_MODEL_WHEN_AVAILABLE and (old_bets_output_layer or other_old_bets_output_layer):
        player_one.bets_output_array = []
        player_one.bets_output_array.append([bets_output_layer, bets_input_layer]) # lastest model
        if old_bets_output_layer:
            player_one.bets_output_array.append([old_bets_output_layer, old_bets_input_layer])
        if other_old_bets_output_layer:
            player_one.bets_output_array.append([other_old_bets_output_layer, other_old_bets_input_layer])
        print('loaded player_one with %d-mixed model!' % len(player_one.bets_output_array))

    # Player 2 plays the least late model... unless player 1 is playing a mixed bag.
    # NOTE: This sounds confusing, but is not. We need to test vs human (with mixed model, if given)
    # Otherwise, we want to test the latest model, against the mix. Or the latest model, against the oldest given.
    player_two.output_layer = output_layer
    player_two.input_layer = input_layer
    player_two.holdem_output_layer = holdem_output_layer
    player_two.holdem_input_layer = holdem_input_layer
    if not compare_models:
        player_two.bets_output_layer = bets_output_layer
        player_two.bets_input_layer = bets_input_layer
    else:
        # If given a tag via args, use it for player_one.
        if args.CNN_old_model_tag:
            player_two.tag = args.CNN_old_model_tag
    # enable, to make betting decisions with learned model (instead of heurstics)
    player_two.use_learning_action_model = True

    # Use action percent model to choose actions.... but only for 2nd player (latest model)
    # NOTE: Can thus test value-action model vs action% model. 
    # TODO: If this is an issue, use command line flag. 
    if USE_ACTION_PERCENTAGE and (not human_player):
        player_two.use_action_percent_model = True

    # If we want to supply "old" model, for new CNN vs old CNN. 
    if (not human_player) and old_bets_output_layer and (not player_one.bets_output_array):
        player_two.bets_output_layer = old_bets_output_layer
        player_two.bets_input_layer = old_bets_input_layer
        player_two.old_bets_output_model = True

        # Are we using dense model (DNN) for A vs B comparison?
        if old_bets_layers and len(old_bets_layers) <= 6:
            player_two.is_dense_model = True

    # Use the "other" "old" model, if provided
    if (not human_player) and other_old_bets_output_layer and (not player_one.bets_output_array):
        player_two.bets_output_layer = other_old_bets_output_layer
        player_two.bets_input_layer = other_old_bets_input_layer
        player_two.other_old_bets_output_model = True

        # Are we using dense model (DNN) for A vs B comparison?
        if old_bets_layers and len(old_bets_layers) <= 6:
            player_two.is_dense_model = True

    # Return the two players, initialized with correct models, or mixes of models...
    return (player_one, player_two)

# ACPC line at a time. Loads everything, from hand randomness, to bets, to player names.
# 1:b333c/kk/kb333b1167c/b4500f:Jh8h|8d4h/Qc3h2h/5h/2c:1500:-1500:Slumbot|moscow25
# ...
# 19:b444c/kb444c/b312f:KsJc|Th7s/Js8c4c/Qc:888:-888:Slumbot|moscow25
# TODO: Even better, to turn this into a generator... especially for large data files.
acpc_line_regex = re.compile(r'(?:STATE)?:?(\d+):([^:]*):([^|]*)\|([^|/]*)([^:]*):(-?\d+)[:\|](-?\d+):([^|]*)\|([^|]*)')
# (num, bets, p1_hand, p2_hand, board, p1_result, p2_results, p1_name, p2_name)
def load_hand_history(hand_history_filename):
    line_reader = open(hand_history_filename, 'rU')
    processed_lines = []
    for line in line_reader:
        print(line)
        hand_result = acpc_line_regex.match(line.strip())
        chomped_line = hand_result.groups()
        print(chomped_line)

        # NOTE: If doing a generator, we'd return a tuple here.
        processed_lines.append(chomped_line)
    print('read %d lines from data file' % len(processed_lines))
    return processed_lines

# Just run regexp on the line.
# (num, bets, p1_hand, p2_hand, board, p1_result, p2_results, p1_name, p2_name)
def chomp_hand_history_line(line):
    hand_result = acpc_line_regex.match(line.strip()) #match(line.strip())
    if not hand_result:
        print('failure ACPC regexp match: %s' % line.strip())
        return None
    chomped_line = hand_result.groups()
    return chomped_line

# Play a bunch of hands.
# For now... just rush toward full games, and skip details, or fill in with hacks.
def play(sample_size, output_file_name=None, draw_model_filename=None, holdem_model_filename=None,
         bets_model_filename=None, old_bets_model_filename=None, other_old_bets_model_filename=None, 
         human_player=None, compare_models=None, hand_history_filename=None):
    # If we're given a path to ACPC history file, open a connection, and we'll process lines one at a time.
    history_line_reader = None
    if hand_history_filename:
        # Will throw error if file path is unreadable...
        history_line_reader = open(hand_history_filename, 'rU')

    # If we get lines back... create two players, name them, and use randomness going forward.
    if history_line_reader:
        # Just setup dummy players. 
        player_one = TripleDrawAIPlayer()
        player_two = TripleDrawAIPlayer()
        player_one.opponent = player_two
        player_two.opponent = player_one

        # TODO: What other minimum do they need, if we are not loading a model? 

    else:
        # If we load NN-models, do so in this (overly complicated) function
        (player_one, player_two) = generate_player_models(draw_model_filename=draw_model_filename, 
                                                          holdem_model_filename=holdem_model_filename,
                                                          bets_model_filename=bets_model_filename, 
                                                          old_bets_model_filename=old_bets_model_filename, 
                                                          other_old_bets_model_filename=other_old_bets_model_filename, 
                                                          human_player=human_player, 
                                                          compare_models=compare_models)

    # Compute hand values, or compare hands.
    if FORMAT == 'holdem' or FORMAT == 'nlh':
        cashier = HoldemCashier() # Compares by Hold'em (poker high hand) rules
    else:
        cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules

    # TODO: Initialize CSV writer
    csv_header_map = CreateMapFromCSVKey(TRIPLE_DRAW_EVENT_HEADER)
    csv_writer=None
    bufsize = 0 # Write immediately to CSV file. Why? Don't want to lose last hand, etc. Writing to CSV is not dominant operation.
    if output_file_name:
        output_file = open(output_file_name, 'a', bufsize) # append to file... 
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(TRIPLE_DRAW_EVENT_HEADER)

    # Run a bunch of individual hands.
    # Hack: Player one is always on the button...
    round = 1
    # track results... by player, and by small blind/big blind.
    player_one_results = []
    player_two_results = []
    sb_results = []
    bb_results = []
    line = None # Only used if loading lines (randomness and moves) from hand histories.
    try:
        now = time.time()
        if history_line_reader:
            line = history_line_reader.readline().strip()
        while (not history_line_reader and round < sample_size) or line:
            if line:
                print('\nloading hand from ACPC line: |%s|' % line)
                # Ugly hack, in case line fails to parse. Just keep going.
                chomp_result = chomp_hand_history_line(line)
                if chomp_result:
                    (history_num, bets_string, p1_hand_string, p2_hand_string, board_string, p1_result, p2_results, p1_name, p2_name) = chomp_result
                else:
                    line = history_line_reader.readline().strip()
                    continue
            else:
                # Pass empty declarations, if bets & randomness not set from hand history
                bets_string = None
                p1_hand_string = None
                p2_hand_string = None
                board_string = None

            # ACPC log can also get weird, to determine players.
            # We can assume that two players distinct are present, but can't be sure if they trade buttons 100% accurately.
            # 3 cases: Tags not set, player_one == p1, player_two == p2
            if not line:
                # If no ACPC line... alternate button by rounds
                if round % 2:
                    player_one_is_button = True
                else:
                    player_one_is_button = False
            elif line and (not(player_one.tag) or not(player_two.tag)):
                print('New match. Assigning player_one and player_two randomly')
                player_one_is_button = True
                player_one.tag = p2_name
                player_two.tag = p1_name
                print('Player one starts with the button: %s' % player_one.tag)
            elif line and player_one.tag and player_one.tag == p2_name and player_two.tag and player_two.tag == p1_name:
                print('Player_one is button: %s' % player_one.tag)
                player_one_is_button = True
            elif line and player_one.tag and player_one.tag == p1_name and player_two.tag and player_two.tag == p2_name:
                print('Player_two is button: %s' % player_two.tag)
                player_one_is_button = False
            elif line and player_one.tag:
                assert player_one.tag == p1_name or player_one.tag == p2_name, 'Unknown tag |%s| for reading ACPC logs.' % player_one.tag
            else:
                assert False, 'Unknown situation, with players, name and buttons.'

            # Switches button, every other hand. Keeps same model, or definition if a human player.
            if player_one_is_button:
                (bb_result, sb_result) = game_round(round, cashier, player_button=player_one, player_blind=player_two, 
                                                    button_hand_string = p2_hand_string, blind_hand_string = p1_hand_string,
                                                    board_string = board_string, bets_string = bets_string,
                                                    csv_writer=csv_writer, csv_header_map=csv_header_map,
                                                    player_button_average = np.mean(player_one_results),
                                                    player_blind_average = np.mean(player_two_results))
                player_one_result = sb_result
                player_two_result = bb_result
            else:
                (bb_result, sb_result) = game_round(round, cashier, player_button=player_two, player_blind=player_one, 
                                                    button_hand_string = p2_hand_string, blind_hand_string = p1_hand_string,
                                                    board_string = board_string, bets_string = bets_string,
                                                    csv_writer=csv_writer, csv_header_map=csv_header_map,
                                                    player_button_average = np.mean(player_two_results),
                                                    player_blind_average = np.mean(player_one_results))
                player_two_result = sb_result
                player_one_result = bb_result

            player_one_results.append(player_one_result)
            player_two_results.append(player_two_result)
            sb_results.append(sb_result)
            bb_results.append(bb_result)

            # If we loaded ACPC line, check for correctness
            if line:
                print('ACPC line: |%s|' % line)
            print ('hand %d took %.1f seconds...\n' % (round, time.time() - now))

            print('BB results mean %.2f stdev %.2f: %s (%s)' % (np.mean(bb_results), np.std(bb_results), bb_results[-10:], len(bb_results)))
            print('SB results mean %.2f stdev %.2f: %s (%s)' % (np.mean(sb_results), np.std(sb_results), sb_results[-10:], len(sb_results)))
            print('p1 results (%s) mean %.2f stdev %.2f: %s (%s)' % (player_one.player_tag(), 
                                                     np.mean(player_one_results), np.std(player_one_results),
                                                     player_one_results[-10:], len(player_one_results)))
            print('p2 results (%s) mean %.2f stdev %.2f: %s (%s)' % (player_two.player_tag(), 
                                                     np.mean(player_two_results), np.std(player_two_results),
                                                     player_two_results[-10:], len(player_two_results)))

            # Update counter, and read next line if external randomness
            round += 1
            if history_line_reader:
                line = history_line_reader.readline().strip()

            #sys.exit(-3)

    except KeyboardInterrupt:
        pass

    print('completed %d rounds of heads up play' % round)
    sys.stdout.flush()

if __name__ == '__main__':
    samples = 20000 # number of hands to run
    output_file_name = 'triple_draw_events_%d.csv' % samples

    # Input model filename if given
    # TODO: set via command line flagz
    draw_model_filename = None # how to draw, given cards, numDraws (also outputs hand value estimate)
    holdem_model_filename = None # value of holdem hands... if we are playing holdem!
    bets_model_filename = None # what is the value of bet, raise, check, call, fold in this instance?
    old_bets_model_filename = None # use "old" model if we want to compare CNN vs CNN
    other_old_bets_model_filename = None # a third "old" model
    human_player = False # do we want one player to be human?
    compare_models = False # do we want an A/B test instead of latest model vs trailing average?

    # Now fill in these values from command line parameters...
    # TODO: better organize, what game we are playing, and whether against human, etc. 
    draw_model_filename = args.draw_model
    holdem_model_filename = args.holdem_model
    bets_model_filename = args.CNN_model
    old_bets_model_filename = args.CNN_old_model
    other_old_bets_model_filename = args.CNN_other_old_model
    if args.output:
        output_file_name = args.output
    if args.human_player:
        human_player = True
    if args.compare_models:
        compare_models = True

    # Alternatively, load ACPC histories (NLH only) 
    hand_history_filename = args.hand_history

    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name,
         draw_model_filename=draw_model_filename, 
         holdem_model_filename = holdem_model_filename,
         bets_model_filename=bets_model_filename, 
         old_bets_model_filename=old_bets_model_filename, 
         other_old_bets_model_filename=other_old_bets_model_filename, 
         human_player=human_player,
         compare_models=compare_models,
         hand_history_filename=hand_history_filename)
