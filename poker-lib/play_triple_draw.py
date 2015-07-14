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
import scipy.stats as ss
import lasagne
import theano
import theano.tensor as T

from poker_lib import *
from poker_util import *
from draw_poker_lib import * 

from draw_poker import cards_input_from_string
from draw_poker import hand_input_from_context
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector
from triple_draw_poker_full_output import evaluate_single_event # just give it the 26x17x17 bits... and get a vector back
# from triple_draw_poker_full_output import evaluate_batch_hands # much faster to evaluate a batch of hands

print('parsing command line args %s' % sys.argv)
parser = argparse.ArgumentParser(description='Play heads-up triple draw against a convolutional network. Or see two networks battle it out.')
parser.add_argument('-draw_model', '--draw_model', required=True, help='neural net model for draws, or simulate betting if no bet model') # draws, from 32-length array
parser.add_argument('-CNN_model', '--CNN_model', default=None, help='neural net model for betting') # Optional CNN model. If not supplied, uses draw model to "sim" decent play
parser.add_argument('-output', '--output', help='output CSV') # CSV output file, in append mode.
parser.add_argument('--human_player', action='store_true', help='pass for p2 = human player') # Declare if we want a human competitor? (as player_2)
parser.add_argument('-CNN_old_model', '--CNN_old_model', default=None, help='pass for p2 = old model (or second model)') # useful, if we want to test two CNN models against each other.
parser.add_argument('-CNN_other_old_model', '--CNN_other_old_model', default=None, help='pass for p2 = other old model (or 3rd model)') # and a third model, 
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

# Build up a CSV, of all information we might want for CNN training
TRIPLE_DRAW_EVENT_HEADER = ['hand', 'draws_left', 'best_draw', 'hand_after',
                            'bet_model', 'value_heuristic', 'position',  'num_cards_kept', 'num_opponent_kept',
                            'action', 'pot_size', 'bet_size', 'pot_odds', 'bet_this_hand',
                            'actions_this_round', 'actions_full_hand', 
                            'total_bet', 'result', 'margin_bet', 'margin_result',
                            'current_margin_result', 'future_margin_result',
                            'oppn_hand', 'current_hand_win']

BATCH_SIZE = 100 # Across all cases

RE_CHOOSE_FOLD_DELTA = 0.50 # If "random action" chooses a FOLD... re-consider %% of the time.
# Tweak values by up to X (100*chips). Seems like a lot... but not really. 
# NOTE: Does *not* apply to folds. Don't tweak thos.
# NOTE: Lets us break out of a rut of similar actions, etc.
PREDICTION_VALUE_NOISE_HIGH = 0.06
PREDICTION_VALUE_NOISE_LOW = -0.03 # Do decrease it sometimes... so that we don't massively inflate value of actions
PREDICTION_VALUE_NOISE_AVERAGE = (PREDICTION_VALUE_NOISE_HIGH + PREDICTION_VALUE_NOISE_LOW)/2.0 

# Alternatively, use a more sophisticated "tail distribution" from Gumbel
# http://docs.scipy.org/doc/numpy/reference/generated/numpy.random.gumbel.html
# mean = mu + 0.58821 * beta (centered around mu). So match above
PREDICTION_VALUE_NOISE_BETA = 0.04 # 0.06 # vast majority of change within +- 0.05 value, but can stray quite a bit further. Helps make random-ish moves
PREDICTION_VALUE_NOISE_MU = PREDICTION_VALUE_NOISE_AVERAGE - 0.58821 * PREDICTION_VALUE_NOISE_BETA

# Don't boost aggressive actions so much... we want to see more calls, check, especially checks, attempted in spots that might be close.
AGGRESSIVE_ACTION_NOISE_FACTOR = 1.0 # 0.5
MULTIPLE_MODELS_NOISE_FACTOR = 0.3 # Reduce noise... by a lot... if using multiple models already (noise that way)

# Enable, to use 0-5 num_draw model. Recommends when to snow, and when to break, depending on context.
USE_NUM_DRAW_MODEL = True
NUM_DRAW_MODEL_RATE = 0.7 # how often do we use num_draw model? Just use context-free 0-32 output much/most of the time...
NUM_DRAW_MODEL_NOISE_FACTOR = 0.2 # Add noise to predictions... but just a little. 
FAVOR_DEFAULT_NUM_DRAW_MODEL = True # Enable, to boost # of draw cards preferred by 0-32 model. Else, too noisy... but strong preference for other # of cards still matters.

INCLUDE_HAND_CONTEXT = True # False 17 or so extra "bits" of context. Could be set, could be zero'ed out.
SHOW_HUMAN_DEBUG = True # Show debug, based on human player...
SHOW_MACHINE_DEBUG_AGAINST_HUMAN = False # True, to see machine logic when playing (will reveal hand)
USE_MIXED_MODEL_WHEN_AVAILABLE = True # When possible, including against human, use 2 or 3 models, and choose randomly which one decides actions.
RETRY_FOLD_ACTION = True # If we get a model that says fold preflop... try again. But just once. We should avoid raise/fold pre

# From experiments & guesses... what contitutes an 'average hand' (for our opponent), at this point?
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
    if bets_this_round >= 1:
        baseline += 0.05 * (bets_this_round) 
        baseline += 0.05 * (bets_this_round - 1)

    return baseline

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

# Should inherit from more general player... when we need one. (For example, manual player who chooses his own moves and own draws)
class TripleDrawAIPlayer():
    # TODO: Initialize model to use, etc.
    def __init__(self):
        self.draw_hand = None

        # TODO: Name, and track, multiple models. 
        # This is the draw model. Also, outputs the heuristic (value) of a hand, given # of draws left.
        self.output_layer = None # for draws
        self.bets_output_layer = None 
        self.use_learning_action_model = False # should we use the trained model, to make betting decisions?
        self.old_bets_output_model = False # "old" model used for NIPS... should be set from the outside
        self.other_old_bets_output_model = False # the "other" old bets model [CNN3 vs CNN5, etc]
        self.bets_output_array = [] # if we use *multiple* models, and choose randomly between them
        self.is_human = False

        # Current 0-1000 value, based on cards held, and approximation of value from draw model.
        # For example, if no more draws... heuristic is actual hand.
        self.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 
        self.num_cards_kept = 0 # how many cards did we keep... with out last draw?
        self.cards = [] # Display purposes only... easy way to see current hand as last evaluated

        # TODO: Use this to track number of cards discarded, etc. Obviously, don't look at opponent's cards.
        self.opponent = None

    def player_tag(self):
        if self.is_human:
            return 'man'
        elif self.use_learning_action_model and self.bets_output_layer:
            # Backward-compatibility hack, to allow support for "old" model used for NIPS paper (CNN)
            # NOTE: Will be deprecated...
            if self.bets_output_array and len(self.bets_output_array) > 0:
                return 'CNN_45' # we sample from multiple models!
            if self.other_old_bets_output_model:
                return 'CNN_4'
            elif self.old_bets_output_model:
                return 'CNN_5'
            else:
                return 'CNN_6'
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
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = num_draws) #, test_batch=self.test_batch)
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
        if draw_recommendations and USE_NUM_DRAW_MODEL and random.random() <= NUM_DRAW_MODEL_RATE:
            # The point isn't to over-rule 0-32 model in close cases. The point is to look for *clear advantages* to 
            # snowing a hand, or breaking a hand. Therefore, add a bonus to the move already preferred by 0-32 model.
            if FAVOR_DEFAULT_NUM_DRAW_MODEL:
                default_num_kept = best_draw_num_kept # 0-5
                for prediction in draw_recommendations:
                    action = prediction[1]
                    # Boost the value by fixed amount... and also noise (but only on the upside)
                    # This will reduce randomly choosing an inferior draw. But not every time. And *much better* draw action wins easily.
                    if drawCategoryNumCardsKept[action] == default_num_kept:
                        noise = (max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA)) + max(0.0, np.random.gumbel(PREDICTION_VALUE_NOISE_MU, PREDICTION_VALUE_NOISE_BETA))) / 2.0
                        noise += PREDICTION_VALUE_NOISE_AVERAGE * 5.0
                        prediction[0] += noise
                        if debug:
                            print('\tBoosted %d-card draw by %.3f' % (5-default_num_kept, noise))
                # Re-sort results, of course.
                draw_recommendations.sort(reverse=True)

            action = draw_recommendations[0][1]
            cards_kept = drawCategoryNumCardsKept[action]
            best_draw = best_draws_by_num_kept[cards_kept]
            if debug:
                print('\tchosen to use num_draw model with %.1f%%' % (NUM_DRAW_MODEL_RATE * 100.0))
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

    # Apply the CNN... to get "value" of the current hand. best draw for hands with draws left; current hand for no more draws.
    # NOTE: Similar to draw_move() but we don't make any actual draws with the hand.
    def update_hand_value(self, num_draws=0, debug = True):
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
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = max(num_draws, 1)) #, test_batch=self.test_batch)

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
             has_button = True, pot_size=0, actions_this_round=[], cards_kept=0, opponent_cards_kept=0, 
             debug = True, retry = False):
        # Reduce debug, if opponent is human, and could see.
        if (self.opponent and self.opponent.is_human and (not SHOW_MACHINE_DEBUG_AGAINST_HUMAN)) or self.is_human:
            debug = False

        # If we have context and bets model that also outputs draws... at least give it a look.
        bets_layer = self.bets_output_layer # use latest "bets" layer, even if multiple available.
        value_predictions = None
        if bets_layer and self.use_learning_action_model:
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
            
            # Now hand context
            if debug:
                print('context %s' % ([hand_string_dealt, num_draws_left, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept]))
            hand_context_input = hand_input_from_context(position=has_button, pot_size=pot_size, bets_string=bets_string,
                                                         cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept)
            full_input = np.concatenate((cards_input, hand_context_input), axis = 0)
            # TODO: Rewrite with input and output layer... so that we can avoid putting all this data into Theano.shared()
            bets_vector = evaluate_single_event(bets_layer, full_input)

            # Show the values for all draws [0, 5] cards kept.
            if debug:
                print('vals\t%s' % ([val - 2.0 for val in bets_vector[:5]]))
                print('acts\t%s' % ([val for val in bets_vector[5:10]]))
                print('drws\t%s' % ([val - 2.0 for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]))
            #value_predictions = [[(bets_vector[category_from_event_action(action)] - 2.0), action, '%s: %.3f' % (actionName[action], bets_vector[category_from_event_action(action)] - 2.0)] for action in actions]
            value_predictions = [[(bets_vector[action] - 2.0), action, '%s: %.3f' % (drawCategoryName[action], bets_vector[action] - 2.0)] for action in DRAW_CATEGORY_SET]
            value_predictions.sort(reverse=True)
            
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
    def choose_action(self, actions, round, bets_this_round = 0, 
                      has_button = True, pot_size=0, actions_this_round=[], cards_kept=0, opponent_cards_kept=0, 
                      debug = True, retry = False):
        # Reduce debug, if opponent is human, and could see.
        if self.opponent and self.opponent.is_human and (not SHOW_MACHINE_DEBUG_AGAINST_HUMAN):
            debug = False

        # print('Choosing among actions %s for round %s' % (actions, round))
        # self.choose_random_action(actions, round)

        # Either use bets model... or if given several... choose one at random to apply
        # NOTE: We do not want to average the *values* in models, but the actual choices. Thus we'll be stochastic & unpredicatable.
        bets_layer = None
        if self.bets_output_array:
            bets_layer = random.choice(self.bets_output_array)
            if debug:
                print('chose bets model %d in %d-length models index...' % (self.bets_output_array.index(bets_layer), len(self.bets_output_array)))
        else:
            bets_layer = self.bets_output_layer

        if bets_layer and self.use_learning_action_model:
            #print('We has a *bets* output model. Use it!')
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
            if debug:
                print('context %s' % ([hand_string_dealt, num_draws_left, has_button, pot_size, bets_string, cards_kept, opponent_cards_kept]))
            hand_context_input = hand_input_from_context(position=has_button, pot_size=pot_size, bets_string=bets_string,
                                                         cards_kept=cards_kept, opponent_cards_kept=opponent_cards_kept)
            full_input = np.concatenate((cards_input, hand_context_input), axis = 0)

            # TODO: Rewrite with input and output layer... so that we can avoid putting all this data into Theano.shared()
            bets_vector = evaluate_single_event(bets_layer, full_input)

            """
            # For special (and slow) debug... show all models, if available...
            if debug and self.bets_output_array:
                for model_layer in self.bets_output_array:
                    possible_bets_vector = evaluate_single_event(model_layer, full_input)
                    print('vals\t%s' % ([val - 2.0 for val in possible_bets_vector[:5]]))
                    """

            # Show all the raw returns from training. [0:5] -> values of actions [5:10] -> probabilities recommended
            # NOTE: Actions are exploratory, and wrong. But see how it goes...
            if debug:
                print('vals\t%s' % ([val - 2.0 for val in bets_vector[:5]]))
                print('acts\t%s' % ([val for val in bets_vector[5:10]]))
                print('drws\t%s' % ([val - 2.0 for val in bets_vector[KEEP_0_CARDS:(KEEP_5_CARDS+1)]]))
            value_predictions = [[(bets_vector[category_from_event_action(action)] - 2.0), action, '%s: %.3f' % (actionName[action], bets_vector[category_from_event_action(action)] - 2.0)] for action in actions]
            value_predictions.sort(reverse=True)
            
            if debug:
                print(value_predictions)

            best_action_no_noise = value_predictions[0][1]

            # Now apply noise to action values, if requested.
            # Why? If actions are close, don't be vulnerable to small differences that lock us into action.
            # NOTE: We do *not* want more folds, so only increase values of non-fold actions. Clear folds still fold.
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

                    # And boost bet/raise somewhat less than the other actions.
                    if action in ALL_BETS_SET:
                        noise *= AGGRESSIVE_ACTION_NOISE_FACTOR

                    # NOTE: Do *not* apply noise, if already using a mixed model. That is noise enough.
                    if self.bets_output_array:
                        noise *= MULTIPLE_MODELS_NOISE_FACTOR

                    prediction[0] += noise
                value_predictions.sort(reverse=True)
                if debug:
                    print(value_predictions)
            
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
                                          debug = debug, retry = True)

            # Purely for debug
            if debug:
                self.create_heuristic_action_distribution(round, bets_this_round = bets_this_round, has_button = has_button)
            if (best_action_no_noise != best_action) and debug:
                print('--> changed best action %s -> %s from tweaking!' % (actionName[best_action_no_noise], actionName[best_action]))

            #print(best_action)
            print('\n%s\n' % actionName[best_action])
            
            # Internal variable, for easy switch between learning model, and heuristic model below.
            if self.use_learning_action_model:
                return best_action
        else:
            print('No *bets* output model specified (or not used) for player %s' % self.name)
        return self.choose_heuristic_action(allowed_actions = list(actions), 
                                            round = round, 
                                            bets_this_round = bets_this_round, 
                                            has_button = has_button)

    # Use known game information, and especially hand heuristic... to output probability preference for actions.
    # (bet_raise, check_call, fold)
    # TODO: Pass along other important hand aspects here... # of bets made, hand history, opponent draw #, etc
    def create_heuristic_action_distribution(self, round, bets_this_round = 0, has_button = True):
        # Baseline is 2/2/0.5 bet/check/fold
        bet_raise = 2.0
        check_call = 2.0
        fold = 0.5 # 1.0

        # If player is on the button, he should check less, and bet more. 
        # TODO: This change a lot, once we look at previous draw & bet patterns.
        # NOTE: What we want to avoid is player constantly betting into a much stronger hand.
        if has_button:
            bet_raise = 2.0
            check_call = 1.0
            fold = 0.5

        # See our value, and typical opponent hand value... adjust our betting pattern.
        hand_value = self.heuristic_value
        baseline_value = baseline_heuristic_value(round, bets_this_round)

        print('Player %s (has_button %d) our hand value %.2f, compared to current baseline %.2f %s' % (self.name, has_button, hand_value, baseline_value, hand_string(self.cards)))
        
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
            fold_increase = 0.5 / 0.10 * (hand_value - baseline_value)
            #print('increasing fold by %.2f' % fold_increase)
            fold -= fold_increase      

        # Decrease folding as the pot grows... shift these to calls.
        # NOTE: Also balances us out, in terms of not raising and folding too much. We won't over-fold to 4bet, etc
        if bets_this_round > 1:
            fold_decrease = 0.5 / 2 * (bets_this_round - 1) 
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

        return (max(bet_raise + raise_minimum, raise_minimum), check_call, max(fold, 0.0))
        

    # Computes a distribution over actions, based on (hand_value, round, other info)
    # Then, probabilistically chooses a single action, from the distribution.
    # NOTE: allowed_actions needs to be a list... so that we can match probabilities for each.
    def choose_heuristic_action(self, allowed_actions, round, bets_this_round = 0, has_button = True):
        #print('Allowed actions %s' % ([actionName[action] for action in allowed_actions]))

        # First, create a distribution over actions.
        # NOTE: Resulting distribution is *not* normalized. Could return (3, 2, 0.5)
        (bet_raise, check_call, fold) = self.create_heuristic_action_distribution(round, 
                                                                                  bets_this_round = bets_this_round,
                                                                                  has_button = has_button)

        # Normalize so sum adds to 1.0
        action_sum = bet_raise + check_call + fold
        assert action_sum > 0.0, 'actions sum to impossible number %s' % [bet_raise, check_call, fold]

        bet_raise /= action_sum
        check_call /= action_sum
        fold /= action_sum

        # Match outputs above to actual game actions. Assign values directly to action.probability
        print('(bet/raise %.2f, check/call %.2f, fold %.2f)' % (bet_raise, check_call, fold))
        
        # Good for easy lookup of "are we allowed to bet here"?
        all_actions_set = set(allowed_actions)

        action_probs = []
        for action in allowed_actions:
            probability = 0.0
            if action == CALL_SMALL_STREET or  action == CALL_BIG_STREET:
                #print('CALL take all of the check/call credit: %s' % check_call)
                probability += check_call
                
                # if we are not allowed to bet or raise... take that credit also. [betting capped, etc]
                if not(set(ALL_BETS_SET) & all_actions_set):
                    #print('since no BET/RAISE, CALL takes all bet/raise credit: %s' % bet_raise)
                    probability += bet_raise
            elif action == BET_SMALL_STREET or action == BET_BIG_STREET:
                #print('BET take all of the bet/raise credit: %s' % bet_raise)
                probability += bet_raise
            elif action == RAISE_SMALL_STREET or action == RAISE_BIG_STREET:
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
            else:
                assert False, 'Unknown possible action %s' % actionName[action]
                
            action_probs.append(probability)
                
        # Probabilities should add up to 1.0...
        action_distribution = action_probs

        # Then sample a single action, from this distribution.
        choice_action = np.random.choice(len(allowed_actions), 1, p = action_distribution)
        #print('choice: %s' % allowed_actions[choice_action[0]])
        return allowed_actions[choice_action[0]]

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

    def choose_action(self, actions, round, bets_this_round = 0, 
                      has_button = True, pot_size=0, actions_this_round=[], cards_kept=0, opponent_cards_kept=0):
        print('Choosing among actions %s for round %s' % ([actionName[action] for action in actions], round))
        if SHOW_HUMAN_DEBUG:
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

        # Prompt user for action, and see if it parses...
        # TODO: Separate function, or library method...
        user_action = None
        while not user_action:
            user_move_string = raw_input("Please select action %s -->   " % [actionName[action] for action in actions])
            print('User action: |%s|' % user_move_string)
            if len(user_move_string) > 0:
                user_char = user_move_string.lower()[0]
                if user_char == 'c':
                    print('action check/call')
                    allowed_actions = set(list(ALL_CALLS_SET) + [CHECK_HAND])
                elif user_char == 'b' or user_char == 'r':
                    print('action bet/raise')
                    allowed_actions = ALL_BETS_SET
                elif user_char == 'f':
                    print('action FOLD')
                    allowed_actions = set([FOLD_HAND])
                else:
                    print('unparseable action... try again')
                    continue

                user_actions = list(set.intersection(set(actions), allowed_actions))
                if (not user_actions) or len(user_actions) != 1:
                    print('unparseable action... try again')
                    continue
                user_action = user_actions[0]

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

# As simply as possible, simulate a full round of triple draw. Have as much logic as possible, contained in the objects
# Actors:
# cashier -- evaluates final hands
# deck -- dumb deck, shuffled once, asked for next cards
# dealer -- runs the game. Tracks pot, propts players for actions. Decides when hand ends.
# players -- acts directly on a poker hand. Makes draw and betting decisions... when propted by the dealer
def game_round(round, cashier, player_button=None, player_blind=None, csv_writer=None, csv_header_map=None):
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

    deck = PokerDeck(shuffle=True)

    dealer = TripleDrawDealer(deck=deck, player_button=player_button, player_blind=player_blind)
    dealer.play_single_hand()

    # TODO: Should output results.
    # TODO: Also output game history for training data

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
    for event in dealer.hand_history:
        # Back-update hand result, and decision result for all moves made
        event.update_result(winners, final_bets)
        print(event)
        if csv_header_map:
            event_line = event.csv_output(csv_header_map)
            print(event_line)

        # Write events, for training.
        # TODO: Include draw events.
        if csv_writer:
            csv_writer.writerow(event_line)
    # TODO: Flush buffer here?

    # If we are tracking results... return results (wins/losses for player by order
    bb_result = dealer.hand_history[0].margin_result
    sb_result = dealer.hand_history[1].margin_result
    return (bb_result, sb_result)

# Play a bunch of hands.
# For now... just rush toward full games, and skip details, or fill in with hacks.
def play(sample_size, output_file_name=None, draw_model_filename=None, bets_model_filename=None, 
         old_bets_model_filename=None, other_old_bets_model_filename=None, human_player=None):
    # Compute hand values, or compare hands.
    cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules

    # TODO: Initialize CSV writer
    csv_header_map = CreateMapFromCSVKey(TRIPLE_DRAW_EVENT_HEADER)
    csv_writer=None
    if output_file_name:
        output_file = open(output_file_name, 'a') # append to file... 
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(TRIPLE_DRAW_EVENT_HEADER)


    # Test the model, by giving it dummy inputs
    # Test cases -- it keeps the two aces. But can it recognize a straight? A flush? Trips? Draw?? Two pair??
    test_cases = [['As,Ad,4d,3s,2c', 1], ['As,Ks,Qs,Js,Ts', 2], ['3h,3s,3d,5c,6d', 3],
                  ['3h,4s,3d,5c,6d', 2], ['2h,3s,4d,6c,5s', 1], ['3s,2h,4d,8c,5s', 3],
                  ['8s,Ad,Kd,8c,Jd', 3], ['8s,Ad,2d,7c,Jd', 2], ['2d,7d,8d,9d,4d', 1]] 

    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[1])

    # NOTE: Num_draws and full_hand must match trained model.
    # TODO: Use shared environemnt variables...
    test_batch = np.array([cards_input_from_string(hand_string=case[0], 
                                                   include_num_draws=True, num_draws=case[1],
                                                   include_full_hand = True, 
                                                   include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], np.int32)

    # If model file provided, unpack model, and create intelligent agent.
    output_layer = None
    if draw_model_filename and os.path.isfile(draw_model_filename):
        print('\nExisting model in file %s. Attempt to load it!\n' % draw_model_filename)
        all_param_values_from_file = np.load(draw_model_filename)
        
        # Size must match exactly!
        output_layer = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )
        print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file)
        predict_model(output_layer=output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled model...')
    else:
        print('No model provided or loaded. Expect error if model required. %s', draw_model_filename)

    # If supplied, also load the bets model. conv(xCards + xNumDraws + xContext) --> values for all betting actions
    bets_output_layer = None
    if bets_model_filename and os.path.isfile(bets_model_filename):
        print('\nExisting *bets* model in file %s. Attempt to load it!\n' % bets_model_filename)
        bets_all_param_values_from_file = np.load(bets_model_filename)

        # Size must match exactly!
        bets_output_layer = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )
        print('filling model with shape %s, with %d params' % (str(bets_output_layer.get_output_shape()), len(bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(bets_output_layer, bets_all_param_values_from_file)
        predict_model(output_layer=bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *bets* model provided or loaded. Expect error if model required. %s', bets_model_filename)

    # If supplied "old" bets layer supplied... then load it as well.
    old_bets_output_layer = None
    if old_bets_model_filename and os.path.isfile(old_bets_model_filename):
        print('\nExisting *old bets* model in file %s. Attempt to load it!\n' % old_bets_model_filename)
        old_bets_all_param_values_from_file = np.load(old_bets_model_filename)

        # Size must match exactly!
        old_bets_output_layer = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )
        print('filling model with shape %s, with %d params' % (str(old_bets_output_layer.get_output_shape()), len(old_bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(old_bets_output_layer, old_bets_all_param_values_from_file)
        predict_model(output_layer=old_bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *old bets* model provided or loaded. Expect error if model required. %s', old_bets_model_filename)

    # Lastly, load a third model...
    other_old_bets_output_layer = None
    if other_old_bets_model_filename and os.path.isfile(other_old_bets_model_filename):
        print('\nExisting *old bets* model in file %s. Attempt to load it!\n' % other_old_bets_model_filename)
        other_old_bets_all_param_values_from_file = np.load(other_old_bets_model_filename)

        # Size must match exactly!
        other_old_bets_output_layer = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )
        print('filling model with shape %s, with %d params' % (str(other_old_bets_output_layer.get_output_shape()), len(other_old_bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(other_old_bets_output_layer, other_old_bets_all_param_values_from_file)
        predict_model(output_layer=other_old_bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *old bets* model provided or loaded. Expect error if model required. %s', other_old_bets_model_filename)

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
    
    # Add model, to players.

    # Player 1 plays the latest model... or a mixed bag of models, if provided.
    player_one.output_layer = output_layer
    player_one.bets_output_layer = bets_output_layer
    # enable, to make betting decisions with learned model (instead of heurstics)
    player_one.use_learning_action_model = True
    if USE_MIXED_MODEL_WHEN_AVAILABLE and (old_bets_output_layer or other_old_bets_output_layer):
        player_one.bets_output_array = []
        player_one.bets_output_array.append(bets_output_layer) # lastest model
        if old_bets_output_layer:
            player_one.bets_output_array.append(old_bets_output_layer)
        if other_old_bets_output_layer:
            player_one.bets_output_array.append(other_old_bets_output_layer)
        print('loaded player_one with %d-mixed model!' % len(player_one.bets_output_array))

    # Player 2 plays the least late model... unless player 1 is playing a mixed bag.
    # NOTE: This sounds confusing, but is not. We need to test vs human (with mixed model, if given)
    # Otherwise, we want to test the latest model, against the mix. Or the latest model, against the oldest given.
    player_two.output_layer = output_layer
    player_two.bets_output_layer = bets_output_layer
    # enable, to make betting decisions with learned model (instead of heurstics)
    player_two.use_learning_action_model = True

    # If we want to supply "old" model, for new CNN vs old CNN. 
    if (not human_player) and old_bets_output_layer and (not player_one.bets_output_array):
        player_two.bets_output_layer = old_bets_output_layer
        player_two.old_bets_output_model = True

    # Use the "other" "old" model, if provided
    if (not human_player) and other_old_bets_output_layer and (not player_one.bets_output_array):
        player_two.bets_output_layer = other_old_bets_output_layer
        player_two.other_old_bets_output_model = True

    # Run a bunch of individual hands.
    # Hack: Player one is always on the button...
    round = 1
    # track results... by player, and by small blind/big blind.
    player_one_results = []
    player_two_results = []
    sb_results = []
    bb_results = []
    try:
        now = time.time()
        while round < sample_size:
            # TODO: Implement human player.
            # Switches button, every other hand. Relevant, if one of the players uses a different moves model.
            if round % 2:
                (bb_result, sb_result) = game_round(round, cashier, player_button=player_one, player_blind=player_two, 
                                                     csv_writer=csv_writer, csv_header_map=csv_header_map)
                player_one_result = sb_result
                player_two_result = bb_result
            else:
                (bb_result, sb_result) = game_round(round, cashier, player_button=player_two, player_blind=player_one, 
                                                     csv_writer=csv_writer, csv_header_map=csv_header_map)
                player_two_result = sb_result
                player_one_result = bb_result

            player_one_results.append(player_one_result)
            player_two_results.append(player_two_result)
            sb_results.append(sb_result)
            bb_results.append(bb_result)

            print ('hand %d took %.1f seconds...\n' % (round, time.time() - now))

            print('BB results mean %.2f stdev %.2f: %s (%s)' % (np.mean(bb_results), np.std(bb_results), bb_results[-10:], len(bb_results)))
            print('SB results mean %.2f stdev %.2f: %s (%s)' % (np.mean(sb_results), np.std(sb_results), sb_results[-10:], len(sb_results)))
            print('p1 results (%s) mean %.2f stdev %.2f: %s (%s)' % (player_one.player_tag(), 
                                                     np.mean(player_one_results), np.std(player_one_results),
                                                     player_one_results[-10:], len(player_one_results)))
            print('p2 results (%s) mean %.2f stdev %.2f: %s (%s)' % (player_two.player_tag(), 
                                                     np.mean(player_two_results), np.std(player_two_results),
                                                     player_two_results[-10:], len(player_two_results)))

            round += 1

            #sys.exit(-3)

    except KeyboardInterrupt:
        pass

    print('completed %d rounds of heads up play' % round)
    sys.stdout.flush()

if __name__ == '__main__':
    samples = 5000 # number of hands to run
    output_file_name = 'triple_draw_events_%d.csv' % samples

    # Input model filename if given
    # TODO: set via command line flagz
    draw_model_filename = None # how to draw, given cards, numDraws (also outputs hand value estimate)
    bets_model_filename = None # what is the value of bet, raise, check, call, fold in this instance?
    old_bets_model_filename = None # use "old" model if we want to compare CNN vs CNN
    other_old_bets_model_filename = None # a third "old" model
    human_player = False # do we want one player to be human?

    # Now fill in these values from command line parameters...
    draw_model_filename = args.draw_model
    bets_model_filename = args.CNN_model
    old_bets_model_filename = args.CNN_old_model
    other_old_bets_model_filename = args.CNN_other_old_model
    if args.output:
        output_file_name = args.output
    if args.human_player:
        human_player = True

    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name,
         draw_model_filename=draw_model_filename, 
         bets_model_filename=bets_model_filename, 
         old_bets_model_filename=old_bets_model_filename, 
         other_old_bets_model_filename=other_old_bets_model_filename, 
         human_player=human_player)
