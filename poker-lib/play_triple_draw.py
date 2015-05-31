import sys
import csv
import logging
import math
import time
import re
import random
import os.path # checking file existence, etc
import numpy as np
import scipy.stats as ss
import lasagne
import theano
import theano.tensor as T

from poker_lib import *
from poker_util import *
from draw_poker_lib import * 

from draw_poker import cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector
# from triple_draw_poker_full_output import evaluate_batch_hands # much faster to evaluate a batch of hands

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
                            'value_heuristic', 'position',  'num_cards_kept', 'num_opponent_kept',
                            'action', 'pot_size', 'bet_size', 'pot_odds', 'bet_this_hand',
                            'actions_this_round', 'actions_full_hand', 
                            'total_bet', 'result', 'margin_bet', 'margin_result']

BATCH_SIZE = 100 # Across all cases

RE_CHOOSE_FOLD_DELTA = 0.50 # If "random action" chooses a FOLD... re-consider %% of the time.

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


# Should inherit from more general player... when we need one. (For example, manual player who chooses his own moves and own draws)
class TripleDrawAIPlayer():
    # TODO: Initialize model to use, etc.
    def __init__(self):
        self.draw_hand = None

        # TODO: Name, and track, multiple models. 
        # This is the draw model. Also, outputs the heuristic (value) of a hand, given # of draws left.
        self.output_layer = None 

        # Current 0-1000 value, based on cards held, and approximation of value from draw model.
        # For example, if no more draws... heuristic is actual hand.
        self.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 
        self.num_cards_kept = 0 # how many cards did we keep... with out last draw?

        # TODO: Use this to track number of cards discarded, etc. Obviously, don't look at opponent's cards.
        #self.opponent_hand = None

    # Takes action on the hand. But first... get Theano output...
    def draw_move(self, deck, num_draws = 1):
        hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
        print('dealt %s for draw %s' % (hand_string_dealt, num_draws))

        # Get 32-length vector for each possible draw, from the model.
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = num_draws) #, test_batch=self.test_batch)

        print('All 32 values: %s' % str(hand_draws_vector))

        best_draw = np.argmax(hand_draws_vector)
        
        print('Best draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        
        draw_string = ''
        for i in range(0,5):
            if not (i in all_draw_patterns[best_draw]):
                draw_string += '%d' % i

        print('Draw string from AI! |%s|' % draw_string)

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
    def update_hand_value(self, num_draws=0):
        # For no more draws... use "final hand." Otherwise we run into issues with showdown, etc
        if (num_draws >= 1):
            hand_string_dealt = hand_string(self.draw_hand.dealt_cards)
        else:
            hand_string_dealt = hand_string(self.draw_hand.final_hand)

        print('dealt %s for draw %s' % (hand_string_dealt, num_draws))

        # Get 32-length vector for each possible draw, from the model.
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt, num_draws = max(num_draws, 1)) #, test_batch=self.test_batch)

        # Except for num_draws == 0, value is value of the best draw...
        if num_draws >= 1:
            print('All 32 values: %s' % str(hand_draws_vector))
            best_draw = np.argmax(hand_draws_vector)
        else:
            print('With no draws left, heurstic value is for pat hand.')
            best_draw = 31
        
        print('Best draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        self.heuristic_value = expected_payout

        return expected_payout

    # Apply current model, based on known information, to draw 0-5 cards from the deck.
    def draw(self, deck, num_draws = 1):
        self.draw_move(deck, num_draws)

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
    def choose_action(self, actions, round, bets_this_round = 0, has_button = True):
        # print('Choosing among actions %s for round %s' % (actions, round))
        # self.choose_random_action(actions, round)
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

        print('Player %s (has_button %d) our hand value %.2f, compared to current baseline %.2f' % (self.name, has_button, hand_value, baseline_value))
        
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
            
            # Start to fold more, especially if we are 0.20 or more behind expect opponenet hand (2-card draw vs pat hand, etc)
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

        if raise_minimum and raise_minimum > bet_raise:
            print('resetting mimimum raise to %.2f. Fortune favors the bold!' % max(bet_raise + raise_minimum, raise_minimum))

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


# As simply as possible, simulate a full round of triple draw. Have as much logic as possible, contained in the objects
# Actors:
# cashier -- evaluates final hands
# deck -- dumb deck, shuffled once, asked for next cards
# dealer -- runs the game. Tracks pot, propts players for actions. Decides when hand ends.
# players -- acts directly on a poker hand. Makes draw and betting decisions... when propted by the dealer
def game_round(round, cashier, player_button=None, player_blind=None, csv_writer=None, csv_header_map=None):
    print '\n-- New Round %d --\n' % round
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


# Play a bunch of hands.
# For now... just rush toward full games, and skip details, or fill in with hacks.
def play(sample_size, output_file_name=None, model_filename=None):
    # Compute hand values, or compare hands.
    cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules

    # TODO: Initialize CSV writer
    csv_header_map = CreateMapFromCSVKey(TRIPLE_DRAW_EVENT_HEADER)
    csv_writer=None
    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(TRIPLE_DRAW_EVENT_HEADER)

    # If model file provided, unpack model, and create intelligent agent.
    output_layer = None
    if model_filename and os.path.isfile(model_filename):
        print('\nExisting model in file %s. Attempt to load it!\n' % model_filename)
        all_param_values_from_file = np.load(model_filename)
        
        # Size must match exactly!
        output_layer = build_model(
            17, # 15, #input_height=dataset['input_height'],
            17, # 15, #input_width=dataset['input_width'],
            32, #output_dim=dataset['output_dim'],
        )

        print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file)

        # TODO: "test cases" should also be a function, like "evaluate_single_hand"

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
                                                       include_full_hand = True) for case in test_cases], np.int32)
        predict_model(output_layer=output_layer, test_batch=test_batch)

        print('Cases again %s' % str(test_cases))

        print('Creating player, based on this pickled model...')
    else:
        print('No model provided or loaded. Expect error if model required. %s', model_filename)

    # We initialize deck, and dealer, every round. But players kept constant, and reset for each trial.
    # NOTE: This can, and will change, if we do repetative simulation, etc.
    player_one = TripleDrawAIPlayer()
    player_two = TripleDrawAIPlayer()
    
    # Add model, to players.
    player_one.output_layer = output_layer
    player_two.output_layer = output_layer

    # Run a bunch of individual hands.
    # Hack: Player one is always on the button...
    round = 0
    try:
        now = time.time()
        while round < sample_size:
            # TODO: Implement human player, switch button.
            game_round(round, cashier, player_button=player_one, player_blind=player_two, 
                       csv_writer=csv_writer, csv_header_map=csv_header_map)

            print ('hand %d took %.1f seconds...' % (round, time.time() - now))

            round += 1

            #sys.exit(-3)

    except KeyboardInterrupt:
        pass

    print('completed %d rounds of heads up play' % round)
    sys.stdout.flush()

if __name__ == '__main__':
    samples = 200 # number of hands to run
    output_file_name = 'triple_draw_events_%d.csv' % samples

    # Input model filename if given
    # TODO: set via command line flagz
    model_filename = None
    if len(sys.argv) >= 2:
        model_filename = sys.argv[1]

    if len(sys.argv) >= 3:
        output_file_name = sys.argv[2]

    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name, model_filename=model_filename)
