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

BATCH_SIZE = 100 # Across all cases

# Heuristics, to evaluate hand actions. On 0-1000 scale, where wheel is 1000 points, and bad hand is 50-100 points.
# Meant to map to rough % of winning at showdown. Tuned for ring game, so random hand << 500.
RANDOM_HAND_HEURISTIC_BASELINE = 0.300 # baseline, before looking at any cards.

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
        baseline = RANDOM_HAND_HEURISTIC_BASELINE + 0.125
    elif round == DRAW_3_BET_ROUND:
        # Not a typo. Value doesn't change with final round... (in this hack, anyway)
        # NOTE: Where value really changes... is facing a bet. Or two. Or three.
        baseline = RANDOM_HAND_HEURISTIC_BASELINE + 0.125
        
    # Increase the baseline... especially as we get into 3-bet and 4-bet territory.
    if bets_this_round >= 1:
        baseline += 0.05 * (bets_this_round) 
        baseline += 0.025 * (bets_this_round - 1)

    return baseline


# Should inherit from more general player... when we nee one.
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

        # TODO: Use this to track number of cards discarded, etc. Obviously, don't look at opponent's cards.
        self.opponent_hand = None

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
        print('Choosing among actions %s for round %s' % (actions, round))
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
            print('increasing bet/raise by %.2f' % bet_increase)
            bet_raise += bet_increase
            
            fold_decrease = 0.5 / 0.10 * (hand_value - baseline_value)
            print('decreasing fold by %.2f' % fold_decrease)
            fold -= fold_decrease
        elif hand_value < baseline_value:
            # Quickly stop raising, if our hand is below expectation
            bet_decrease = 1.0 / 0.10 * (hand_value - baseline_value)
            print('decreasing bet/raise by %.2f' % bet_decrease)
            bet_raise += bet_decrease
            
            # Start to fold more, especially if we are 0.20 or more behind expect opponenet hand (2-card draw vs pat hand, etc)
            fold_increase = 0.5 / 0.10 * (hand_value - baseline_value)
            print('increasing fold by %.2f' % fold_increase)
            fold -= fold_increase      

        # Decrease folding as the pot grows... shift these to calls.
        # NOTE: Also balances us out, in terms of not raising and folding too much. We won't over-fold to 4bet, etc
        if bets_this_round > 1:
            fold_decrease = 0.5 / 2 * (bets_this_round - 1) 
            print('decreaseing fold by %.2f since much action this street.' % fold_decrease)
            fold -= fold_decrease

            print('incrase call by similar amount %.2f' % 2 * fold_decrease)
            check_call += 2 * fold_decrease

        # We should often consider some chance at a really aggressive action, even in bad spots.
        # For example, on busted river...
        raise_minimum = 0.0
        if bets_this_round == 0:
            raise_minimum += 0.25
        if has_button and bets_this_round < 2:
            raise_minimum += 0.25

        if raise_minimum and raise_minimum > bet_raise:
            print('resetting mimimum raise to %.2f. Fortune favors the bold!' % bet_raise)

        return (max(bet_raise, raise_minimum), check_call, max(fold, 0.0))
        

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
                print('CALL take all of the check/call credit: %s' % check_call)
                probability += check_call
                
                # if we are not allowed to bet or raise... take that credit also. [betting capped, etc]
                if not(set(ALL_BETS_SET) & all_actions_set):
                    print('since no BET/RAISE, CALL takes all bet/raise credit: %s' % bet_raise)
                    probability += bet_raise
            elif action == BET_SMALL_STREET or action == BET_BIG_STREET:
                print('BET take all of the bet/raise credit: %s' % bet_raise)
                probability += bet_raise
            elif action == RAISE_SMALL_STREET or action == RAISE_BIG_STREET:
                print('RAISE take all of the bet/raise credit: %s' % bet_raise)
                probability += bet_raise
            elif action == FOLD_HAND:
                print('FOLD take all of the fold credit: %s' % fold)
                probability += fold
            elif action == CHECK_HAND:
                print('CHECK take all of the check/call credit: %s' % check_call)
                probability += check_call

                # If we can't fold... credit goes here.
                if not(FOLD_HAND in all_actions_set):
                    print('Since no FOLD, CHECK takes all fold credit: %s' % fold)
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


# Should inherit from more general dealer class... when we need one.
class TripleDrawDealer():
    def __init__(self, deck, player_button, player_blind):
        self.deck = deck # Assume it's shuffled, if needs to be shuffled
        self.player_button = player_button # player to act last on every street
        self.player_blind = player_blind # player to act first, except when posting forced blind

        # Hack, for convenience.
        # to distinguish between B = button and F = blind player, first to act
        self.player_button.name = 'B'
        self.player_blind.name = 'F'
        

        # For debug. Start just by listing actions taken, in readable format.
        self.hand_history = []

        self.pot_size = 0.0
        self.live = False
        
    def reset(self):
        # TODO
        raise NotImplementedError()

    # Forced actions.
    def post_big_blind(self):
        assert(self.action_on == self.player_blind)
        action = PostBigBlind(actor_name = self.action_on.name, pot_size = self.pot_size)
        self.process_action(action, pass_control = True)
        #self.pass_control()

    def post_small_blind(self):
        assert(self.action_on == self.player_button)
        action = PostSmallBlind(actor_name = self.action_on.name, pot_size = self.pot_size)
        self.process_action(action, pass_control = False)
        # DO NOT pass_control()

    # Make any last checks, add pots to the bet, (usually), pass control to the next player
    def process_action(self, action, pass_control = True):
        # TODO: Add checks, that the action make sense.

        # Handle the betting. Deduct from the player. Add to the pot.
        bet_size = action.bet_size
        self.pot_size += bet_size
        self.action_on.bet_this_hand += bet_size
        self.action_on.bet_this_street += bet_size

        # The only action that ends the round immediately... is a fold.
        # We can still pass control. But before taking another action, player must check that both hands still live!
        if action.action_type == FOLD_HAND:
            print('Player %s is folding! Make sure that no further actions' % self.action_on.name)
            self.action_on.live = False

        self.hand_history.append(action)

        if pass_control:
            self.pass_control()

    # Action on the *other* player
    def pass_control(self):
        if self.action_on == self.player_blind:
            self.action_on = self.player_button
        elif self.action_on == self.player_button:
            self.action_on = self.player_blind
        else:
            assert False, 'Control on unknown player %s' % self.action_on
        
        # Do the same for off-action player. 
        if self.action_off == self.player_blind:
            self.action_off = self.player_button
        elif self.action_off == self.player_button:
            self.action_off = self.player_blind
        else:
            assert False, 'Control off of unknown player %s' % self.action_off

        print('Passed control, to player %s' % self.action_on.name)
            

    # Play full betting round on a loop... until action ends.
    def play_betting_round(self, round):
        # Check for conditions that must be met, to continue.
        if not(self.player_blind.live and self.player_button.live):
            print('Exiting betting round %d. Since one of the players is not live (folded)' % round)
            return
        
        # Determine, if we are facing a raise, can call, etc. 
        # NOTE: This needs to handle calling or raising the SB.
        # TODO: Add correctness, and sanity checks.
        bet_on_action = self.action_on.bet_this_street
        bet_off_action = self.action_off.bet_this_street

        bet_this_street = SMALL_BET_SIZE
        is_small_street = True
        if round >= DRAW_2_BET_ROUND:
            # print('since round: %d >= %d, this is big-bet street' % (round, DRAW_2_BET_ROUND))
            bet_this_street = BIG_BET_SIZE
            is_small_street = False
        max_bet = MAXIMUM_BETS_ALLOWED * bet_this_street

        assert bet_on_action <= max_bet and bet_off_action <= max_bet, 'Max_bet = %d, but players have bet %d and %d' % (max_bet, bet_on_action, bet_off_action)

        # Now, based on situation, collect actions are allowed for active play...
        allowed_actions = set([])
        if bet_on_action > bet_off_action:
            assert bet_on_action <= bet_off_action, ('On action %s, but somehow has bet %d > %d' %
                                                     (self.action_on.name, bet_on_action, bet_off_action))
        elif bet_on_action == bet_off_action:
            # If more betting left, option to check or bet
            if bet_on_action < max_bet:
                allowed_actions.add(CHECK_HAND)
                if bet_on_action == 0:
                    allowed_actions.add(BET_SMALL_STREET if is_small_street else BET_BIG_STREET)
                else:
                    # If already put money in, always a raise... BB for example.
                    allowed_actions.add(RAISE_SMALL_STREET if is_small_street else RAISE_BIG_STREET)
        else:
            # If we're facing a bet, always option to call or fold.
            allowed_actions.add(FOLD_HAND)
            allowed_actions.add(CALL_SMALL_STREET if is_small_street else CALL_BIG_STREET)

            # If opponent's bet hasn't topped the max.... we can also raise.
            # TODO: Determine if the raise constitutes a 3-bet or 4-bet... if we ever implement those.
            # NOTE: Why do we care? These are different actions. Certainly to a human. So maybe.
            if bet_off_action < max_bet:
                allowed_actions.add(RAISE_SMALL_STREET if is_small_street else RAISE_BIG_STREET)

        # Exit quickly... if there are no actions (thus street is capped out)
        if not allowed_actions:
            print('No more allowed actions! Street must be capped betting.')
            return
        
        # If still here... ther are legal actions that a player may take!
        print('Allowed actions for player %s: %s' % (self.action_on.name, [actionName[action] for action in allowed_actions]))

        # TODO: Here the agent... would choose a good action.
        print('TODO: Agent for player %s should choose a good action.' % self.action_on.name)
        best_action = self.action_on.choose_action(actions=allowed_actions, 
                                                   round=round, 
                                                   bets_this_round = max(bet_on_action, bet_off_action) / bet_this_street,
                                                   has_button = (self.action_on == self.player_button))

        # If action returned, complete the action... and keep going
        if (best_action):
            print(best_action)
            print('best action chosen is %s' % actionName[best_action])
            # We keep betting after this action... as long last action allows it.
            keep_betting = True
            # Create the action
            if best_action == CALL_SMALL_STREET:
                action = CallSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
                if not(round == PRE_DRAW_BET_ROUND and bet_off_action == BIG_BLIND_SIZE):
                    print('chosen action, closes the action')
                    keep_betting = False
            elif best_action == CALL_BIG_STREET:
                action = CallBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
                if not(round == PRE_DRAW_BET_ROUND and bet_off_action == BIG_BLIND_SIZE):
                    print('chosen action, closes the action')
                    keep_betting = False
            elif best_action == BET_SMALL_STREET:
                action = BetSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
            elif best_action == BET_BIG_STREET:
                action = BetBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
            elif best_action == RAISE_SMALL_STREET:
                action = RaiseSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
            elif best_action == RAISE_BIG_STREET:
                action = RaiseBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
            elif best_action == FOLD_HAND:
                action = FoldStreet(self.action_on.name, self.pot_size)
                print('chosen action, closes the action')
                keep_betting = False
            elif best_action == CHECK_HAND:
                action = CheckStreet(self.action_on.name, self.pot_size)

                # Logic for checking is a bit tricky. Action ended if button checks... except on first, when F player checks ends it.
                if (round == PRE_DRAW_BET_ROUND and self.action_on == self.player_blind) or (round != PRE_DRAW_BET_ROUND and self.action_on == self.player_button):
                    print('chosen action, closes the action')
                    keep_betting = False
                else:
                    print('this check... does not end the action.')
            else:
                assert False, 'Unknown best_action %s' % actionName[best_action]

            self.process_action(action, pass_control = True)
            if keep_betting:
                print('chosen action, allows further betting.')
                self.play_betting_round(round)
            
            
            



    # Assumes that everything has been initialized, or reset as needed.
    # Through constants, hard-coded to 50-100 blinds. And 100-200 betting. 
    def play_single_hand(self):
        # Deal initial hands to players
        draw_hand_blind = PokerHand()
        deal_cards = self.deck.deal(5)
        draw_hand_blind.deal(deal_cards)
        self.player_blind.draw_hand = draw_hand_blind
        self.player_blind.live = True
        self.player_blind.bet_this_hand = 0.0
        self.player_blind.bet_this_street = 0.0

        draw_hand_button = PokerHand()
        deal_cards = self.deck.deal(5)
        draw_hand_button.deal(deal_cards)
        self.player_button.draw_hand = draw_hand_button
        self.player_button.live = True
        self.player_button.bet_this_hand = 0.0
        self.player_button.bet_this_street = 0.0

        print('starting new hand. Blind %s and button %s' % (hand_string(self.player_blind.draw_hand.dealt_cards),
                                                             hand_string(self.player_button.draw_hand.dealt_cards)))

        # Post blinds -- hack to foce bets
        self.pot_size = 0.0
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button
        self.post_big_blind()
        self.post_small_blind()

        print('After blinds posted... pot %d, player %s has bet %d, player %s has bet %d' % (self.pot_size, 
                                                                                             self.player_blind.name,
                                                                                             self.player_blind.bet_this_hand,
                                                                                             self.player_button.name,
                                                                                             self.player_button.bet_this_hand))

        print(self.hand_history)

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=3)                                                           
        self.player_button.update_hand_value(num_draws=3)
        
        # Play out a full round of betting.
        # Will go back & forth between players betting, until
        # A. Player calls (instead of raise or fold)
        # B. Player folds (thus concedes the hand)
        self.play_betting_round(round = PRE_DRAW_BET_ROUND)

        print(self.hand_history)

        if self.player_blind.live and self.player_button.live:
            print('\n-- 1st draw --\n')
            print('Both players live. So continue betting after the 1st draw.')
        else:
            return

        # Make draws for each player, in turn
        if (self.live):
            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, num_draws=3)
            self.player_button.draw(deck=self.deck, num_draws=3)

        # TODO: Switch to pre-draw & evaluate heuristics in a function?
        draw_hand_blind = PokerHand()
        draw_hand_blind.deal(self.player_blind.draw_hand.final_hand)
        self.player_blind.draw_hand = draw_hand_blind
        draw_hand_button = PokerHand()
        draw_hand_button.deal(self.player_button.draw_hand.final_hand)
        self.player_button.draw_hand = draw_hand_button

        # Next round. We bet again, then draw again
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=2)                                                           
        self.player_button.update_hand_value(num_draws=2)

        self.play_betting_round(round = DRAW_1_BET_ROUND)
        
        print(self.hand_history)

        if self.player_blind.live and self.player_button.live:
            print('\n-- 2nd draw --\n')
            print('Both players live. So continue betting after the 2nd draw.')
        else:
            return

        # Make draws for each player, in turn
        if (self.live):
            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, num_draws=2)
            self.player_button.draw(deck=self.deck, num_draws=2)

        # TODO: Switch to pre-draw & evaluate heuristics in a function?
        draw_hand_blind = PokerHand()
        draw_hand_blind.deal(self.player_blind.draw_hand.final_hand)
        self.player_blind.draw_hand = draw_hand_blind
        draw_hand_button = PokerHand()
        draw_hand_button.deal(self.player_button.draw_hand.final_hand)
        self.player_button.draw_hand = draw_hand_button

        # Next round. We bet again, then draw again
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=1)                                                           
        self.player_button.update_hand_value(num_draws=1)

        self.play_betting_round(round = DRAW_2_BET_ROUND)
        
        print(self.hand_history)

        if self.player_blind.live and self.player_button.live:
            print('\n-- 3rd draw --\n')
            print('Both players live. So continue betting after the 3rd draw.')
        else:
            return

        # Make draws for each player, in turn
        if (self.live):
            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, num_draws=1)
            self.player_button.draw(deck=self.deck, num_draws=1)

        # NOTE: Do *not* copy hands for final round of betting. It screws up "showdown" evaluation and debug. Evaluate on "final_hand" instead.
        """
        # TODO: Switch to pre-draw & evaluate heuristics in a function?
        draw_hand_blind = PokerHand()
        draw_hand_blind.deal(self.player_blind.draw_hand.final_hand)
        self.player_blind.draw_hand = draw_hand_blind
        draw_hand_button = PokerHand()
        draw_hand_button.deal(self.player_button.draw_hand.final_hand)
        self.player_button.draw_hand = draw_hand_button
        """

        # Next round. We bet again, then draw again
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=0)                                                           
        self.player_button.update_hand_value(num_draws=0)

        self.play_betting_round(round = DRAW_3_BET_ROUND)
        
        print(self.hand_history)

        print('Made it all the way, with betting on the river')


    # Declare a winner... assuming hand ends now.
    def get_hand_result(self, cashier):
        if self.player_blind.live and not self.player_button.live:
            print('\nPlayer F wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_blind.draw_hand))
        elif not self.player_blind.live and self.player_button.live:
            print('\nPlayer B wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_button.draw_hand))
        elif not self.player_blind.live and not self.player_button.live:
            print('Error! both players are dead.')
        else:
            # TODO: Handle ties & split pots!
            best_hand = cashier.showdown([self.player_blind.draw_hand, self.player_button.draw_hand])
            if best_hand == self.player_blind.draw_hand:
                print('\nPlayer F wins on showdown! %d chips in the pot.' % self.pot_size)
            elif best_hand == self.player_button.draw_hand:
                print('\nPlayer B wins on showdown! %d chips in the pot.' % self.pot_size)
            else:
                print('Tie! or Error. %d chips in the pot.' % self.pot_size)

            if best_hand:
                print(str(best_hand))


# As simply as possible, simulate a full round of triple draw. Have as much logic as possible, contained in the objects
# Actors:
# cashier -- evaluates final hands
# deck -- dumb deck, shuffled once, asked for next cards
# dealer -- runs the game. Tracks pot, propts players for actions. Decides when hand ends.
# players -- acts directly on a poker hand. Makes draw and betting decisions... when propted by the dealer
def game_round(round, cashier, player_button=None, player_blind=None):
    print '\n-- New Round %d --\n' % round
    deck = PokerDeck(shuffle=True)

    dealer = TripleDrawDealer(deck=deck, player_button=player_button, player_blind=player_blind)
    dealer.play_single_hand()

    # TODO: Should output results.
    # TODO: Also output game history for training data

    dealer.get_hand_result(cashier)

    


# Play a bunch of hands.
# For now... just rush toward full games, and skip details, or fill in with hacks.
def play(sample_size, output_file_name, model_filename=None):
    # Compute hand values, or compare hands.
    cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules

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
            game_round(round, cashier, player_button=player_one, player_blind=player_two)

            print ('hand %d took %.1f seconds...' % (round, time.time() - now))

            round += 1

            #sys.exit(-3)

    except KeyboardInterrupt:
        pass

    print('completed %d rounds of heads up play' % round)
    sys.stdout.flush()

if __name__ == '__main__':
    samples = 100
    output_file_name = '%d_samples_model_choices.csv' % samples

    # Input model filename if given
    # TODO: set via command line flagz
    model_filename = None
    if len(sys.argv) >= 2:
        model_filename = sys.argv[1]

        # Uniquely ID details
        output_file_name = '%d_samples_model_choices_%s.csv' % (samples, model_filename)

    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name, model_filename=model_filename)
