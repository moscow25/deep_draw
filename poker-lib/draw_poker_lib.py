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
from holdem_lib import * # if we want to support holdem hands!
from poker_util import *

from draw_poker import cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Methods, useful for organizing & evaluating a draw poker match. Include everything that doesn't involve model building, model evaluation.
"""

# Heuristics, to evaluate hand actions. On 0-1000 scale, where wheel is 1000 points, and bad hand is 50-100 points.
# Meant to map to rough % of winning at showdown. Tuned for ring game, so random hand << 500.

# For 'deuce,' random heuristic 0.300 worked. We need to either tie this to FORMAT, etc.
# RANDOM_HAND_HEURISTIC_BASELINE = 0.300 # baseline, before looking at any cards.

# For 'holdem,' we need a higher baseline heuristic. Average hand is by definition 0.500. Since we literally compare to random hands, HU.
RANDOM_HAND_HEURISTIC_BASELINE = 0.4000 # baseline, before considering cards

# Cashier for 2-7 lowball. Evaluates hands, as well as compares hands.
class DeuceLowball(PayoutTable):
    #def __init__():

    # In this context, payout means 0-1000 heuristic value, for a final hand.
    def payout(self, hand):
        hand.evaluate() # computes ranks, including for 2-7 lowball
        return hand.deuce_heuristic

    # Compare hands.
    # TODO: Hand split pots, other % payouts. Should really output [hand_id: % pot]
    def showdown(self, hands):
        #print('using DeuceLowball to compare hands')
        # As a hack... output hand with best (2-7) rank. Ties go to best position...
        best_rank = 0
        best_hand = None
        for hand in hands:
            hand.evaluate()
            if hand.rank > best_rank:
                best_hand = hand
                best_rank = hand.rank
            elif hand.rank == best_rank:
                # print('Need to implement ties & splits!')
                return None
                #raise NotImplementedError()
        return best_hand

# Cashier for 2-7 lowball. Evaluates hands, as well as compares hands.
class HoldemCashier(PayoutTable):
    #def __init__():

    """
    # In this context, payout means 0-1000 heuristic value, for a final hand.
    def payout(self, hand):
        hand.evaluate() # computes ranks, including for 2-7 lowball
        return hand.deuce_heuristic
        """

    # Compare hands.
    # TODO: Hand split pots, other % payouts. Should really output [hand_id: % pot]
    def showdown(self, hands):
        #print('using HoldemCashier to compare hands')
        # As a hack... output hand with best (2-7) rank. Ties go to best position...
        best_rank = 1000000
        best_hand = None
        for hand in hands:
            hand.evaluate()
            if hand.rank < best_rank:
                best_hand = hand
                best_rank = hand.rank
            elif hand.rank == best_rank:
                print('Need to implement ties & splits!')
                return None
                #raise NotImplementedError()
        return best_hand

# At risk of over-engineering, use a super-class to encapsulate each type
# of poker action. Specifically, betting, checking, raising.
# NOTE: Usually, is prompted, decides on an action, and passes control.
# (but not always, as there are forced bets, etc)
# This system strong assumes, that actions take place in order, by a dealer. Poker is a turn-based game.
class PokerAction:
    # TODO: Copy of the game state (hands, etc)? Zip of the information going into this action?
    def __init__(self, action_type, actor_name, pot_size, bet_size, format = 'deuce'):
        self.type = action_type
        self.name = actionName[action_type]
        self.format = format

        # For now... just a hack to distinguish between B = button and F = blind player, first to act
        self.actor_name = actor_name
        self.pot_size = pot_size # before the action
        self.bet_size = bet_size # --> bet being made with *this action*
        self.pot_odds = (pot_size / bet_size if bet_size else 0.0)
        self.value = RANDOM_HAND_HEURISTIC_BASELINE # baseline of baselines
        self.hand = None
        self.best_draw = None
        self.hand_after = None
        if format == 'deuce':
            self.cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules
        elif format == 'holdem':
            self.cashier = HoldemCashier() # Categories, in hold'em format.
    
    # What's currently winning? (not expected to win, but if we have to showdown)
    # Assumes that we know our hand and opponent hand. But can also handle errors.
    # TODO: Use this to estimate future winning percentage? In allin situations...
    def current_win_percentage(self):
        # TODO: Fix "win%" for non-deuce hands. Natural for holdem... except we need to implement preflop comps (or skip)
        #if (not self.hand) or (not self.oppn_hand) or not(format == 'deuce'):
        #    return 0.5 # unknown, so 50/50
        #print('calculating win percentages')
        #print([self.format, self.hand, self.oppn_hand, self.best_draw])

        if self.format == 'deuce' and self.hand and self.oppn_hand:
            our_hand = PokerHand(cards = self.hand)
            oppn_hand = PokerHand(cards = self.oppn_hand)
            winners = self.cashier.showdown([our_hand, oppn_hand])
            if winners == our_hand:
                return 1.0
            elif winners == oppn_hand:
                return 0.0
            else:
                return 0.5 # chop, or unknown.
        elif self.format == 'holdem' and self.hand and self.oppn_hand and self.best_draw:
            # Project community hand, if available. Can only evaluate if flop or more (otherwise, no 5 card hand)
            flop = []
            turn = []
            river = []
            if self.best_draw:
                flop = self.best_draw
            if self.hand_after:
                if len(self.hand_after) == 1:
                    turn = self.hand_after
                elif len(self.hand_after) == 2:
                    turn = [self.hand_after[0]]
                    river = [self.hand_after[1]]
                else:
                    assert False, 'Unparsable turn/river %s' % self.hand_after
            community = HoldemCommunityHand(flop=flop, turn=turn, river=river)
            our_hand = HoldemHand(cards=self.hand, community=community)
            oppn_hand = HoldemHand(cards=self.oppn_hand, community=community)
            our_hand.evaluate()
            oppn_hand.evaluate()
            winners = self.cashier.showdown([our_hand, oppn_hand])
            if winners == our_hand:
                return 1.0
            elif winners == oppn_hand:
                return 0.0
            else:
                return 0.5 # chop, or unknown.
        else:
            return 0.5 # unknown, so 50/50


    # Context needed, to record actions properly to CVS...
    def add_context(self, hand, draws_left, position, 
                    actions_this_round, actions_full_hand, 
                    value = RANDOM_HAND_HEURISTIC_BASELINE, bet_this_hand = 0,
                    num_cards_kept = 0, num_opponent_kept = 0, bet_model = '', oppn_hand = None,
                    bet_val_vector=[], act_val_vector=[], num_draw_vector=[]):
        self.hand = list(hand) # makes copy of 5-card array
        self.oppn_hand = list(oppn_hand) # copy of opponent's current hand
        self.current_hand_win = self.current_win_percentage() # where are we at, right now? [Current hands comparison]

        # Save vectors from the for our debug pleasure
        self.bet_val_vector = bet_val_vector
        self.act_val_vector = act_val_vector
        self.num_draw_vector = num_draw_vector

        self.draws_left = draws_left
        self.value = value # heuristic estimation
        self.bet_this_hand = bet_this_hand # how much did we already commit into this hand, previously [useful for calculating value]
        self.position = position # position = enum
        self.num_cards_kept = num_cards_kept
        self.num_opponent_kept = num_opponent_kept

        # Tag the bet model, if appropriate
        self.bet_model = bet_model
        
        # Array of PokerAction items -> '011' = check, bet, raise
        # NOTE: Blinds are skipped (for space), since always there. FOLD ends the action.
        self.actions_this_round_string = ''
        for action in actions_this_round:
            if action.type in ALL_BETS_SET:
                self.actions_this_round_string += '1'
            elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                self.actions_this_round_string += '0'
            else:
                # Don't encode non-bets
                continue

        # Actions for the entire hand.
        # TODO: Add breakers, '|' for draw event.
        self.actions_full_hand_string = ''
        for action in actions_full_hand:
            if action.type in ALL_BETS_SET:
                self.actions_full_hand_string += '1'
            elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                self.actions_full_hand_string += '0'
            else:
                # Don't encode non-bets
                continue

    # After hand is over... add information about wins & losses
    # Winners: {'name': chips} to handle split pots, etc
    def update_result(self, winners, final_bets, hand_num, running_average):
        # How much we won
        result = 0.0
        if winners.has_key(self.actor_name):
            result = winners[self.actor_name]

        # How much we bet in total, on this hand.
        final_bet = final_bets[self.actor_name]

        # How much did we bet... not including dead money before this bet?
        margin_bet = final_bet
        if self.bet_this_hand:
            margin_bet -= self.bet_this_hand

        # Our value, having taken this action. Including this bet, and all future bets. (But excluding previous bets)
        margin_result = result - margin_bet 

        self.result = result
        self.total_bet = final_bet
        self.margin_bet = margin_bet
        self.margin_result = margin_result

        # Try to break up margin result (rewards) into 'current' and 'future' values.
        # 'current' includes pot before the bet, and this current bet.
        # 'future' includes all future bets won or lost.
        # NOTE: Does not properly support chops, split pots, etc. Assumes that we won the pot, lost the pot, or folded.
        if margin_result == 0.0:
            self.current_margin_result = 0
            self.future_margin_result = 0
        elif margin_result > 0:
            self.current_margin_result = self.pot_size
            self.future_margin_result = margin_result - self.current_margin_result
        else:
            self.current_margin_result = -1 * self.bet_size
            self.future_margin_result = margin_result - self.current_margin_result

        # Details for debug
        self.hand_num = hand_num
        self.running_average = running_average # NOTE: A step behind, but.... that's ok.
        
        
    # Consise summary, of the action taken.
    def __str__(self):
        return('%s(%s)\tPot: %d\tBet: %d' % (self.name, self.actor_name, self.pot_size, self.bet_size))
        #raise NotImplementedError()

    # Return array of outputs, corresponding to CSV header map order. Empty fields are ''
    def csv_output(self, header_map):
        output_map = {}
        if hasattr(self, 'hand') and self.hand:
            output_map['hand'] = hand_string(self.hand)

        # Draw hand information, if present
        if self.best_draw:
            output_map['best_draw'] = hand_string(self.best_draw)
        if self.hand_after:
            output_map['hand_after'] = hand_string(self.hand_after)

        if hasattr(self, 'draws_left'):
            output_map['draws_left'] = self.draws_left
        output_map['value_heuristic'] = int(self.value * 100000) / 100000.0 # round for better readability
        if hasattr(self, 'position'):
            output_map['position'] = self.position
        if hasattr(self, 'num_cards_kept'):
            output_map['num_cards_kept'] = self.num_cards_kept
            output_map['num_opponent_kept'] = self.num_opponent_kept
        if hasattr(self, 'bet_model'):
            output_map['bet_model'] = self.bet_model
        output_map['action'] = self.name
        output_map['pot_size'] = self.pot_size
        output_map['bet_size'] = self.bet_size
        output_map['pot_odds'] = self.pot_odds
        if hasattr(self, 'bet_this_hand'):
            output_map['bet_this_hand'] = self.bet_this_hand
        if hasattr(self, 'actions_this_round_string'):
            output_map['actions_this_round'] = self.actions_this_round_string
        if hasattr(self, 'actions_full_hand_string'):
            output_map['actions_full_hand'] = self.actions_full_hand_string

        # Results, if present
        if hasattr(self, 'result'):
            output_map['total_bet'] = self.total_bet
            output_map['result'] = self.result
            output_map['margin_bet'] = self.margin_bet
            output_map['margin_result'] = self.margin_result
            
            # Break up into 'current' and 'future' results... (to apply discount later, if we like)
            output_map['current_margin_result'] = self.current_margin_result
            output_map['future_margin_result'] = self.future_margin_result

        # Opponent hand, if present.
        if hasattr(self, 'oppn_hand'):
            output_map['oppn_hand'] = hand_string(self.oppn_hand)
            output_map['current_hand_win'] = self.current_hand_win

        # TODO: Info & debug
        # ['hand_num', 'running_average', 'bet_val_debug', 'act_val_debug', 'num_draw_debug']
        if hasattr(self, 'hand_num'):
            output_map['hand_num'] = self.hand_num
        if hasattr(self, 'running_average'):
            output_map['running_average'] = self.running_average
        if hasattr(self, 'bet_val_vector'):
            output_map['bet_val_vector'] = self.bet_val_vector
            output_map['act_val_vector'] = self.act_val_vector
            output_map['num_draw_vector'] = self.num_draw_vector
        
        # ['hand', 'draws_left', 'bet_model', 'value_heuristic', 'position', 'num_cards_kept', 'num_opponent_kept', 'best_draw', 'hand_after', 'action', 'pot_size', 'bet_size', 'pot_odds', 'bet_this_hand', 'actions_this_round', 'actions_full_hand', 'total_bet', 'result', 'margin_bet', 'margin_result', 'current_margin_result', 'future_margin_result', 'oppn_hand', 'current_hand_win', 'hand_num', 'running_average', 'bet_val_vector', 'act_val_vector', 'num_draw_vector']
        output_row = VectorFromKeysAndSparseMap(keys=header_map, sparse_data_map=output_map, default_value = '')
        return output_row

# Encode a draw event. Doesn't 100% fit in... but confusing not to record the draws, along with the bets.
class DrawAction(PokerAction):
    def __init__(self, actor_name, pot_size, hand_before, best_draw, hand_after):
        PokerAction.__init__(self, action_type = DRAW_ACTION, actor_name = actor_name, pot_size = pot_size, bet_size = 0, format='deuce')
        self.hand = list(hand_before)
        self.best_draw = list(best_draw)
        self.hand_after = list(hand_after)

# Simple encoding, for each possible action
class PostBigBlind(PokerAction):
    def __init__(self, actor_name, pot_size, format):
        PokerAction.__init__(self, action_type = POST_BIG_BLIND, actor_name = actor_name, pot_size = pot_size, bet_size = BIG_BLIND_SIZE, format=format)

class PostSmallBlind(PokerAction):
    def __init__(self, actor_name, pot_size, format):
        PokerAction.__init__(self, action_type = POST_SMALL_BLIND, actor_name = actor_name, pot_size = pot_size, bet_size = SMALL_BLIND_SIZE, format=format)

class CheckStreet(PokerAction):
    def __init__(self, actor_name, pot_size, format):
        PokerAction.__init__(self, action_type = CHECK_HAND, actor_name = actor_name, pot_size = pot_size, bet_size = 0, format=format)

class FoldStreet(PokerAction):
    def __init__(self, actor_name, pot_size, format):
        PokerAction.__init__(self, action_type = FOLD_HAND, actor_name = actor_name, pot_size = pot_size, bet_size = 0, format=format)

# Cost of the other actions... to be computed, from $ spent by player on this street, already.
# NOTE: Think of it like internet, with chips left in front of players... until betting round is finished.
class CallSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = CALL_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)

class CallBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = CALL_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)

class BetSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street + SMALL_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = BET_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)

class BetBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street + BIG_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = BET_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)

class RaiseSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street + SMALL_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = RAISE_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)

class RaiseBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street, format):
        total_bet_size = biggest_bet_this_street + BIG_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = RAISE_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet, format=format)


# Should inherit from more general dealer class... when we need one.
# TODO: Make sure we can handle multiple types of players, not just AI player. (For example, manual-input player)
# TODO: Think about saving 'game state,' loading 'game state' and pausing action. Useful for API, and for simulation.
class TripleDrawDealer():
    def __init__(self, deck, player_button, player_blind, format = 'deuce'):
        self.deck = deck # Assume it's shuffled, if needs to be shuffled
        self.player_button = player_button # player to act last on every street
        self.player_blind = player_blind # player to act first, except when posting forced blind
        self.format = format # 'deuce' triple draw, or 'holdem', etc?

        # Hack, for convenience.
        # to distinguish between B = button and F = blind player, first to act
        self.player_button.name = 'B'
        self.player_blind.name = 'F'
        

        # For debug. Start just by listing actions taken, in readable format.
        self.hand_history = []
        self.hand_history_this_round = []

        self.pot_size = 0.0
        self.live = False
        
    def reset(self):
        # TODO
        raise NotImplementedError()

    # Forced actions.
    def post_big_blind(self):
        assert(self.action_on == self.player_blind)
        round = PRE_DRAW_BET_ROUND
        action = PostBigBlind(actor_name = self.action_on.name, pot_size = self.pot_size, format = self.format)
        action.add_context(hand=(self.action_on.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_on.draw_hand.final_hand), 
                           draws_left=drawsLeft[round], 
                           position = POSITION_BUTTON if self.action_on == self.player_button else POSITION_BLIND, 
                           actions_this_round = self.hand_history_this_round,
                           actions_full_hand = self.hand_history,
                           bet_this_hand = self.action_on.bet_this_hand,
                           bet_model = self.action_on.player_tag(),
                           oppn_hand = (self.action_off.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_off.draw_hand.final_hand))
        self.process_action(action, pass_control = True)
        #self.pass_control()

    def post_small_blind(self):
        assert(self.action_on == self.player_button)
        round = PRE_DRAW_BET_ROUND
        action = PostSmallBlind(actor_name = self.action_on.name, pot_size = self.pot_size, format = self.format)
        action.add_context(hand=(self.action_on.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_on.draw_hand.final_hand), 
                           draws_left=drawsLeft[round], 
                           position = POSITION_BUTTON if self.action_on == self.player_button else POSITION_BLIND, 
                           actions_this_round = self.hand_history_this_round,
                           actions_full_hand = self.hand_history,
                           bet_this_hand = self.action_on.bet_this_hand,
                           bet_model = self.action_on.player_tag(),
                           oppn_hand = (self.action_off.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_off.draw_hand.final_hand))
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
        if action.type == FOLD_HAND:
            print('Player %s is folding! Make sure that no further actions' % self.action_on.name)
            self.action_on.live = False

        # Add to the history. Both for whole hand, and for this round. 
        # Why this round? Very important for context, w/r/t betting decisions. Previous rounds less important.
        self.hand_history.append(action)
        self.hand_history_this_round.append(action)

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

        #print('Passed control, to player %s' % self.action_on.name)
            

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

        # Here the agent... would choose a good action.
        best_action = self.action_on.choose_action(actions=allowed_actions, 
                                                   round=round, 
                                                   bets_this_round = max(bet_on_action, bet_off_action) / bet_this_street,
                                                   has_button = (self.action_on == self.player_button),
                                                   pot_size=self.pot_size, 
                                                   actions_this_round=self.hand_history_this_round,
                                                   actions_whole_hand=self.hand_history,
                                                   cards_kept=self.action_on.num_cards_kept, 
                                                   opponent_cards_kept=self.action_off.num_cards_kept)

        # If action returned, complete the action... and keep going
        if (best_action):
            # print('\nBest action chose ->  %s' % (actionName[best_action]))
            # We keep betting after this action... as long last action allows it.
            keep_betting = True
            # Create the action
            if best_action == CALL_SMALL_STREET:
                action = CallSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
                if not(round == PRE_DRAW_BET_ROUND and bet_off_action == BIG_BLIND_SIZE):
                    #print('\nchosen action, closes the action')
                    keep_betting = False
            elif best_action == CALL_BIG_STREET:
                action = CallBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
                if not(round == PRE_DRAW_BET_ROUND and bet_off_action == BIG_BLIND_SIZE):
                    #print('\nchosen action, closes the action')
                    keep_betting = False
            elif best_action == BET_SMALL_STREET:
                action = BetSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
            elif best_action == BET_BIG_STREET:
                action = BetBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
            elif best_action == RAISE_SMALL_STREET:
                action = RaiseSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
            elif best_action == RAISE_BIG_STREET:
                action = RaiseBigStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
            elif best_action == FOLD_HAND:
                action = FoldStreet(self.action_on.name, self.pot_size, format=self.format)
                #print('\nchosen action, closes the action')
                keep_betting = False
            elif best_action == CHECK_HAND:
                action = CheckStreet(self.action_on.name, self.pot_size, format=self.format)

                # Logic for checking is a bit tricky. Action ended if button checks... except on first, when F player checks ends it.
                if (round == PRE_DRAW_BET_ROUND and self.action_on == self.player_blind) or (round != PRE_DRAW_BET_ROUND and self.action_on == self.player_button):
                    #print('\nchosen action, closes the action')
                    keep_betting = False
                else:
                    print('\nthis check... does not end the action.')
            else:
                assert False, 'Unknown best_action %s' % actionName[best_action]

            # HACK: Use "best_draw" and "hand_after" fields, to record the flop, turn and river in draw games.
            if self.format == 'holdem':
                action.best_draw = self.action_on.holdem_hand.community.flop
                action.hand_after = self.action_on.holdem_hand.community.turn + self.action_on.holdem_hand.community.river

            # Add additional information to the action.
            # NOTE: More convenient to do this in one place, since action-independent context...
            action.add_context(hand=(self.action_on.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_on.draw_hand.final_hand), 
                               draws_left=drawsLeft[round], 
                               position = POSITION_BUTTON if self.action_on == self.player_button else POSITION_BLIND, 
                               actions_this_round = self.hand_history_this_round,
                               actions_full_hand = self.hand_history,
                               value = self.action_on.heuristic_value,
                               bet_this_hand = self.action_on.bet_this_hand,
                               num_cards_kept = self.action_on.num_cards_kept, 
                               num_opponent_kept = self.action_off.num_cards_kept,
                               bet_model = self.action_on.player_tag(),
                               oppn_hand = (self.action_off.draw_hand.dealt_cards if round < DRAW_3_BET_ROUND else self.action_off.draw_hand.final_hand),
                               bet_val_vector = self.action_on.bet_val_vector,
                               act_val_vector = self.action_on.act_val_vector,
                               num_draw_vector = self.action_on.num_draw_vector)

            self.process_action(action, pass_control = True)
            if keep_betting:
                #print('chosen action, allows further betting.')
                self.play_betting_round(round)
            

    # Assumes that everything has been initialized, or reset as needed.
    # Through constants, hard-coded to 50-100 blinds. And 100-200 betting. 
    # NOTE: We've expanded it to include flop games. split up later... but good to share common infrastructure.
    def play_single_hand(self):
        # If community card game, create blank community object first.
        if self.format == 'holdem':
            self.community = HoldemCommunityHand()
        else:
            self.community = None

        # Deal initial hands to players
        # NOTE: Nice thing with hold'em hands... we're done with player hands, except to update shared community cards.
        if self.format == 'holdem':
            deal_cards = self.deck.deal(2)
            holdem_hand_blind = HoldemHand(cards = deal_cards, community = self.community)
            self.player_blind.holdem_hand = holdem_hand_blind
            self.player_blind.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE # Change this, as needed
            self.player_blind.draw_hand = holdem_hand_blind # For some debug convenience...
        else:
            draw_hand_blind = PokerHand()
            deal_cards = self.deck.deal(5)
            draw_hand_blind.deal(deal_cards)
            self.player_blind.draw_hand = draw_hand_blind
            self.player_blind.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 
            self.player_blind.num_cards_kept = 0

        self.player_blind.live = True
        self.player_blind.bet_this_hand = 0.0
        self.player_blind.bet_this_street = 0.0

        if self.format == 'holdem':
            deal_cards = self.deck.deal(2)
            holdem_hand_button = HoldemHand(cards = deal_cards, community = self.community)
            self.player_button.holdem_hand = holdem_hand_button
            self.player_button.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE # Change this, as needed
            self.player_button.draw_hand = holdem_hand_button
        else:
            draw_hand_button = PokerHand()
            deal_cards = self.deck.deal(5)
            draw_hand_button.deal(deal_cards)
            self.player_button.draw_hand = draw_hand_button
            self.player_button.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 
            self.player_button.num_cards_kept = 0

        self.player_button.live = True
        self.player_button.bet_this_hand = 0.0
        self.player_button.bet_this_street = 0.0

        self.hand_history_this_round = []

        if not(self.player_blind.is_human or self.player_button.is_human):
            print('starting new hand [format = %s]. Blind %s and button %s' % (self.format, 
                                                                               hand_string(self.player_blind.draw_hand.dealt_cards),
                                                                               hand_string(self.player_button.draw_hand.dealt_cards)))
        else:
            print('starting new hand...')

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

        # print(self.hand_history)

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

        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if self.format != 'holdem':
            print('\n-- 1st draw --\n')
            print('Both players live. So continue betting after the 1st draw.')

            # Similar to "player.move()" in the single-draw video poker context
            self.player_blind.draw(deck=self.deck, num_draws=3, 
                                   has_button = False,
                                   pot_size=self.pot_size, 
                                   actions_this_round=self.hand_history_this_round, 
                                   actions_whole_hand=self.hand_history,
                                   cards_kept=self.player_blind.num_cards_kept, 
                                   opponent_cards_kept=self.player_button.num_cards_kept)
            self.player_button.draw(deck=self.deck, num_draws=3,
                                    has_button = True,
                                    pot_size=self.pot_size, 
                                    actions_this_round=self.hand_history_this_round, 
                                    actions_whole_hand=self.hand_history,
                                    cards_kept=self.player_button.num_cards_kept, 
                                    opponent_cards_kept=self.player_blind.num_cards_kept)
            draw_action = DrawAction(actor_name = self.player_blind.name, pot_size = self.pot_size, 
                                     hand_before = self.player_blind.draw_hand.dealt_cards, 
                                     best_draw = self.player_blind.draw_hand.held_cards, 
                                     hand_after = self.player_blind.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_blind.draw_hand.dealt_cards,
                                    draws_left=3, 
                                    position = POSITION_BLIND, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_blind.heuristic_value,
                                    bet_this_hand = self.player_blind.bet_this_hand,
                                    bet_model = self.player_blind.player_tag(),
                                    oppn_hand = (self.player_button.draw_hand.dealt_cards),
                                    bet_val_vector = self.player_blind.bet_val_vector,
                                    act_val_vector = self.player_blind.act_val_vector,
                                    num_draw_vector = self.player_blind.num_draw_vector)
            self.hand_history.append(draw_action)
            
            draw_action = DrawAction(actor_name = self.player_button.name, pot_size = self.pot_size, 
                                     hand_before = self.player_button.draw_hand.dealt_cards, 
                                     best_draw = self.player_button.draw_hand.held_cards, 
                                     hand_after = self.player_button.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_button.draw_hand.dealt_cards,
                                    draws_left=3, 
                                    position = POSITION_BUTTON, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_button.heuristic_value,
                                    bet_this_hand = self.player_button.bet_this_hand,
                                    bet_model = self.player_button.player_tag(),
                                    oppn_hand = (self.player_blind.draw_hand.final_hand),                              
                                    bet_val_vector = self.player_button.bet_val_vector,
                                    act_val_vector = self.player_button.act_val_vector,
                                    num_draw_vector = self.player_button.num_draw_vector) # use hand we are already up against!
            self.hand_history.append(draw_action)

            # Adjust the current poker hand, after our draw
            draw_hand_blind = PokerHand()
            draw_hand_blind.deal(self.player_blind.draw_hand.final_hand)
            self.player_blind.draw_hand = draw_hand_blind
            draw_hand_button = PokerHand()
            draw_hand_button.deal(self.player_button.draw_hand.final_hand)
            self.player_button.draw_hand = draw_hand_button

        # Alternatively, if we are playing Holdem, deal the flop!
        if self.format == 'holdem':
            self.community.deal(deck=self.deck)
            print('\n-- Flop -- %s\n' % [hand_string(self.community.flop), hand_string(self.community.turn), hand_string(self.community.river)])

        # Next round. We bet again, (then draw again)
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0
        self.hand_history_this_round = []

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        #print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=2)                                                           
        self.player_button.update_hand_value(num_draws=2)

        self.play_betting_round(round = DRAW_1_BET_ROUND)
        
        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if self.format != 'holdem':
            print('\n-- 2nd draw --\n')
            print('Both players live. So continue betting after the 2nd draw.')

            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, num_draws=2, 
                                   has_button = False,
                                   pot_size=self.pot_size, 
                                   actions_this_round=self.hand_history_this_round, 
                                   actions_whole_hand=self.hand_history,
                                   cards_kept=self.player_blind.num_cards_kept, 
                                   opponent_cards_kept=self.player_button.num_cards_kept)
            self.player_button.draw(deck=self.deck, num_draws=2,
                                    has_button = True,
                                    pot_size=self.pot_size, 
                                    actions_this_round=self.hand_history_this_round, 
                                    actions_whole_hand=self.hand_history,
                                    cards_kept=self.player_button.num_cards_kept, 
                                    opponent_cards_kept=self.player_blind.num_cards_kept)

            draw_action = DrawAction(actor_name = self.player_blind.name, pot_size = self.pot_size, 
                                     hand_before = self.player_blind.draw_hand.dealt_cards, 
                                     best_draw = self.player_blind.draw_hand.held_cards, 
                                     hand_after = self.player_blind.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_blind.draw_hand.dealt_cards,
                                    draws_left=2, 
                                    position = POSITION_BLIND, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_blind.heuristic_value,
                                    bet_this_hand = self.player_blind.bet_this_hand,
                                    bet_model = self.player_blind.player_tag(),
                                    oppn_hand = (self.player_button.draw_hand.dealt_cards),
                                    bet_val_vector = self.player_blind.bet_val_vector,
                                    act_val_vector = self.player_blind.act_val_vector,
                                    num_draw_vector = self.player_blind.num_draw_vector)
            self.hand_history.append(draw_action)
            draw_action = DrawAction(actor_name = self.player_button.name, pot_size = self.pot_size, 
                                     hand_before = self.player_button.draw_hand.dealt_cards, 
                                     best_draw = self.player_button.draw_hand.held_cards, 
                                     hand_after = self.player_button.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_button.draw_hand.dealt_cards,
                                    draws_left=2, 
                                    position = POSITION_BUTTON, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_button.heuristic_value,
                                    bet_this_hand = self.player_button.bet_this_hand,
                                    bet_model = self.player_button.player_tag(),
                                    oppn_hand = (self.player_blind.draw_hand.final_hand),
                                    bet_val_vector = self.player_button.bet_val_vector,
                                    act_val_vector = self.player_button.act_val_vector,
                                    num_draw_vector = self.player_button.num_draw_vector) # use hand that we are already up against
            self.hand_history.append(draw_action)

            # Update the correct draw hand
            draw_hand_blind = PokerHand()
            draw_hand_blind.deal(self.player_blind.draw_hand.final_hand)
            self.player_blind.draw_hand = draw_hand_blind
            draw_hand_button = PokerHand()
            draw_hand_button.deal(self.player_button.draw_hand.final_hand)
            self.player_button.draw_hand = draw_hand_button

        # Alternatively, if we are playing Holdem, deal the turn!
        if self.format == 'holdem':
            self.community.deal(deck=self.deck)
            print('\n-- Turn -- %s\n' % [hand_string(self.community.flop), hand_string(self.community.turn), hand_string(self.community.river)])

        # Next round. We bet again, then draw again
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0
        self.hand_history_this_round = []

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        #print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=1)                                                           
        self.player_button.update_hand_value(num_draws=1)

        self.play_betting_round(round = DRAW_2_BET_ROUND)
        
        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if self.format != 'holdem':
            print('\n-- 3rd draw --\n')
            print('Both players live. So continue betting after the 3rd draw.')
            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, num_draws=1, 
                                   has_button = False,
                                   pot_size=self.pot_size, 
                                   actions_this_round=self.hand_history_this_round, 
                                   actions_whole_hand=self.hand_history,
                                   cards_kept=self.player_blind.num_cards_kept, 
                                   opponent_cards_kept=self.player_button.num_cards_kept)
            self.player_button.draw(deck=self.deck, num_draws=1,
                                    has_button = True,
                                    pot_size=self.pot_size, 
                                    actions_this_round=self.hand_history_this_round, 
                                    actions_whole_hand=self.hand_history,
                                    cards_kept=self.player_button.num_cards_kept, 
                                    opponent_cards_kept=self.player_blind.num_cards_kept)
            
            draw_action = DrawAction(actor_name = self.player_blind.name, pot_size = self.pot_size, 
                                     hand_before = self.player_blind.draw_hand.dealt_cards, 
                                     best_draw = self.player_blind.draw_hand.held_cards, 
                                     hand_after = self.player_blind.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_blind.draw_hand.dealt_cards,
                                    draws_left=1, 
                                    position = POSITION_BLIND, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_blind.heuristic_value,
                                    bet_this_hand = self.player_blind.bet_this_hand,
                                    bet_model = self.player_blind.player_tag(),
                                    oppn_hand = (self.player_button.draw_hand.dealt_cards),
                                    bet_val_vector = self.player_blind.bet_val_vector,
                                    act_val_vector = self.player_blind.act_val_vector,
                                    num_draw_vector = self.player_blind.num_draw_vector)
            self.hand_history.append(draw_action)
            draw_action = DrawAction(actor_name = self.player_button.name, pot_size = self.pot_size, 
                                     hand_before = self.player_button.draw_hand.dealt_cards, 
                                     best_draw = self.player_button.draw_hand.held_cards, 
                                     hand_after = self.player_button.draw_hand.final_hand)
            draw_action.add_context(hand=self.player_button.draw_hand.dealt_cards,
                                    draws_left=1, 
                                    position = POSITION_BUTTON, 
                                    actions_this_round = self.hand_history_this_round,
                                    actions_full_hand = self.hand_history,
                                    value = self.player_button.heuristic_value,
                                    bet_this_hand = self.player_button.bet_this_hand,
                                    bet_model = self.player_button.player_tag(),
                                    oppn_hand = (self.player_blind.draw_hand.final_hand),
                                    bet_val_vector = self.player_button.bet_val_vector,
                                    act_val_vector = self.player_button.act_val_vector,
                                    num_draw_vector = self.player_button.num_draw_vector) # last draw on last hand = against opponent's final hand!
            self.hand_history.append(draw_action)

        # Alternatively, if we are playing Holdem, deal the river!
        if self.format == 'holdem':
            self.community.deal(deck=self.deck)
            print('\n-- Rivr -- %s\n' % [hand_string(self.community.flop), hand_string(self.community.turn), hand_string(self.community.river)])

        # Next round. We bet again, then draw again
        self.live = True 
        self.action_on = self.player_blind
        self.action_off = self.player_button

        # TODO: function, to prepare betting round, reset intermediate values.
        self.player_blind.bet_this_street = 0.0
        self.player_button.bet_this_street = 0.0
        self.hand_history_this_round = []

        # Now, query the CNN, to find out current value of each hand.
        # NOTE: We re-run the computation a minute later for "best draw..." but that's fine. Redundancy is ok.
        #print('--> compute player heuristics')
        self.player_blind.update_hand_value(num_draws=0)                                                           
        self.player_button.update_hand_value(num_draws=0)

        self.play_betting_round(round = DRAW_3_BET_ROUND)
        
        # print(self.hand_history)

        print('Made it all the way, with betting on the river')


    # Declare a winner... assuming hand ends now.
    def get_hand_result(self, cashier):
        winners = {self.player_button.name: 0.0, self.player_blind.name: 0.0}
        if self.player_blind.live and not self.player_button.live:
            print('\nPlayer F wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_blind.draw_hand))
            winners[self.player_blind.name] = self.pot_size
        elif not self.player_blind.live and self.player_button.live:
            print('\nPlayer B wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_button.draw_hand))
            winners[self.player_button.name] = self.pot_size
        elif not self.player_blind.live and not self.player_button.live:
            print('Error! both players are dead.')
        else:
            # TODO: Handle ties & split pots!
            best_hand = cashier.showdown([self.player_blind.draw_hand, self.player_button.draw_hand])
            if best_hand == self.player_blind.draw_hand:
                print('\nPlayer F wins on showdown! %d chips in the pot.' % self.pot_size)
                winners[self.player_blind.name] = self.pot_size
            elif best_hand == self.player_button.draw_hand:
                print('\nPlayer B wins on showdown! %d chips in the pot.' % self.pot_size)
                winners[self.player_button.name] = self.pot_size
            else:
                winners[self.player_blind.name] = self.pot_size / 2.0
                winners[self.player_button.name] = self.pot_size / 2.0
                print('Tie! or Error. %d chips in the pot.' % self.pot_size)

            if best_hand:
                print(str(best_hand))

        return winners

