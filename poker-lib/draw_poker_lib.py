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

from draw_poker import cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Methods, useful for organizing & evaluating a draw poker match.
"""

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
        # As a hack... output hand with best (2-7) rank. Ties go to best position...
        best_rank = 0
        best_hand = None
        for hand in hands:
            hand.evaluate()
            if hand.rank > best_rank:
                best_hand = hand
                best_rank = hand.rank
            elif hand.rank == best_rank:
                print('Need to implement ties & splits!')
                raise NotImplementedError()
        return best_hand

# At risk of over-engineering, use a super-class to encapsulate each type
# of poker action. Specifically, betting, checking, raising.
# NOTE: Usually, is prompted, decides on an action, and passes control.
# (but not always, as there are forced bets, etc)
# This system strong assumes, that actions take place in order, by a dealer. Poker is a turn-based game.
class PokerAction:
    # TODO: Copy of the game state (hands, etc)? Zip of the information going into this action?
    def __init__(self, action_type, actor_name, pot_size, bet_size):
        self.action_type = action_type
        self.action_name = actionName[action_type]

        # Use externally, if we choose to set it. Convenient/hack way to use array of actions, then apply probability policy.
        self.probability = 0.0

        # For now... just a hack to distinguish between B = button and F = blind player, first to act
        self.actor_name = actor_name
        self.pot_size = pot_size # before the action
        self.bet_size = bet_size # if applicable
        
    # Consise summary, of the action taken.
    def __str__(self):
        raise NotImplementedError()

# Simple encoding, for each possible action
class PostBigBlind(PokerAction):
    def __init__(self, actor_name, pot_size):
        PokerAction.__init__(self, action_type = POST_BIG_BLIND, actor_name = actor_name, pot_size = pot_size, bet_size = BIG_BLIND_SIZE)

class PostSmallBlind(PokerAction):
    def __init__(self, actor_name, pot_size):
        PokerAction.__init__(self, action_type = POST_BIG_BLIND, actor_name = actor_name, pot_size = pot_size, bet_size = SMALL_BLIND_SIZE)

class CheckStreet(PokerAction):
    def __init__(self, actor_name, pot_size):
        PokerAction.__init__(self, action_type = CHECK_HAND, actor_name = actor_name, pot_size = pot_size, bet_size = 0)

class FoldStreet(PokerAction):
    def __init__(self, actor_name, pot_size):
        PokerAction.__init__(self, action_type = FOLD_HAND, actor_name = actor_name, pot_size = pot_size, bet_size = 0)

# Cost of the other actions... to be computed, from $ spent by player on this street, already.
# NOTE: Think of it like internet, with chips left in front of players... until betting round is finished.
class CallSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = CALL_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

class CallBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = CALL_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

class BetSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street + SMALL_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = BET_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

class BetBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street + BIG_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = BET_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

class RaiseSmallStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street + SMALL_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = RAISE_SMALL_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

class RaiseBigStreet(PokerAction):
    def __init__(self, actor_name, pot_size, player_bet_this_street, biggest_bet_this_street):
        total_bet_size = biggest_bet_this_street + BIG_BET_SIZE;
        this_bet = total_bet_size - player_bet_this_street;
        assert this_bet > 0, 'error calling %s' % self.__name__
        PokerAction.__init__(self, action_type = RAISE_BIG_STREET, actor_name = actor_name, pot_size = pot_size, bet_size = this_bet)

# TODO: Implement 3-betting and 4-betting...
