import sys
import logging
import math
import re
import random
import numpy as np
import scipy.stats as ss
import itertools
from poker_hashes import *
from poker_util import *
from poker_lib import *

"""
An extension of poker_lib for Hold'em games: Texas Hold'em, and Omaha Hold'em.

Anything that is useful for a more general poker sense belongs in poker_lib.
Anything that's a util, not based on Card or Deck class... belongs in poker_util.
"""


##########################
# Values for Hold'em games.

# "Rounds" in the game, equivalent to 3,2,1 draws left, similar to draw games.
PREFLOP_ROUND = 1
FLOP_ROUND = 2
TURN_ROUND = 3
RIVER_ROUND = 4
holdemRoundsLeft = {PREFLOP_ROUND:3, FLOP_ROUND: 2, TURN_ROUND: 1, RIVER_ROUND:0}
HOLDEM_ROUNDS_SET = set([PREFLOP_ROUND, FLOP_ROUND, TURN_ROUND, RIVER_ROUND])

# Holdem equivalent for DRAW_VALUE_KEYS. [0.0, 1.0] values trained for simulated holdem hands below.
# TODO: Make this an easier lookup, etc.
HOLDEM_VALUE_KEYS = ['best_value'] + [categoryName[category] for category in HIGH_HAND_CATEGORIES]

# Evaluate a 2-card hold'em hand, with 3+ community cards
# NOTE: Can use 0-2 from dealt_cards and the rest from community.
# TODO: Add game type (or new function) to support Omaha when we get there (needs two cards exclusively)
# NOTE: Will raise an exception if not enough cards.
exclude_1_choose_6 = [set([0]), set([1]), set([2]), set([3]), set([4]), set([5])]
exclude_2_choose_7 = [set([0,1]), set([0,2]), set([0,3]), set([0,4]), set([0,5]), set([0,6]), 
                      set([1,2]), set([1,3]), set([1,4]), set([1,5]), set([1,6]),
                      set([2,3]), set([2,4]), set([2,5]), set([2,6]),
                      set([3,4]), set([3,5]), set([3,6]),
                      set([4,5]), set([4,6]),
                      set([5,6])]
def hand_rank_community_cards(dealt_cards, community_cards):
    assert len(dealt_cards) == 2, 'Need holdem hand for eval. Given %s' % dealt_cards
    all_cards = dealt_cards + community_cards
    #print('given %d cards to evaluate: %s' % (len(all_cards), [hand_string(dealt_cards), hand_string(community_cards)]))

    # For now, can support only 5-7 cards in hand
    if len(all_cards) < 5 or len(all_cards) > 7:
        print('Illegal number of cards for evaluation! %s' % hand_string(all_cards))
        return
    elif len(all_cards) == 5:
        return hand_rank_five_card(all_cards)
    elif len(all_cards) == 6:
        exclude_array = exclude_1_choose_6
    elif len(all_cards) == 7:
        exclude_array = exclude_2_choose_7

    # Try every draw, and return the best rank
    best_rank = -1
    for exclude_index in exclude_array:
        exclude_cards = []
        include_cards = []
        for index in range(len(all_cards)):
            card = all_cards[index]
            if index in exclude_index:
                exclude_cards.append(card)
            else:
                include_cards.append(card)
        #print('trying hand %s with excluded %s' % (hand_string(include_cards), hand_string(exclude_cards)))
        rank = hand_rank_five_card(include_cards)
        if best_rank < 0 or rank < best_rank:
            #print('best hand!')
            best_rank = rank
    return best_rank


# Community cards. Not part of the deck. But also not really a hand. 
# NOTE: We can hard-wire a hand having flop, turn and river.
class HoldemCommunityHand(object):
    def __init__(self, flop = [], turn = [], river = []):
        self.flop = flop
        self.turn = turn
        self.river = river
        self.round = None
        self.update_round()

    # Update current hand round, or throw error if hand bug.
    def update_round(self):
        if not (self.flop) and not (self.turn) and not (self.river):
            self.round = PREFLOP_ROUND
        elif len(self.flop) == 3 and not (self.turn) and not (self.river):
            self.round = FLOP_ROUND
        elif len(self.flop) == 3 and len(self.turn) == 1 and not (self.river):
            self.round = TURN_ROUND
        elif len(self.flop) == 3 and len(self.turn) == 1 and len(self.river) == 1:
            self.round = RIVER_ROUND
        else:
            assert False, 'Illegal collection of flop, turn, river %s' % [self.flop, self.turn, self.river]
        return self.round

    # All community cards
    def cards(self):
        return self.flop + self.turn + self.river

    # Deal the next round (if any)
    # runway = True to finish dealing hand to the end.
    def deal(self, deck, runway = False):
        self.update_round()
        if self.round == PREFLOP_ROUND:
            cards = deck.deal(3)
            self.flop = cards
            self.update_round()
            #print('dealt flop %s' % hand_string(self.flop))
        elif self.round == FLOP_ROUND:
            cards = deck.deal(1)
            self.turn = cards
            self.update_round()
            #print('dealt turn %s' % hand_string(self.turn))
        elif self.round == TURN_ROUND:
            cards = deck.deal(1)
            self.river = cards
            self.update_round()
            #print('dealt river %s' % hand_string(self.river))
        else:
            return

        # Keep going, if we aren't yet at the river, and runway requested
        if runway:
            self.deal(deck, runway)


    # Return cards to the deck, as part of misdeal or simulation.
    def undeal(self, deck):
        self.update_round()
        if self.round == PREFLOP_ROUND:
            return
        elif self.round == FLOP_ROUND:
            cards = self.flop
            self.flop = []
        elif self.round == TURN_ROUND:
            cards = self.turn
            self.turn = []
        elif self.round == RIVER_ROUND:
            cards = self.river
            self.river = []
        #print('rewinding round & returning %s' % hand_string(cards))
        deck.return_cards(cards, shuffle=False)
        self.update_round()

    # Return cards to the deck, until we reach desired round, or at the beginning of the hand
    def rewind(self, deck, round=PREFLOP_ROUND):
        while self.round != PREFLOP_ROUND and self.round > round:
            self.undeal(deck)
        
    def __str__(self):
        return '%s%s%s' % (hand_string(self.flop), hand_string(self.turn), hand_string(self.river))


# A hand for Hold'em. Replicates some of the DrawHand methods... but not all. And try to keep it simpler.
class HoldemHand(object):
    def __init__(self, cards = None, community = None):
        self.dealt_cards = []
        self.rank = -1
        self.category = -1
        self.category_name = 'none'
        if cards:
            self.deal(cards)
        if community:
            self.community = community
        else:
            self.community = None

    # cards from the deck.
    # NOTE: Should be two of them in Texas Holdem.... but can be four in Omaha
    def deal(self, cards):
        # TODO: Remove or change when we get to Omaha
        assert len(cards) == 2
        assert not self.dealt_cards 
        for card in cards:
            self.dealt_cards.append(card)

        # For backward compatibility...
        self.final_hand = self.dealt_cards

    # Look up with hash tables. Can only evaluate if community cards present
    def evaluate(self):
        if not self.community:
            print('Can not evaluate Holdem hand without community cards!')
            return

        self.rank = hand_rank_community_cards(self.dealt_cards, self.community.cards())
        self.category = hand_category(self.rank)
        self.category_name = categoryName[self.category]


    def __str__(self):
        return 'Holdem hand: %s Community %s (rank: %d, category: %s)' % (hand_string(self.dealt_cards),
                                                                          self.community,
                                                                          self.rank, self.category_name)
