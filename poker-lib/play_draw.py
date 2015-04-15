# TODO: remove un-used chess components

import sys
import csv
import logging
import math
import re
import random
import numpy as np
import scipy.stats as ss
from poker_lib import *
from poker_util import *

"""
#import train
import pickle
import theano
import theano.tensor as T
#import math
import chess, chess.pgn
#from parse_game import bb2array
import heapq
import time
#import re
import string
#import numpy
import sunfish
import pickle
#import random
import traceback
"""

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing basic video poker, and training/generating trianing data, for a neural network AI to learn how to play such games.

Hat tip to Erik of DeepPink, which implements a similar system for chess.
https://github.com/erikbern/deep-pink

Basic components:
- Card, Deck, PokerHand classes
- shuffle, deal, draw operations
- evaulate a poker hand [imports C-lang methods]
- assign rewards to hands

I/O components:
- dummy models (linear, random, rules based, etc)
- pickle and un-pickle theano neural-net model
- save training data [as a hand history]

AI training:
- TBD
""" 

POKER_GAME_HEADER = ['dealt_cards', 'held_cards', 'discards', 'draw_cards', 'final_hand', 'rank', 'category', 'category_name', 'reward']

# Interface for AI to select moves
# NOTE: move(hand) operates directly on hand, and makes draw.
# NOTE: For more complex card game, will take in full state, not just current hand
class CardPlayer(object):
    def move(self, hand, deck):
        raise NotImplementedError()

# Never does anything.
class DeadPlayer(CardPlayer):
    # Takes direct action on the hand.
    def move(self, hand, deck):
        logging.debug('No action, on hand %s' % hand)
        hand.draw('')

# Draws random cards.
class RandomPlayer(CardPlayer):
    # Takes direct action on the hand.
    def move(self, hand, deck):
        draw_string = ''
        for i in range(0,5):
            if random.random() > 0.50:
                draw_string += '%d' % i

        discards = hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        hand.deal(new_cards, final_hand=True)
        

# Human-input player.
class ManualPlayer(CardPlayer):
    # Takes direct action on the hand.
    # Requires user to enter string like '034' or '21' for card positions to discard
    def move(self, hand, deck):
        # TODO
        raise NotImplementedError()

# Uses logic to read the hand, pre-process for structures like 4-flush, pairs, etc.
# Then uses fixed, hardcoded rules, to make a decision.
class RuleBasedPlayer(CardPlayer):
    # Use as dump for hand-specific functions, values, to pass as features
    def pre_process_hand(self, hand, deck):
        # Useful side ways to store the hand

        # Evaluate held cards (pre-draw hand)
        self.rank = hand_rank_five_card(self.held_cards)
        self.category = hand_category(self.rank)
        self.category_name = categoryName[self.category]
        
        # Do we have royal flush draw?

    # Complete the move, if we know what to discard.
    def move_with_discards(self, hand, deck, discards):
        raise NotImplementedError()

    # Complete the move, if we know what to keep.
    def move_with_held(self, hand, deck, held_cards):
        raise NotImplementedError()

    def move(self, hand, deck):
        raise NotImplementedError()

# Implements the rule table outlined here:
# http://wizardofodds.com/games/video-poker/strategy/jacks-or-better/9-6/intermediate/
class WizardJacksPlayer(RuleBasedPlayer):
    def move(self, hand, deck):
         self.pre_process_hand(self, hand, deck)

         # Now, go through the various cases. Long if-then. 25 cases.
         # Needs to know:
         # -- hand category
         # -- best royal flush draw
         # -- best straight flush draw
         # -- straight draw, etc


# Play a round of poker
# A. Create a deck, shuffle & deal hand.
# B. Pass game info to AI, get response, draw
# F. Evaluate final hand
# G. Return result
def game(round):
    print '\n-- New Round %d --\n' % round

    deck = PokerDeck(shuffle=True)
    #print deck
    draw_hand = PokerHand()
    #print draw_hand
    deal_cards = deck.deal(5)
    #print deal_cards
    draw_hand.deal(deal_cards)
    #print draw_hand

    # Now hand is dealt. Use an agent to choose what do discard.
    #player = DeadPlayer() # asleep at the wheel
    player = RandomPlayer() # random draw
    player.move(hand=draw_hand, deck=deck)

    print draw_hand

    # Evaluate the hand. This means returning a number corresponding to hand order, and category.
    draw_hand.evaluate()

    # Compute rewards with payout table
    cashier = JacksOrBetter() # "976-9-6" Jacks or Better -- with 100% long-term payout.
    pay_him = cashier.payout(draw_hand)
    draw_hand.reward = pay_him
    
    print 'pay the man his $%d' % pay_him

    return (draw_hand, pay_him)

# Saves to rows, matching supplied header. Currently:
# ['dealt_cards', 'held_cards', 'discards', 'draw_cards', 'final_hand', 'rank', 'category', 'category_name', 'reward']
def output_hand_csv(poker_hand, header_map):
    output_map = {}
    if poker_hand.dealt_cards:
        output_map['dealt_cards'] = ''.join([str(card) for card in poker_hand.dealt_cards])
    if poker_hand.held_cards:
        output_map['held_cards'] = ''.join([str(card) for card in poker_hand.held_cards])
    if poker_hand.discards:
        output_map['discards'] = ''.join([str(card) for card in poker_hand.discards])
    if poker_hand.draw_cards:
        output_map['draw_cards'] = ''.join([str(card) for card in poker_hand.draw_cards])
    if poker_hand.final_hand:
        output_map['final_hand'] = ''.join([str(card) for card in poker_hand.final_hand])
    output_map['rank'] = poker_hand.rank
    output_map['category'] = poker_hand.category
    output_map['category_name'] = poker_hand.category_name
    output_map['reward'] = poker_hand.reward

    output_row = VectorFromKeysAndSparseMap(keys=header_map, sparse_data_map=output_map, default_value = '')
    return output_row

# Play a bunch of hands, and record results for training
# For now... just takes random actions.
def play(sample_size, output_file_name):
    #func = get_model_from_pickle('model.pickle')
    round = 0
    results = []
    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(POKER_GAME_HEADER)
        csv_header_map = CreateMapFromCSVKey(POKER_GAME_HEADER)
    else:
        csv_writer = None

    while round < sample_size:
        hand, payout = game(round)
        results.append(payout)

        # Save hand to CSV, if output supplied.
        if csv_writer:
            hand_csv_row = output_hand_csv(poker_hand=hand, header_map=csv_header_map)
            csv_writer.writerow(hand_csv_row)

            # Hack, to show matrix for final hand.
            #print hand_to_matrix(hand.final_hand)
            pretty_print_hand_matrix(hand.final_hand)

        round += 1
        #sys.exit(-3)

    if csv_writer:
        print '\nwrote %d rows' % len(results)
        output_file.close()

    print '\npaid %s' % sorted(results, reverse=True)[0:200]
    print '\nstats:\tave: %.2f\tstdev: %.2f\tskew: %.2f' % (np.mean(results), np.std(results), ss.skew(results))
        
if __name__ == '__main__':
    samples = 10000
    output_file_name = '%d_samples.csv' % samples
    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name)


