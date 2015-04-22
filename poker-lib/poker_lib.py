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

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Python implementation for draw poker hand processing
"""

# Run more trials for royal flush draw...
# Why? Get better ground truth -- in all cases. Either under-sample or over-sample, for rare, high-value events.
ROYAL_DRAW_MULTIPLIER = 10

# definitions for poker hands
CLUB	= 0x8000
DIAMOND = 0x4000
HEART	= 0x2000
SPADE	= 0x1000
suitsArray = [CLUB, DIAMOND, HEART, SPADE]
#print 'suits: %s' % suitsArray

# Sometimes we need [CLUB, DIAMOND, HEART, SPADE] -> [0,1,2,3]
suits_to_matrix = {CLUB: 0, DIAMOND: 1, HEART: 2, SPADE: 3}

Deuce	= 0
Trey	= 1
Four	= 2
Five	= 3
Six     = 4
Seven	= 5
Eight	= 6
Nine	= 7
Ten     = 8
Jack	= 9
Queen	= 10
King	= 11
Ace     = 12
ranksArray = [Deuce, Trey, Four, Five, Six, Seven, Eight, Nine, Ten, Jack, Queen, King, Ace]
#print 'ranks: %s' % ranksArray
royalRanksSet = set([ Ten, Jack, Queen, King, Ace ])

ROYAL_FLUSH     = 10
STRAIGHT_FLUSH	= 1
FOUR_OF_A_KIND	= 2
FULL_HOUSE	= 3
FLUSH		= 4
STRAIGHT	= 5
THREE_OF_A_KIND	= 6
TWO_PAIR	= 7
JACKS_OR_BETTER = 77 # useful category, for Jacks-or-better payout structure
ONE_PAIR	= 8
HIGH_CARD	= 9

categoryName = {ROYAL_FLUSH: 'royal', 
                STRAIGHT_FLUSH: 'straight flush',
                FOUR_OF_A_KIND: 'quads',
                FULL_HOUSE: 'house',
                FLUSH: 'flush',
                STRAIGHT: 'straight',
                THREE_OF_A_KIND: 'trips',
                TWO_PAIR: 'two pair',
                JACKS_OR_BETTER: 'jacks+',
                ONE_PAIR: 'pair',
                HIGH_CARD: 'hi card'}

#///////////////////////////////////////////////////////
#    /*
#     ** each of the thirteen card ranks has its own prime number
#     **
#     ** deuce = 2
#     ** trey  = 3
#     ** four  = 5
#     ** five  = 7
#     ** ...
#     ** king  = 37
#     ** ace   = 41
#     */
#    int primes[] = { 2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41 };
#    ///////////////////////////////////////////////////////
#    
#    return primes[n];
hashPrimes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41]
def hash_prime(value):
    return hashPrimes[value]
    
suitName = {CLUB: 'Club', SPADE: 'Spade', HEART: 'Heart', DIAMOND: 'Diamond'}
suitSymbol = {CLUB: 'c', SPADE: 's', HEART: 'h', DIAMOND: 'd'}
valueName = {Deuce: 'Deuce', Trey: 'Trey', Four: 'Four', Five: 'Five', Six: 'Six', Seven: 'Seven', Eight: 'Eight', Nine: 'Nine', Ten: 'Ten', Jack: 'Jack', Queen: 'Queen', King: 'King', Ace: 'Ace'}
valueSymbol = {Deuce: '2', Trey: '3', Four: '4', Five: '5', Six: '6', Seven: '7', Eight: '8', Nine: '9', Ten: 'T', Jack: 'J', Queen: 'Q', King: 'K', Ace: 'A'}

# Easy reverse lookup from string
suitFromChar = {'c': CLUB, 's': SPADE, 'h':HEART,'d':DIAMOND}
valueFromChar = {'2': Deuce, '3': Trey, '4': Four, '5':Five, '6':Six, '7':Seven, '8':Eight, '9':Nine, 'T':Ten, 'J':Jack, 'Q':Queen, 'K':King, 'A':Ace}
       

###########################################################
# For estimating the "value" of each hand, it can be useful to try every possible draw variant.
# Five-card hands include 32 draws (1x keep none, 5x keep one, 10x keep two, 10x keep three, 5x keep four, 1x keep five)
# Since this is an invariant of the game, like ranks and suits, encode the possibilities into a fixed array. 
all_draw_patterns = [ set([]),
                      set([0]), set([1]), set([2]), set([3]), set([4]),
                      set([0,1]), set([0,2]), set([0,3]), set([0,4]), set([1,2]), set([1,3]), set([1,4]), set([2,3]), set([2,4]), set([3,4]), 
                      set([0,1,2]), set([0,1,3]), set([0,1,4]), set([0,2,3]), set([0,2,4]), set([0,3,4]), set([1,2,3]), set([1,2,4]), set([1,3,4]), set([2,3,4]),
                      set([0,1,2,3]), set([0,1,2,4]), set([0,1,3,4]), set([0,2,3,4]), set([1,2,3,4]),
                      set([0,1,2,3,4]) ]

print all_draw_patterns


# There are X ways to scramble the suits in a hand. Note that any mapping still results in same output.
# Scramble is always from [CLUB, DIAMOND, HEART, SPADE]
all_suit_scrambles_array = itertools.permutations([CLUB, DIAMOND, HEART, SPADE])
all_suit_scrambles_maps = [{CLUB: permutation[0], DIAMOND: permutation[1], HEART: permutation[2], SPADE: permutation[3]} for permutation in all_suit_scrambles_array]

print all_suit_scrambles_maps


# stuff for simulation, 32 categories of output (above)
# All the data to store, from a hand simulation
POKER_FULL_SIM_HEADER = ['hand', 'best_value', 'best_draw', 'sample_size', 'pay_scheme']
for draw_pattern in all_draw_patterns:
    draw_to_string = '[%s]' % ','.join([str(i) for i in list(draw_pattern)])
    POKER_FULL_SIM_HEADER.append('%s_value' % draw_to_string)
    POKER_FULL_SIM_HEADER.append('%s_draw' % draw_to_string)

# 32 keys corresponding to output
DRAW_VALUE_KEYS = []
for draw_pattern in all_draw_patterns:
    draw_to_string = '[%s]' % ','.join([str(i) for i in list(draw_pattern)])
    DRAW_VALUE_KEYS.append('%s_value' % draw_to_string)

print(['%d: %s' % (i, all_draw_patterns[i]) for i in range(len(all_draw_patterns))])

#///////////////////////////////////////////////////////////////////////////////
#//   Card object encoding into an integer. This is #standard for poker evaluation.
#//   since it's easy to compute hand ranks, and exact hand values (suit invariant)
#//   with just a few bit shifts, and an array lookup.
#//
#//   An integer is made up of four bytes.  The high-order
#//   bytes are used to hold the rank bit pattern, whereas
#//   the low-order bytes hold the suit/rank/prime value
#//   of the card.
#//
#//   +--------+--------+--------+--------+
#//   |xxxbbbbb|bbbbbbbb|cdhsrrrr|xxpppppp|
#//   +--------+--------+--------+--------+
#//
#//   p = prime number of rank (deuce=2,trey=3,four=5,five=7,...,ace=41)
#//   r = rank of card (deuce=0,trey=1,four=2,five=3,...,ace=12)
#//   cdhs = suit of card
#//   b = bit turned on depending on rank of card
# self.hashTag = [PokerHashes prime:self.value] | (self.value << 8) | self.suit | (1 << (16+self.value));
#///////////////////////////////////////////////////////////////////////////////
def card_hash_tag(suit, value):
    #print 'Computing hash tag for suit %d, value %d %s-%s' % ( suit, value, valueName[value], suitSymbol[suit])
    hash_tag =  (hash_prime(value)) | (value << 8) | suit | (1 << (16+value))
    #print hash_tag
    #print format(hash_tag, '#032b')
    return hash_tag

# A bit more involved than you think. Encodes the card in C-style, for lib lookup, hand evaluation, etc.
# NOTE: Re-uses naming conventions from C-code for convenience.
class Card(object):
    def __init__(self, suit, value):
        self.suit = suit
        self.value = value
        # self.tag = ## does not appear to be used in game logic?

        # 32-bit encoding
        self.hashTag = card_hash_tag(suit, value)

    # Other TODOs

    def __str__(self):
        return '%s%s' % (valueSymbol[self.value], suitSymbol[self.suit])

    def __eq__(self, other):
        return (self.suit == other.suit and self.value == other.value)

# card from string Ks
def card_from_string(card_str):
    return Card(suit=suitFromChar[card_str[1]], value=valueFromChar[card_str[0]])

# Return array of equivalent hands (same order, different suits)
# NOTE: array of cards, not strings.
def hand_suit_scrambles(hand_array):
    # to-string of hands seen
    unique_strings = set([])
    uniques = [] # output of card arrays
    
    for suit_scramble in all_suit_scrambles_maps:
        new_hand = [Card(suit=suit_scramble[card.suit], value=card.value) for card in hand_array]
        new_hand_string = hand_string(new_hand)
        if not(new_hand_string in unique_strings):
            uniques.append(new_hand)
            unique_strings.add(new_hand_string)

    #print('for hand %s, found %d unique hands equivalent by suit permutation' % (hand_string(hand_array), len(uniques)))
    #print unique_strings

    return uniques


# Type of hand, from raw rank.
PAIR_JACKS_MIN_RANK = 4205
def hand_category(val):
    if (val > 6185): return(HIGH_CARD);        # 1277 high card
    if (val > 4205): return(ONE_PAIR);         # TODO: Test! Confirm!! Jacks or better!!
    if (val > 3325): return(JACKS_OR_BETTER);  # 2860 one pair
    if (val > 2467): return(TWO_PAIR);         #  858 two pair
    if (val > 1609): return(THREE_OF_A_KIND);  #  858 three-kind
    if (val > 1599): return(STRAIGHT);         #   10 straights
    if (val > 322):  return(FLUSH);            # 1277 flushes
    if (val > 166):  return(FULL_HOUSE);       #  156 full house
    if (val > 10):   return(FOUR_OF_A_KIND);   #  156 four-kind
    if (val == 1):   return(ROYAL_FLUSH);
    return(STRAIGHT_FLUSH);                   #   10 straight-flushes

# Binary search, on products array. Not sure why no better way... but as long as it works.
def  hard_findit(key):
    low = 0
    high = 4887
    mid = 0
    
    while ( low <= high ):
        mid = (high+low) / 2;      # divide by two
        if ( key < products[mid] ):
            high = mid - 1;
        elif ( key > products[mid] ):
            low = mid + 1;
        else:
            return( mid );

    print "ERROR:  no match found; key = %d" % key
    return( -1 );



# Takes 5-card hand array as input
def hand_rank_five_card(hand):
    c0 = hand[0].hashTag
    c1 = hand[1].hashTag
    c2 = hand[2].hashTag
    c3 = hand[3].hashTag
    c4 = hand[4].hashTag

    # Unique hand hash -- flushes excluded.
    q = (c0|c1|c2|c3|c4) >> 16

    #print 'trying easy product q = %d' % q
    #print format(q, '#032b')

    # check for Flushes and StraightFlushes
    if ( c0 & c1 & c2 & c3 & c4 & 0xF000 ):
        #print 'looking in flushes'
        return flushes[q]
   
    # check for Straights and HighCard hands
    s = unique5[q]
    if s:
        #print 'found in unique5'
        return s

    # let's do it the hard way -- find hand in the sorted array, with binary search.
    #print 'do it the hard way'
    q_hard = (c0 & 0xFF) * (c1 & 0xFF) * (c2 & 0xFF) * (c3 & 0xFF) * (c4 & 0xFF);
    #print 'product: %d' % q_hard
    q_findit = hard_findit(q_hard)
    #print 'found q_findit: %d' % q_findit
    return values[q_findit]

# Helper function to turn a poker hand (array of cards) into 2D array.
# if pad_to_fit... pass along to card input creator, to create 14x14 array instead of 4x13
# NOTE: Try 17x17 padding!
def hand_to_matrix(poker_hand, pad_to_fit=False, pad_size=17):
    # initialize empty 4x13 matrix
    # Unless pad to fit... in which case pad to 18x18 # 14x14
    if pad_to_fit:
        matrix = np.array([[0 for x in range(pad_size)] for x in range(pad_size)], np.int32)
    else:
        matrix = np.array([[0 for x in range(len(ranksArray))] for x in range(len(suitsArray))], np.int32)
    for card in poker_hand:
        #print card
        #print ([suits_to_matrix[card.suit]], [card.value])
        if pad_to_fit:
            if pad_size == 17:
                # add 5 empty rows to start, and 5 empty rows to finish
                suit_offset = 6
                # add empty column to start 
                value_offset = 2
            elif pad_size == 15:
                suit_offset = 5
                value_offset = 1
        else:
            suit_offset = 0
            value_offset = 0
        matrix[suits_to_matrix[card.suit] + suit_offset][card.value + value_offset] = 1
        
    return matrix

# And just for a single card
def card_to_matrix(card, pad_to_fit=False):
    return hand_to_matrix([card], pad_to_fit=pad_to_fit)

# really just str() of hand matrix
def pretty_print_hand_matrix(poker_hand):
    matrix = hand_to_matrix(poker_hand)

    header = 'x' + ''.join([valueSymbol[v] for v in ranksArray])
    print header
    for suit in suitsArray:
        row = suitSymbol[suit] + ''.join([(str(c) if c  else '.') for c in matrix[suits_to_matrix[suit]]])
        print row
    

# Interface for payout table. Just a wrapper around a table.
class PayoutTable(object):
    def payout(self, hand):
        raise NotImplementedError()

# "976-9-6" Jacks or Better -- with 100% long-term payout.
# As described here: 
"""
HAND		PAYOFF	COMBINATIONS		PROBABILITY	RETURN
Royal flush	976	554,637,108		0.000028	0.027157
Straight flush	50	2,218,146,252		0.000111	0.005564
Four of a kind	25	46,944,834,792		0.002355	0.058878
Full house	9	228,938,634,648		0.011485	0.103367
Flush	        6	221,687,899,200		0.011122	0.066729
Straight	4	225,414,163,536		0.011308	0.045234
Three of a Kind	3	1,478,350,617,468	0.074165	0.222495
Two pair	2	2,569,778,781,732	0.128919	0.257839
Jacks or better	1	4,240,539,117,972	0.212737	0.212737
Nothing	        0	10,918,803,684,492	0.547769	0.000000
Total		        19,933,230,517,200	1.000000	1.000000
"""
jacks_or_better_table_976_9_6 = {ROYAL_FLUSH: 976, 
                                 STRAIGHT_FLUSH: 50,
                                 FOUR_OF_A_KIND: 25,
                                 FULL_HOUSE: 9,
                                 FLUSH: 6,
                                 STRAIGHT: 4,
                                 THREE_OF_A_KIND: 3,
                                 TWO_PAIR: 2,
                                 JACKS_OR_BETTER: 1,
                                 ONE_PAIR: 0,
                                 HIGH_CARD: 0}
class JacksOrBetter(PayoutTable):
    def __init__(self):
        self.payout_table = jacks_or_better_table_976_9_6

    # Takes in PokerHand object, evaluates on final hand...
    def payout(self, hand):
        if hand.rank < 0:
            hand.evaluate()

        # Return, from payout table.
        return self.payout_table[hand.category]

    # Just takes in rank.
    def payout_rank(self, rank):
        category = hand_category(rank)
        return self.payout_table[category]

# Is there 1+ cards in hand, leading to royal flush draw?
def is_royal_flush_draw(cards):
    if not cards:
        return False
    suit = 0
    # cycle through cards: break if wrong suit, or non-royal flush card
    for card in cards:
        if not suit:
            suit = card.suit
        if suit != card.suit:
            # wrong suit
            return False
        if not(card.value in royalRanksSet):
            # not a royal card
            return False
    return True

# Possibly overkill, but a wrapper on simulating a situation.
# In short, array of results, best result, average result
# NOTE: For simplification, for now... result = scalar reward only (no debug)
class HandSimResult(object):
    def __init__(self):
        # For reference, and debug
        self.draw_index = 0 # int i in [32] row array of possible 5-card draws.
        self.draw_cards = [] # cards kept (for debug)
        self.draw_string = '' # also for debug, or indexing

        # Actual results
        self.results = []
        self.average_value = 0.0
        self.best_value = 0.0

    # Doesn't update averages.
    def add_result(self, value):
        self.results.append(value)

    def evaluate(self):
        self.average_value = np.mean(self.results)
        self.best_value = np.max(self.results)

    def __str__(self):
        self.evaluate()
        return '%d sample:\t%.2f average\t%.2f maximum' % (len(self.results), self.average_value, self.best_value)

    """
    def __cmp__(self,other):
        return cmp(self.average_value,other.average_value)

    def __gt__(self, sim_result_2):
        return self.average_value > sim_result_2.average_value
        """

    def __lt__(self, sim_result_2):
        return self.average_value < sim_result_2.average_value

    """
    def __eq__(self, sim_result_2):
        return self.average_value == sim_result_2.average_value
        """

# A wrapper around array of cards. But deals with draws, remembering discards, etc
class PokerHand(object):
    def __init__(self):
        self.dealt_cards = []
        self.held_cards = []
        self.discards = []
        self.draw_cards = []
        self.final_hand = []
        self.rank = -1
        self.category = -1
        self.category_name = 'none'

        # optionally, store reward here
        self.reward = 0
        
    # Receive cards from the deck.
    # NOTE: Hardcoded for 1-time draw.
    def deal(self, cards, final_hand=False):
        if final_hand:
            for held_card in self.held_cards:
                self.final_hand.append(held_card)

        for card in cards:
            #print card
            if final_hand:
                self.final_hand.append(card)
                self.draw_cards.append(card)
            else:
                self.dealt_cards.append(card)
                self.held_cards.append(card)

        # Sort?
        """
        if not final_hand:
            self.dealt_cards.sort()
            self.held_cards.sort()
            """

        if final_hand:
            self.evaluate()

    # A bit of a hack. Takes dealt_hand, and draws in place (draw_set of positions, eg {0,4})
    # NOTE: Postions are positions to *keep*. 
    # For thrown away cards, draws from deck, evaluates hand. Return rank, and return cards to the deck, with a shuffle.
    def draw_in_place(self, deck, draw_set, debug_delta=0.000):
        dummy_hand = []
        cards_return = []
        # A. fill dummy hand with cards kept (defined by draw_set)
        # B. add dummy cards from deck
        for i in range(5):
            if i in draw_set:
                dummy_hand.append(self.dealt_cards[i])
            else:
                draw_card = deck.deal_single(track_deal=False)
                dummy_hand.append(draw_card)
                cards_return.append(draw_card)

        # C. evaluate hand, save value
        dummy_rank = hand_rank_five_card(dummy_hand)

        # print it once in a while...
        if random.random() <= debug_delta:
            print 'dummy_hand [%d] %s' % (dummy_rank, ','.join([str(card) for card in dummy_hand]))

        # D. return drawn cards to deck & shuffle it
        deck.return_cards(cards_return, shuffle=True)

        # E. exit with hand evaluation
        return dummy_rank
        

    # For "dealt_cards", tries every possible draw X times, and saves results in a matrix.
    # NOTE: We *completely* ignore draw_cards, final_hand, etc. 
    # NOTE: At the end, deck should contain cards it started with. Possibly, re-shuffled.
    def simulate_all_draws(self, deck, tries, payout_table, debug=True):
        if debug:
            print '\nsimulating all draws for dealt hand [%s]' % (','.join([str(card) for card in self.dealt_cards]))
        self.sim_results = []
        for i in range(len(all_draw_patterns)):
            draw_pattern = all_draw_patterns[i]
            draw_cards = []
            for draw_pos in draw_pattern:
                draw_cards.append(self.dealt_cards[draw_pos])
            #print 'attempting draw_pattern [%s], %d times, on hand [%s]' % (','.join([str(card) for card in draw_cards]), tries, ','.join([str(card) for card in self.dealt_cards]))

            # Hack: 10x tries if royal flush possiblity...
            # NOTE: Need that in the trial. And at the same time, don't want to skew results if randomly get there.
            tries_local = tries
            if is_royal_flush_draw(draw_cards):
                tries_local *= ROYAL_DRAW_MULTIPLIER

            sim_result = HandSimResult()
            sim_result.draw_index = i

            # save for lookup, and debug
            sim_result.draw_cards = draw_cards
            sim_result.draw_string = hand_string(draw_cards)

            for x in range(tries_local):
                # Returns hand rank, and puts cards back in the deck
                hand_rank = self.draw_in_place(deck, draw_pattern)
                hand_payout = payout_table.payout_rank(hand_rank)
                #print '\t$%d' % hand_payout
                sim_result.add_result(hand_payout)

            #print 'for draw_pattern %s, sim result %s\n' %  (str(draw_pattern), str(sim_result))
            
            # Hack, as we just use fixed order of all_draw_patterns to match results
            sim_result.evaluate()
            self.sim_results.append(sim_result)
        
        # print [draw pattern] : [result]
        if debug:
            print 'All %d sim results for hand [%s]:' % (len(self.sim_results), ','.join([str(card) for card in self.dealt_cards]))
            for i in range(len(all_draw_patterns)):
                draw_pattern = all_draw_patterns[i]
                draw_cards = []
                for draw_pos in draw_pattern:
                    draw_cards.append(self.dealt_cards[draw_pos])
                sim_result = self.sim_results[i]
                print '\t[%s]:\t%s' % (','.join([str(card) for card in draw_cards]), str(sim_result))

        # Now, also save the best move.
        # NOTE: Need to have each average evaluated!
        best_result = max(self.sim_results)
        best_index =  self.sim_results.index(best_result)
        best_pattern = all_draw_patterns[best_index]
        best_draw_cards = []
        for draw_pos in best_pattern:
            best_draw_cards.append(self.dealt_cards[draw_pos])
        print '\nbest result:\n\t[%s]:\t%s\n' % (','.join([str(card) for card in best_draw_cards]), str(best_result))

        self.best_result = best_result

    # Look up value, for draw given cards kept
    # NOTE: To compare AI results, for example, versus simulation...
    # NOTE: Card order... is important! Since we do exact string matching.
    def find_draw_value_for_string(self, draw_cards_string):
        for sim_result in self.sim_results:
            if sim_result.draw_string == draw_cards_string:
                print('found sim_result at position %d' % sim_result.draw_index)
                return sim_result.average_value
        print('ERROR: unable to find result for %s' % draw_cards_string)
        return None

    # Recieves string of ints, for positions to draw form (0-4 for 5-card hand)
    # example: '340' draws three cards.
    # NOTE: This is ackward, but good for command-line input
    def draw(self, drawstring=''):
        #print 'processing drawstring |%s|' % drawstring
        drawn_cards = []
        positions_set = set([])
        for position in drawstring:
            pos_int = int(position)
            positions_set.add(pos_int)

        #print 'Try drawing from positions %s' % positions_set
        
        # Draw cards, starting from the end [so that array number preserved]
        for position in sorted(positions_set, reverse=True):
            #print 'Removing card from position %d' % position
            card = self.held_cards.pop(position)
            #print card
            drawn_cards.append(card)
            self.discards.append(card)

        #print 'throwing away %d cards' % len(drawn_cards)
        return drawn_cards

    # Look up with hash tables
    def evaluate(self):
        self.rank = hand_rank_five_card(self.final_hand)
        self.category = hand_category(self.rank)
        self.category_name = categoryName[self.category]

    def __str__(self):
        return '%d-card hand\ndealt: [%s]\nheld: [%s]\ndiscarded: [%s]\ndraw: [%s]\nfinal: [%s] (rank: %d, category: %s)' % (len(self.dealt_cards),
                                                                                      ','.join([str(card) for card in self.dealt_cards]),
                                                                                      ','.join([str(card) for card in self.held_cards]),
                                                                                      ','.join([str(card) for card in self.discards]),
                                                                                      ','.join([str(card) for card in self.draw_cards]),
                                                                                      ','.join([str(card) for card in self.final_hand]),
                                                                                      self.rank, self.category_name)
            
        

class PokerDeck(object):
    def __init__(self, shuffle=True):
        self.cards = []
        for suit in suitsArray:
            for rank in ranksArray:
                card = Card(suit=suit, value=rank)
                self.cards.append(card)
        if (shuffle):
            random.shuffle(self.cards)
        # Just cards, in order, for now.
        self.dealt_cards = [] # Given to players
        self.discard_cards = [] # Returned to the deck
        # NOTE: No burn cards [yet]
        # TODO: Track distinct "deal" and "discard" actions... for takebacks & partial re-shuffle

    # pops cards from deck, returns objects, keeps reference
    def deal(self, num_cards, track_deal=True):
        deal_cards = []
        for x in range(num_cards):
            card = self.cards.pop()
            deal_cards.append(card)
        if track_deal:
            for card in deal_cards:
                self.dealt_cards.append(card)
        return deal_cards

    # remove specific cards, to simulate draw
    def deal_cards(self, cards_array, track_deal=False):
        deal_cards = []
        deck_after = []
        for card in self.cards:
            if card in cards_array:
                deal_cards.append(card)
                if track_deal:
                    self.dealt_cards.append(card)
            else:
                deck_after.append(card)

        assert(len(deal_cards) == len(cards_array))
        self.cards = deck_after

        # Return in the same order...
        #return deal_cards
        return cards_array

    # shortcut, for single card
    def deal_single(self, track_deal=True):
        cards = self.deal(num_cards=1, track_deal=track_deal)
        return cards[0]

    # put cards back in the deck... (for local simulation, etc)
    def return_cards(self, cards_return, shuffle=True):
        for card in cards_return:
            self.cards.append(card)

        if shuffle:
            random.shuffle(self.cards)
        
    # Put discards in the "back" of the deck.
    def take_discards(self, discards):
        for card in discards:
            self.discard_cards.append(card)

