import sys
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

from draw_poker import cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import build_fully_connected_model
from triple_draw_poker_full_output import build_fat_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector
from triple_draw_poker_full_output import evaluate_batch_hands # much faster to evaluate a batch of hands


print('parsing command line args %s' % sys.argv)
parser = argparse.ArgumentParser(description='Play draw video poker, with pre-computed model or rules-based player.')
parser.add_argument('-draw_model', '--draw_model', default=None, help='neural net model for draw poker. Leave empty for rules-based player') # draws, from 32-length array
parser.add_argument('-output', '--output', help='output CSV') # CSV output file, in append mode.
parser.add_argument('-num_hands', '--num_hands', default=None, help='how many hands to run for this model?')
args = parser.parse_args()

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

BATCH_SIZE = 100 # Across all cases

# Use parameters, if we want to apply alternative model shape
USE_FAT_MODEL = False # True # False # True
USE_FULLY_CONNECTED_MODEL = False # True # False

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
RANDOM_PAYOUT = 0.35 # average of random play...
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

        expected_payout = RANDOM_PAYOUT
        return expected_payout

# Human-input player.
class ManualPlayer(CardPlayer):
    # Takes direct action on the hand.
    # Requires user to enter string like '034' or '21' for card positions to discard
    def move(self, hand, deck):
        # TODO
        raise NotImplementedError()

# Simple "one-rule" player.
# Rule: Keep any pair or better. Otherwise, toss all 5 cards.
# Why toss all cards if not keeping a pair? New cards are random, random draw is not...
class OneRuleNaivePlayer(CardPlayer):
    def move(self, hand, deck):
        # What do we have? 
        # print('evaluating hand for Naive play: %s' % hand_string(hand.dealt_cards))
        rank = hand_rank_five_card(hand.dealt_cards)
        category = hand_category(rank)
        if category in set([ROYAL_FLUSH, STRAIGHT_FLUSH, FOUR_OF_A_KIND, FULL_HOUSE, FLUSH, STRAIGHT]):
            print('--> pat hand. Take our bonus.')
            # Now, complete the [empty] draw!
            discards = hand.draw('')
            deck.take_discards(discards)
            new_cards = deck.deal(len(discards))
            hand.deal(new_cards, final_hand=True)
            return jacks_or_better_table_976_9_6[category]

        # Compute frequency for all ranks and suits
        values_count = {value: 0 for value in ranksArray}
        suits_count = {suit: 0 for suit in suitsArray}
        for card in hand.dealt_cards:
            values_count[card.value] += 1
            suits_count[card.suit] += 1
        #print(values_count)
        #print(suits_count)

        # TODO: Flush draw? 3-card royal draw
        has_four_flush = False
        has_three_card_royal = False
        for suit in suitsArray:
            if suits_count[suit] > 3:
                print('--> we has a flush draw')
                has_four_flush = True
            elif suits_count[suit] == 3 and all((card.value in royalRanksSet) or (card.suit != suit) for card in hand.dealt_cards):
                print('--> we has a 3-card royal flush draw')
                has_three_card_royal = True
                
        draw_string = ''
        if category in set([THREE_OF_A_KIND, TWO_PAIR, JACKS_OR_BETTER]):
            print('good pair+ hand. Keep the pair and freeroll...')
            # Toss any cards not part of a pair/trips
            for i in range(0,5):
                card = hand.dealt_cards[i]
                if values_count[card.value] < 2:
                    #print('Card not part of pair+ %s' % card)
                    draw_string += '%d' % i
        
        # If no pair, draw to a flush, or a 3-card royal
        if not draw_string and (has_four_flush or has_three_card_royal):
            print('drawing for a flush')
            for i in range(0,5):
                card = hand.dealt_cards[i]
                if suits_count[card.suit] < 3:
                    #print('Card not part of flush draw %s' % card)
                    draw_string += '%d' % i

        # If no big pair or flush draw, keep a small pair (like 33). Flush is better than small pair.
        if not draw_string and category in set([ONE_PAIR]):
            print('small pair hand. Keep the pair and freeroll...')
            # Toss any cards not part of pair
            for i in range(0,5):
                card = hand.dealt_cards[i]
                if values_count[card.value] < 2:
                    #print('Card not part of small pair %s' % card)
                    draw_string += '%d' % i
            

        # If no clear draw, toss all cards and get fresh hand.
        if not draw_string:
            draw_string = '01234'

        # Now, complete the draw!
        discards = hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        hand.deal(new_cards, final_hand=True)

        expected_payout = jacks_or_better_table_976_9_6[category]
        if expected_payout == 0:
            expected_payout = RANDOM_PAYOUT
        return expected_payout
        


"""
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
"""


# Plays based on a Theano function evaluation!
# NOTE: All sizes & functions must match.
class TrainedModelPlayer(CardPlayer):
    # Network, with weights already put in.
    def __init__(self, output_layer, test_batch, input_layer=None):
        self.output_layer = output_layer
        self.input_layer = input_layer
        #self.test_batch = test_batch

    # From here, will just push to the back of the batch...
    def clear_batch(self):
        # For batching network evaluation...
        self.draw_hands = []
        self.decks = []
        self.expected_payouts = []

    # Simply save the references, for later. Does not check correctness.
    def batch_move(self, hand, deck):
        self.draw_hands.append(hand)
        self.decks.append(deck)

    # creates batch, from the moves waiting for evaluation. Runs once through neural network. Makes moves in-hand
    def complete_batch_moves(self):
        print('completing batch moves.')

        string_cases = [hand_string(hand.dealt_cards) for hand in self.draw_hands]
        assert(len(string_cases) == BATCH_SIZE)

        # print('About to evaluate batch hands on %s' % string_cases)

        hand_draws_matrix = evaluate_batch_hands(self.output_layer, string_cases, input_layer=self.input_layer)

        print('returned cases: %s' % str(hand_draws_matrix.shape))

        for i in range(BATCH_SIZE):
            hand = self.draw_hands[i]
            deck = self.decks[i]
            hand_string_dealt = hand_string(hand.dealt_cards)
            print('dealt %s' % hand_string_dealt)
            
            hand_draws_vector = hand_draws_matrix[i]
            
            print('All 32 values: %s' % str(hand_draws_vector))

            best_draw = np.argmax(hand_draws_vector)
        
            print('Best draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
            expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        
            draw_string = ''
            for i in range(0,5):
                if not (i in all_draw_patterns[best_draw]):
                    draw_string += '%d' % i

            print('Draw string from AI! |%s|' % draw_string)

            discards = hand.draw(draw_string)
            deck.take_discards(discards)
            new_cards = deck.deal(len(discards))
            hand.deal(new_cards, final_hand=True)
        
            self.expected_payouts.append(expected_payout)


    # Takes action on the hand. But first... get Theano output...
    def move(self, hand, deck):
        hand_string_dealt = hand_string(hand.dealt_cards)
        print('dealt %s' % hand_string_dealt)

        # Get 32-length vector for each possible draw, from the model.
        hand_draws_vector = evaluate_single_hand(self.output_layer, hand_string_dealt) #, test_batch=self.test_batch)

        print('All 32 values: %s' % str(hand_draws_vector))

        best_draw = np.argmax(hand_draws_vector)
        
        print('Best draw: %d [value %.2f] (%s)' % (best_draw, hand_draws_vector[best_draw], str(all_draw_patterns[best_draw])))
        expected_payout = hand_draws_vector[best_draw] # keep this, and average it, as well
        
        draw_string = ''
        for i in range(0,5):
            if not (i in all_draw_patterns[best_draw]):
                draw_string += '%d' % i

        print('Draw string from AI! |%s|' % draw_string)

        discards = hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        hand.deal(new_cards, final_hand=True)
        
        return expected_payout



# To avoid un-unnecessary computations... batch the evaluations.
# Everything else is very inexpensive.
def game_batch(round, cashier, player):
    print('starting bactch for round %s' % round)

    # For each batch number, create deck, hand, and cache your request with the AI player
    batches = []

    # Assumes AI player (else no need to batch)
    player.clear_batch()
    for i in range(BATCH_SIZE):
        if i == 0:
            print '\n-- New Round %d --\n' % round

        deck = PokerDeck(shuffle=True)
        draw_hand = PokerHand()
        deal_cards = deck.deal(5)
        draw_hand.deal(deal_cards)
        
        # Saves hand for future evaluation
        player.batch_move(hand=draw_hand, deck=deck)

        batches.append(draw_hand)

    # Now perform all moves, in a batch.
    player.complete_batch_moves()

    # For reference... to compare visually
    print(all_draw_patterns)

    # Now evaluate final hands, and store return values...
    draw_hands = []
    payouts = []
    expected_payouts = []

    # Just iterate over the batch. Expect order to match.
    print('Ran the AI batch. Evaluating %d hands...\n' % len(batches))
    for i in range(len(batches)):
        draw_hand = batches[i]

        print draw_hand
        print('-------')

        # Evaluate the hand. This means returning a number corresponding to hand order, and category.
        draw_hand.evaluate()
        pay_him = cashier.payout(draw_hand)
        draw_hand.reward = pay_him
        expected_payout = player.expected_payouts[i]

        draw_hands.append(draw_hand)
        payouts.append(pay_him)
        expected_payouts.append(expected_payout)

    return (draw_hands, payouts, expected_payouts)

# Play a round of poker
# A. Create a deck, shuffle & deal hand.
# B. Pass game info to AI, get response, draw
# F. Evaluate final hand
# G. Return result
def game(round, cashier, player=None):
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
    if not player:
        # player = RandomPlayer() # random draw
        player = OneRuleNaivePlayer() # keeps any pair, straight, flush, etc.

    # Optionally, player also returns is expectation of the value...
    expected_payout = player.move(hand=draw_hand, deck=deck)

    print draw_hand

    # Evaluate the hand. This means returning a number corresponding to hand order, and category.
    draw_hand.evaluate()
    pay_him = cashier.payout(draw_hand)
    draw_hand.reward = pay_him
    
    print 'pay the man his $%d' % pay_him

    return (draw_hand, pay_him, expected_payout)

# Saves to rows, matching supplied header. Currently:
# ['dealt_cards', 'held_cards', 'discards', 'draw_cards', 'final_hand', 'rank', 'category', 'category_name', 'reward']
def output_hand_csv(poker_hand, header_map):
    output_map = {}
    if poker_hand.dealt_cards:
        output_map['dealt_cards'] = hand_string(poker_hand.dealt_cards)
    if poker_hand.held_cards:
        output_map['held_cards'] = hand_string(poker_hand.held_cards)
    if poker_hand.discards:
        output_map['discards'] = hand_string(poker_hand.discards)
    if poker_hand.draw_cards:
        output_map['draw_cards'] = hand_string(poker_hand.draw_cards)
    if poker_hand.final_hand:
        output_map['final_hand'] = hand_string(poker_hand.final_hand)
    output_map['rank'] = poker_hand.rank
    output_map['category'] = poker_hand.category
    output_map['category_name'] = poker_hand.category_name
    output_map['reward'] = poker_hand.reward

    output_row = VectorFromKeysAndSparseMap(keys=header_map, sparse_data_map=output_map, default_value = '')
    return output_row

# Play a bunch of hands, and record results for training
def play(sample_size, output_file_name, model_filename=None):
    # Compute rewards with payout table
    cashier = JacksOrBetter() # "976-9-6" Jacks or Better -- with 100% long-term payout.

    #func = get_model_from_pickle('model.pickle')

    # If model file provided, unpack model, and create intelligent agent.
    if model_filename and os.path.isfile(model_filename):
        print('\nExisting model in file %s. Attempt to load it!\n' % model_filename)
        all_param_values_from_file = np.load(model_filename)
        
        # Size must match exactly!
        if USE_FAT_MODEL:
            output_layer, input_layer, layers  = build_fat_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )
        elif USE_FULLY_CONNECTED_MODEL:
            output_layer, input_layer, layers  = build_fully_connected_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )
        else:
            output_layer, input_layer, layers  = build_model(
                HAND_TO_MATRIX_PAD_SIZE, 
                HAND_TO_MATRIX_PAD_SIZE,
                32,
                )

        #print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file)

        # Test the model, by giving it dummy inputs
        # Test cases -- it keeps the two aces. But can it recognize a straight? A flush? Trips? Draw?? Two pair??
        test_cases = ['As,Ad,4d,3s,2c', 'As,Ks,Qs,Js,Ts', '3h,3s,3d,5c,6d', '3h,4s,3d,5c,6d', '2h,3s,4d,6c,5s',
                      '8s,Ad,Kd,8c,Jd', '8s,Ad,2d,7c,Jd', '2d,7d,8d,9d,4d', '7c,8c,Tc,Js,Qh', '2c,8s,5h,8d,2s',
                      '[8s,9c,8c,Kd,7h]', '[Qh,3h,6c,5s,4s]', '[Jh,Td,9s,Ks,5s]', '[6c,4d,Ts,Jc,6s]', 
                      '[4h,8h,2c,7d,3h]', '[2c,Ac,Tc,6d,3d]', '[Ad,3c,Tc,4d,5d]'] 

        for i in range(BATCH_SIZE - len(test_cases)):
            test_cases.append(test_cases[1])
        test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)
        predict_model(output_layer=output_layer, input_layer=input_layer , test_batch=test_batch)

        print('Cases again %s' % str(test_cases))

        print('Creating player, based on this pickled model...')

        player = TrainedModelPlayer(output_layer, test_batch, input_layer=input_layer)
    else:
        player = OneRuleNaivePlayer() # RandomPlayer() # random draw

    # Run it for each game...
    round = 0
    results = []
    expected_results = [] # if player supplies us internal estimate of his hand value... 
    if output_file_name:
        output_file = open(output_file_name, 'w')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(POKER_GAME_HEADER)
        csv_header_map = CreateMapFromCSVKey(POKER_GAME_HEADER)
    else:
        csv_writer = None

    now = time.time()
    try:
        while round < sample_size:
            sys.stdout.flush()

            # If AI model, then batch it. Otherwise, just go hand at a time.
            if isinstance(player, TrainedModelPlayer):
                # hand, payout, expected_payout = game(round, cashier, player=player)
                # Need to batch games. Since network evaluation so expensive!
                hand_batch, payout_batch, expected_payout_batch = game_batch(round, cashier, player=player)

                # Iterate over the batch...
                for i in range(BATCH_SIZE):
                    hand = hand_batch[i]
                    payout = payout_batch[i]
                    expected_payout = expected_payout_batch[i]
                
                    results.append(payout)
                    expected_results.append(expected_payout)

                    # Save hand to CSV, if output supplied.
                    if csv_writer:
                        hand_csv_row = output_hand_csv(poker_hand=hand, header_map=csv_header_map)
                        csv_writer.writerow(hand_csv_row)

                    # Hack, to show matrix for final hand.
                    # print hand_to_matrix(hand.final_hand)
                    # print(hand_string(hand.final_hand))
                    # pretty_print_hand_matrix(hand.final_hand)

                    round += 1
            else:
                for i in range(BATCH_SIZE):
                    hand, payout, expected_payout = game(round, cashier, player=player)

                    results.append(payout)
                    expected_results.append(expected_payout)
                    
                    # Save hand to CSV, if output supplied.
                    if csv_writer:
                        hand_csv_row = output_hand_csv(poker_hand=hand, header_map=csv_header_map)
                        csv_writer.writerow(hand_csv_row)

                    round += 1
            
            # Print the best 100 results. Why? Shows skew at the top. 
            print '\npaid %s' % sorted(results, reverse=True)[0:100]
            print('\n%d hands took %.2fs. Running return: %.5f. Expected return: %.5f' % (len(results), 
                                                                                          time.time() - now, 
                                                                                          np.mean(results), 
                                                                                          np.mean(expected_results)))
            # sys.exit(-3)

    except KeyboardInterrupt:
        pass

    if csv_writer:
        print '\nwrote %d rows' % len(results)
        output_file.close()

    print '\npaid %s' % sorted(results, reverse=True)[0:100]
    print '\nexpected %s' % sorted(expected_results, reverse=True)[0:100]
    
    # What do we expect with these hands? And more importantly, how did we make out?
    print '\nexpected:\tave: %.5f\tstdev: %.5f\tskew: %.5f' % (np.mean(expected_results), np.std(expected_results), ss.skew(expected_results))
    print '\nstats(%d):\tave: %.5f\tstdev: %.5f\tskew: %.5f' % (len(results), np.mean(results), np.std(results), ss.skew(results))

    sys.stdout.flush()

if __name__ == '__main__':
    samples = 100
    output_file_name = '%d_samples_model_choices.csv' % samples

    # Input model filename if given
    # TODO: set via command line flagz
    model_filename = args.draw_model
    if args.num_hands:
        samples = int(args.num_hands)
    if args.output:
        output_file_name = args.output

    """
    if len(sys.argv) >= 2:
        model_filename = sys.argv[1]

        # Uniquely ID details
        output_file_name = '%d_samples_model_choices.csv' % (samples)
        """

    # TODO: Take num samples from command line.
    play(sample_size=samples, output_file_name=output_file_name, model_filename=model_filename)


