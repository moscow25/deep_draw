import sys
import csv
import logging
import time
import math
import re
import random
import numpy as np
import scipy.stats as ss
from poker_lib import *
from poker_util import *

from simulate_draw_values import game_full_sim # computes average for a hand & draw

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Analyzes draw-card choices.
"""

# How input is given to us. We only care about "dealt cards" and "held cards"... so can monday morning QB it
POKER_GAME_HEADER = ['dealt_cards', 'held_cards', 'discards', 'draw_cards', 'final_hand', 'rank', 'category', 'category_name', 'reward']

# for CSV output of biggest errors..
ERROR_HAND_HEADER = ['error', 'dealt_hand_string', 'best_result_string', 'best_result_value', 'ai_draw_string', 'ai_draw_value']

# thresholds for error types...
TINY_ERROR_THRESHOLD = 0.08 # seems large, but can still be in the noise
SMALL_ERROR_THRESHOLD = 0.25 # anything bigger, is a major error

# Initialize, to run through inputs, and not read all...
SKIP_LINE_PROBABILITY = 0.80

# How many errors in debug?
NUM_SHOW_ERRORS = 25 

# Get relevant data, simulate draw, and notice differences.
def evaluate_draw_line(line, csv_key_map, tries_per_draw, cashier):
    dealt_hand_string = line[csv_key_map['dealt_cards']]
    print('Evaluating dealt hand: %s' % dealt_hand_string)
    hand_array = hand_string_to_array(dealt_hand_string)
    hand_cards = [Card(suit=suitFromChar[card_str[1]], value=valueFromChar[card_str[0]]) for card_str in hand_array]

    # To keep deck true... need to extract cards from the deck, for what's already our hand...
    deck = PokerDeck(shuffle=True)
    deal_cards = deck.deal_cards(hand_cards)
    draw_hand = PokerHand()
    draw_hand.deal(deal_cards)

    print draw_hand

    # Simulate the hand, and see what sim says!
    draw_hand.simulate_all_draws(deck=deck, tries=tries_per_draw, payout_table=cashier, debug=True)

    # Best result, according to sim
    best_result = draw_hand.best_result
    best_result_string = hand_string(best_result.draw_cards)
    best_result_value = best_result.average_value

    print('Best result: %s [%.2f]' % (best_result_string, best_result_value))

    # Here is the draw we actually made...
    ai_draw_string = line[csv_key_map['held_cards']]
    # Fix a bug in the inputs
    if not ai_draw_string:
        ai_draw_string = '[]'

    print('Draw we actually made: %s' % ai_draw_string)

    if (ai_draw_string == best_result_string):
        print ('\t--> Draws match!!!\n')
        ai_draw_value = best_result_value
        error = 0.0
    else:
        print ('\t--> NO MATCH\n\ncomputing error...\n')
        ai_draw_value = draw_hand.find_draw_value_for_string(ai_draw_string)

        print('AI draw value: %.2f' % ai_draw_value)

        error = best_result_value - ai_draw_value

        print ('error: %.2f\n' % error)
    
    # No need to return hand? Just output the results...
    return (error, dealt_hand_string, best_result_string, best_result_value, ai_draw_string, ai_draw_value)


# Go line-by-line, simulate results, and see if AI made right choice.
# Rank errors by magnitude.
def evaluate_draws(input_filename, output_filename, tries_per_draw, max_input=50):
    # Load input into CSV reader
    csv_reader = csv.reader(open(input_filename, 'rU'))
    csv_key = None
    csv_key_map = None

    cashier = JacksOrBetter() # "976-9-6" Jacks or Better -- with 100% long-term payout.

    lines = 0
    hands = 0

    # result buckets
    no_error = 0
    tiny_error = 0
    small_error = 0
    large_error = 0

    error_results = []

    now = time.time()
    for line in csv_reader:
        lines += 1
        if lines % 10000 == 0:
            print('Read %d lines' % lines)

        # Read the CSV key, so we look for columns in the data, not fixed positions.
        if not csv_key:
            print('CSV key' + str(line))
            csv_key = line
            csv_key_map = CreateMapFromCSVKey(csv_key)
        else:
            # For debugging, or just down-sample, skip lines with probability...
            if random.random() < SKIP_LINE_PROBABILITY:
                continue

            # Skip any mall-formed lines.
            try:
		sys.stdout.flush()
                print('\n----------------')
                # Run simulation, and save results...
                result = evaluate_draw_line(line, csv_key_map, tries_per_draw, cashier)
                print(result) 

                # Remember, for unpacking later.
                (error, dealt_hand_string, best_result_string, best_result_value, ai_draw_string, ai_draw_value) = result

                # Bucket results
                if error == 0:
                    no_error += 1
                elif error < TINY_ERROR_THRESHOLD:
                    tiny_error += 1
                elif error < SMALL_ERROR_THRESHOLD:
                    small_error += 1
                else:
                    large_error += 1

                # Skip minor errors, and record those over a threshold
                if error >= TINY_ERROR_THRESHOLD:
                    error_results.append(result)

            except (IndexError, ValueError):
                print('\nskipping malformed input line:\n|%s|\n' % line)
                continue

            hands += 1

            print('\n%d hands took %.2fs' % (hands, time.time() - now))
            print('%d no error\t%d tiny error\t%d small error\t%d big error\n\tbiggest errors:' % (no_error, tiny_error, small_error, large_error)) 

            # Show biggest errors so far...
            biggest_errors = sorted(error_results, reverse=True)[0:NUM_SHOW_ERRORS]
            for big_error in biggest_errors:
                print big_error

            if hands >= max_input:
                break

    print('Analyzed %d hands for AI draw errors' % hands)

    # Save to CSV.
    if output_filename:
        output_file = open(output_filename, 'w')
        csv_writer = csv.writer(output_file)
        csv_writer.writerow(ERROR_HAND_HEADER)
        csv_header_map = CreateMapFromCSVKey(ERROR_HAND_HEADER)
    else:
        csv_writer = None

    if csv_writer:
        for error_line in sorted(error_results, reverse=True):
            # TODO: index the lines... but really just dump it
            csv_writer.writerow(error_line)


if __name__ == '__main__':
    tries_per_draw = 2000
    max_input = 2000 # hands to examine
    # Load CSV with POKER_GAME_HEADER header.
    if len(sys.argv) >= 2:
        input_filename = sys.argv[1]

        output_filename = 'errors_%d_%s' % (max_input, input_filename)

    # For each hand, analyze the move. Record errors, and magnitudes of errors (skip hands with agreement or small error)
    evaluate_draws(input_filename, output_filename, tries_per_draw, max_input=max_input)
