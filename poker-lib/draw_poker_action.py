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


# If we simulate allin value (vs current hand, and random hand) as part of Holdem output... how many times do we simulate?
# TODO: Record how long it takes, add easy option to turn it off in production.
# NOTE: For 200x counts... stdev is +-0.02 for some cases. So we can be way off w/r/t predictions... but averages out over many hands.
# Noise is ok, and even 500x counts... really slows down the play. Even with caching. Maybe on a fast machine... 
SIMULATE_ALLINS_COUNT = 200 # 500 # 200 # 1000 -- accurate, but takes too long. We should cache... since X vs Y lookup (for bet streets)

# Heuristics, to evaluate hand actions. On 0-1000 scale, where wheel is 1000 points, and bad hand is 50-100 points.
# Meant to map to rough % of winning at showdown. Tuned for ring game, so random hand << 500.
# For 'deuce,' random heuristic 0.300 worked. We need to either tie this to FORMAT, etc.
# RANDOM_HAND_HEURISTIC_BASELINE = 0.300 # baseline, before looking at any cards.
# For 'holdem,' we need a higher baseline heuristic. Average hand is by definition 0.500. Since we literally compare to random hands, HU.
RANDOM_HAND_HEURISTIC_BASELINE = 0.4000 # baseline, before considering cards

# Cashier for 2-7 lowball. Evaluates hands, as well as compares hands.
class DeuceLowball(PayoutTable):
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

# Return a string, encoding action for a round, or for a hand
def encode_bets_string(actions, format='deuce'):
    if format=='nlh':
        return encode_big_bets_string(actions)
    else:
        return encode_limit_bets_string(actions)

# If Limit game, 0 = check/call, 1 = bet/raise. Ignore other actions
def encode_limit_bets_string(actions):
    actions_string = ''
    for action in actions:
        if action.type in ALL_BETS_SET:
            actions_string += '1'
        elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
            actions_string += '0'
        else:
            # Don't encode non-bets
            continue
    return actions_string

# Did we end a bet round?
# A. check-check
# B. call (except to call a blind)
# C. call-check (blinds)
def string_ends_big_bets_round(actions_string):
    if not actions_string:
        return False
    if len(actions_string) >= 2 and actions_string[-1] == 'k' and actions_string[-2] == 'k':
        return True
    if len(actions_string) >= 2 and actions_string[-1] == 'c':
        return True
    if len(actions_string) >= 2 and  actions_string[-1] == 'k' and actions_string[-2] == 'c':
        return True
    return False

# If big bet game, use ACPC encoding. k = check, c = call, bXXX = bet/raise to XXX amount
# NOTE: ACPC encodes bet by player for entire street. So a 3-bet: b300b700b2100f
def encode_big_bets_string(actions):
    actions_string = ''
    for action in actions:
        if action.type in ALL_BETS_SET:
            # Add '/' if previous bet was a call, or check, check
            if string_ends_big_bets_round(actions_string):
                actions_string += '/'
            actions_string += 'b'
            # Encode bet size, if president
            # NOTE: ACPC encodes *entire bet size* this street. So make sure that matches 
            bet_size = 0
            if action.bet_size:
                # bet_size = int(action.bet_size)
                bet_size = int(action.bet_this_street)
            actions_string += '%d' % bet_size
        elif action.type == CHECK_HAND:
            # Add '/' if previous bet was a call, or check, check
            if string_ends_big_bets_round(actions_string):
                actions_string += '/'
            actions_string += 'k'
        elif action.type in ALL_CALLS_SET:
            actions_string += 'c'
        elif action.type == FOLD_HAND:
            actions_string += 'f'
        else:
            # Don't encode non-bets
            continue
    return actions_string
    

# At risk of over-engineering, use a super-class to encapsulate each type
# of poker action. Specifically, betting, checking, raising.
# NOTE: Usually, is prompted, decides on an action, and passes control.
# (but not always, as there are forced bets, etc)
# This system strong assumes, that actions take place in order, by a dealer. Poker is a turn-based game.
class PokerAction:
    # TODO: Copy of the game state (hands, etc)? Zip of the information going into this action?
    def __init__(self, action_type, actor_name, pot_size, bet_size, format = 'deuce', chip_stack = 0.0, bet_this_street = 0.0, bet_faced = 0.0):
        self.type = action_type
        self.name = actionName[action_type]
        self.format = format

        # For now... just a hack to distinguish between B = button and F = blind player, first to act
        self.actor_name = actor_name
        self.pot_size = pot_size # before the action
        self.bet_size = bet_size # --> bet being made with *this action*
        self.bet_this_street = max(bet_this_street, bet_size) # For ACPC compatibility, encode 3bet as b100b300b1200 [total amount]
        self.bet_faced = bet_faced
        self.chip_stack = chip_stack # player stack... *before the bet* [relevent for big-bet only]
        # TODO: Add "stack" or "stack_remaining" for NLH. Also, bool allin?
        self.pot_odds = (pot_size / bet_size if bet_size else 0.0)
        self.value = RANDOM_HAND_HEURISTIC_BASELINE # baseline of baselines
        self.hand = None
        self.best_draw = None
        self.hand_after = None
        if format == 'deuce':
            self.cashier = DeuceLowball() # Computes categories for hands, compares hands by 2-7 lowball rules
        elif format == 'holdem' or format == 'nlh':
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
        elif (self.format == 'holdem' or self.format == 'nlh') and self.hand and self.oppn_hand and self.best_draw:
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
            print('unknown current_win_percentage for format %s' % self.format)
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
        # NOTE: Encodes NLH bets in ACPC format. 'k' = check, 'b455' = bet/raise to 455
        self.actions_this_round_string = encode_bets_string(actions_this_round, format=self.format)
        
        """
        self.actions_this_round_string = ''
        for action in actions_this_round:
            if action.type in ALL_BETS_SET:
                self.actions_this_round_string += '1'
            elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                self.actions_this_round_string += '0'
            else:
                # Don't encode non-bets
                continue
                """

        # Actions for the entire hand.
        # TODO: Add breakers, '|' for draw event.
        self.actions_full_hand_string = encode_bets_string(actions_full_hand, format=self.format)

        """
        self.actions_full_hand_string = ''
        for action in actions_full_hand:
            if action.type in ALL_BETS_SET:
                self.actions_full_hand_string += '1'
            elif action.type == CHECK_HAND or action.type in ALL_CALLS_SET:
                self.actions_full_hand_string += '0'
            else:
                # Don't encode non-bets
                continue
                """

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
    
    # Simulate allin value vs random opponent hand. How good is our hand?
    def simulate_allin_vs_oppn(self, num_samples = SIMULATE_ALLINS_COUNT, allin_cache=None):
        # Currently, allin values only implemented for some games
        if not(self.format == 'holdem' or self.format == 'nlh'):
            return

        # Create objects from current hand... 
        # TODO: Allin simulation should be a library sub-routine...
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


        # If cache exists, look up cache, in case already computed.
        # NOTE: Category values not used. Will be []
        if allin_cache:
            (allin_value, allin_stdev, category_values) = allin_cache.lookup(our_hand.dealt_cards, oppn_hand.dealt_cards, flop, turn, river)
            
            # If cache hit... just output and return.
            if allin_value != None:
                allin_error = allin_stdev / np.sqrt(num_samples)

                # print('[cache] allin value [vs oppn] is %.4f +-%.4f (%.4f stdev)' % (allin_value, allin_error, allin_stdev))

                self.allin_value = allin_value
                self.allin_stdev = allin_stdev
                return


        # Information that's fixed in stone for simulation
        all_dealt_cards = community.cards() + our_hand.dealt_cards + oppn_hand.dealt_cards
        #print('removing %d dealt cards from the deck: %s' % (len(all_dealt_cards), hand_string(all_dealt_cards)))
        hand_round = community.round

        # Deck with remaining cards
        deck = PokerDeck(shuffle=False)
        for card in all_dealt_cards:
            deck.remove_card(card)
        #print('deck contains %d cards after removal' % len(deck.cards))

        # Ok, now we're ready to deal to the end, record, return, shuffle, repeat X times
        # TODO: This should be a library sub-routine
        hand_results = []
        for i in range(num_samples):
            # NOTE: Clear oppn cards, if simulating allin vs random hand
            community.rewind(deck=deck, round=hand_round)
            deck.shuffle()
            #print('deck now %d cards' % len(deck.cards))
            community.deal(deck=deck,runway=True)
            our_hand.evaluate()
            oppn_hand.evaluate()
            #print(our_hand)
            #print(oppn_hand)
            if our_hand.rank > oppn_hand.rank:
                result = 0.0
                #print('oppn wins')
            elif our_hand.rank < oppn_hand.rank:
                result = 1.0
                #print('our hand wins')
            else:
                result = 0.5
                #print('we ties')
            hand_results.append(result)

        allin_value = np.mean(hand_results)
        allin_stdev = np.std(hand_results)
        allin_error = allin_stdev / np.sqrt(num_samples)

        # print('allin value [vs oppn] is %.4f +-%.4f (%.4f stdev)' % (allin_value, allin_error, allin_stdev))

        self.allin_value = allin_value
        self.allin_stdev = allin_stdev

        # If cache exists, update the cache
        if allin_cache:
            allin_cache.insert(our_hand.dealt_cards, oppn_hand.dealt_cards, flop, turn, river, allin_value, allin_stdev)

    # Simulate allin value vs random opponent hand. How good is our hand?
    def simulate_allin_vs_random(self, num_samples = SIMULATE_ALLINS_COUNT, allin_cache=None):
        # Currently, allin values only implemented for some games
        if not(self.format == 'holdem' or self.format == 'nlh'):
            return

        # Create objects from current hand... 
        # TODO: Allin simulation should be a library sub-routine...
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
        oppn_hand = HoldemHand(cards=[], community=community) # empty opponenet hand

        # If cache exists, look up cache, in case already computed.
        if allin_cache:
            (allin_value, allin_stdev, category_values) = allin_cache.lookup(our_hand.dealt_cards, oppn_hand.dealt_cards, flop, turn, river)
            
            # If cache hit... just output and return.
            if allin_value != None:
                # allin_error = allin_stdev / np.sqrt(num_samples)
                # print('[cache] allin value [vs random] is %.4f +-%.4f (%.4f stdev)' % (allin_value, allin_error, allin_stdev))

                self.allin_value_vs_random = allin_value
                self.allin_stdev_vs_random = allin_stdev
                self.category_values_vs_random = category_values
                return

        # Information that's fixed in stone for simulation
        all_dealt_cards = community.cards() + our_hand.dealt_cards + oppn_hand.dealt_cards
        #print('removing %d dealt cards from the deck: %s' % (len(all_dealt_cards), hand_string(all_dealt_cards)))
        hand_round = community.round

        # Deck with remaining cards
        deck = PokerDeck(shuffle=False)
        for card in all_dealt_cards:
            deck.remove_card(card)

        # Ok, now we're ready to deal to the end, record, return, shuffle, repeat X times
        # Calculate wins/losses, and also hand categories made [house, flush, etc]
        # TODO: This should be a library sub-routine
        hand_results = []
        category_results = [0.0 for category in HIGH_HAND_CATEGORIES]
        for i in range(num_samples):
            community.rewind(deck=deck, round=hand_round)
            # Clear oppn cards, since simulating allin vs random hand
            deck.return_cards(cards_return=oppn_hand.dealt_cards, shuffle=False)
            oppn_hand.dealt_cards = []

            # deal new hand and evaluate
            deck.shuffle()
            oppn_hand.dealt_cards = deck.deal(2) # new cards for opponent
            community.deal(deck=deck,runway=True)
            our_hand.evaluate()
            oppn_hand.evaluate()
            if our_hand.rank > oppn_hand.rank:
                result = 0.0
            elif our_hand.rank < oppn_hand.rank:
                result = 1.0
            else:
                result = 0.5
            hand_results.append(result)

            # Record the category of our hand
            category = our_hand.category
            category_index = high_hand_categories_index[category]
            category_results[category_index] += 1.0

        # For correctness, clear 'Random' opponent hand after we're done.
        oppn_hand.dealt_cards = []

        allin_value = np.mean(hand_results)
        allin_stdev = np.std(hand_results)
        allin_error = allin_stdev / np.sqrt(num_samples)

        # print('allin value [vs random] is %.4f +-%.4f (%.4f stdev)' % (allin_value, allin_error, allin_stdev))

        self.allin_value_vs_random = allin_value
        self.allin_stdev_vs_random = allin_stdev

        # Categories (% to make specific hands like pair, flush, etc)
        category_values = [category_results[high_hand_categories_index[category]] / num_samples for category in HIGH_HAND_CATEGORIES]
        #category_values_debug = [[categoryName[category], category_results[high_hand_categories_index[category]] / num_samples] for category in HIGH_HAND_CATEGORIES]
        #print('\n%s category odds %s' % (our_hand, category_values_debug))

        self.category_values_vs_random = category_values

        # If cache exists, update the cache
        if allin_cache:
            allin_cache.insert(our_hand.dealt_cards, oppn_hand.dealt_cards, flop, turn, river, allin_value, allin_stdev, category_values)

    # For training, optionally simulate in-place to get
    # - Allin value vs current opponent
    # - Allin value vs random opponent hand (just our own hand strength)
    # NOTE: Easy to add more outputs... as long as no new loop (over random hands) is needed
    # TODO: This is expensive. Make sure to include an option to disable this run run faster.
    def simulate_allin_values(self, allin_cache=None):
        # Currently, allin values only implemented for some games
        if not(self.format == 'holdem' or self.format == 'nlh'):
            return

        # Print out, how long it takes to simulate allin values (since per-move is needed)
        now = time.time()

        # check if we computed this already
        if not(hasattr(self, 'allin_vs_random') and self.allin_vs_random >= 0.0):
            self.simulate_allin_vs_random(allin_cache=allin_cache)

        # check if we computed this already
        if not(hasattr(self, 'allin_vs_oppn') and self.allin_vs_oppn >= 0.0):
            self.simulate_allin_vs_oppn(allin_cache=allin_cache)

        # print('%.2fs to simulate allin values (%d times)' % (time.time() - now, SIMULATE_ALLINS_COUNT))
        
    # Consise summary, of the action taken.
    def __str__(self):
        #return('%s(%s)\tPot: %d\tBet: %d\t(Face: %d\tStack: %d\tStreet: %d)' % 
        #       (self.name, self.actor_name, self.pot_size, self.bet_size, self.bet_faced, self.chip_stack, self.bet_this_street))
        return('%s(%s)\tPot: %d\tBet: %d' % (self.name, self.actor_name, self.pot_size, self.bet_size))

    # Return array of outputs, corresponding to CSV header map order. Empty fields are ''
    def csv_output(self, header_map, allin_cache = None):
        # If available for this game (Holdem, etc), output allin values for our hand.
        # NOTE: Since we simulate in-place, this can be very expensive. 
        # TODO: Track time spent on this activity.
        # ~> turn off if running in production (not for data generation purposes)
        # In general... include an option to turn off "csv_output" for faster performance (in ACPC, etc)
        self.simulate_allin_values(allin_cache=allin_cache)

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

        # Added for NLH: stack-based numbers.
        # NOTE: All can be derived from other stats, but useful to see directly.
        # 'bet_faced', 'stack_size', 'bet_this_street'
        output_map['bet_faced'] = self.bet_faced
        output_map['stack_size'] = self.chip_stack
        output_map['bet_this_street'] = self.bet_this_street

        # Previous actions.
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
            # Don't save draws vector... for a non-draw game!
            # HACK: NLH uses it too, for something else
            if self.format == 'deuce' or self.format == 'nlh':
                output_map['num_draw_vector'] = self.num_draw_vector

        # If we choose to compute these, allin values vs current opponent, and vs random player.
        if hasattr(self, 'allin_value'):
            output_map['allin_vs_oppn'] = self.allin_value
        if hasattr(self, 'allin_stdev'):
            output_map['stdev_vs_oppn'] = self.allin_stdev
        if hasattr(self, 'allin_value_vs_random'):
            output_map['allin_vs_random'] = self.allin_value_vs_random
        if hasattr(self, 'allin_stdev_vs_random'):
            output_map['stdev_vs_random'] = self.allin_stdev_vs_random

        # Vector, wtih all hi-category hand values (odds to make pair, flush, etc)
        if hasattr(self, 'category_values_vs_random'):
            output_map['allin_categories_vector'] = self.category_values_vs_random
        
        # ['hand', 'draws_left', 'bet_model', 'value_heuristic', 'position', 'num_cards_kept', 'num_opponent_kept', 'best_draw', 'hand_after', 'action', 'pot_size', 'bet_size', 'pot_odds', 
        # 'bet_faced', 'stack_size', 'bet_this_street', 
        # 'bet_this_hand', 'actions_this_round', 'actions_full_hand', 'total_bet', 'result', 'margin_bet', 'margin_result', 'current_margin_result', 'future_margin_result', 'oppn_hand', 'current_hand_win', 'hand_num', 'running_average', 'bet_val_vector', 'act_val_vector', 'num_draw_vector', 'allin_vs_oppn', 'stdev_vs_oppn', 'allin_vs_random', 'stdev_vs_random', 'allin_categories_vector']
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
    def __init__(self, actor_name, pot_size, format, chip_stack, bet_this_street, bet_faced):
        PokerAction.__init__(self, action_type = CHECK_HAND, actor_name = actor_name, pot_size = pot_size, 
                             bet_size = 0, format=format, chip_stack=chip_stack, bet_this_street=bet_this_street, bet_faced=bet_faced)

class FoldStreet(PokerAction):
    def __init__(self, actor_name, pot_size, format, chip_stack, bet_this_street, bet_faced):
        PokerAction.__init__(self, action_type = FOLD_HAND, actor_name = actor_name, pot_size = pot_size, 
                             bet_size = 0, format=format, chip_stack=chip_stack, bet_this_street=bet_this_street, bet_faced=bet_faced)

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

# Also, consider no-Limit actions: NL-bet and NL-raise. 
# NOTE: Design decision: bounds checking should happen elsewhere (see how much is left, etc)
# TODO: Should also report effective stack size. (amount left by player after this action)
# --> 200 BB should be global constant... tracked outside of bets actions
class CallNoLimit(PokerAction):
    def __init__(self, actor_name, pot_size, this_bet, format, chip_stack, bet_this_street, bet_faced):
        assert this_bet > 0, 'error calling %s' % self.__name__
        # TODO: Declare no limit params. Other data to save?
        PokerAction.__init__(self, action_type = CALL_NO_LIMIT, actor_name = actor_name, pot_size = pot_size, 
                             bet_size = this_bet, format=format, chip_stack=chip_stack, bet_this_street=bet_this_street, bet_faced=bet_faced)

class BetNoLimit(PokerAction):
    def __init__(self, actor_name, pot_size, this_bet, format, chip_stack, bet_this_street, bet_faced):
        assert this_bet > 0, 'error calling %s' % self.__name__
        # TODO: Declare no limit params. Other data to save?
        PokerAction.__init__(self, action_type = BET_NO_LIMIT, actor_name = actor_name, pot_size = pot_size, 
                             bet_size = this_bet, format=format, chip_stack=chip_stack, bet_this_street=bet_this_street, bet_faced=bet_faced)

class RaiseNoLimit(PokerAction):
    def __init__(self, actor_name, pot_size, this_bet, format, chip_stack, bet_this_street, bet_faced):
        assert this_bet > 0, 'error calling %s' % self.__name__
        # TODO: Declare no limit params. Other data to save?
        PokerAction.__init__(self, action_type = RAISE_NO_LIMIT, actor_name = actor_name, pot_size = pot_size,
                             bet_size = this_bet, format=format, chip_stack=chip_stack, bet_this_street=bet_this_street, bet_faced=bet_faced)
