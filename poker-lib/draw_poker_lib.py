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

from draw_poker_action import * # encoding bets, draws, etc [and context in CSV]
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
            
    # Complete a heads-up betting cycle, in a big-bet game (presumably NLH)
    # NOTE: Optionally, supplied bet string. In which case, we need to take the next action in the string, or throw error if impossible.
    def play_big_betting_round(self, round, bet_string=None, debug=False):
        # Check for conditions that must be met, to continue.
        if not(self.player_blind.live and self.player_button.live):
            print('Exiting betting round %d. Since one of the players is not live (folded)' % round)
            return

        # Debug/check if we are playing the right game.
        print('\nplay_big_betting_round %d for game type %s, bet_string |%s|' % (round, self.format, bet_string))

        # If supplied a bet_string, decode next action, and collect remainder (for possible recursive call)
        if bet_string:
            if debug:
                print('processing bet_string for round: |%s|' % bet_string)
            # Parse the string as bet, call, check, or fold
            bet = re.match('\S[0-9]*', bet_string)
            bet_type = bet.group(0)[0]
            bet_amount = bet.group(0)[1:]
            if not bet_amount:
                bet_amount = 0
            else:
                bet_amount = int(bet_amount)

            if debug:
                print('bet of type |%s| and size |%s|' % (bet_type, bet_amount))
            
            # Chop bet_string remainder
            bet_string_remainder = bet_string[bet.span()[1]:]
            if debug:
                print('bet_string remainder: |%s|' % bet_string_remainder)
        else:
            bet_string_remainder = None

        # Determine, if we are facing a raise, can call, etc. 
        bet_on_action = self.action_on.bet_this_street
        bet_off_action = self.action_off.bet_this_street

        # For big bet, we need several values:
        # A. min bet this street (as in a limit game)
        # B. bet we face (if can call)
        # C. min raise (if can call)
        # D. max bet (size of allin)
        # E. TODO: pot size (actually, an important figure)
        # NOTE: By "bet," we mean how much chips, this action will put into the pot. Be that call, raise, etc.

        # A. min bet this street (as in a limit game)
        min_bet_this_street = SMALL_BET_SIZE
        is_small_street = True
        if round >= DRAW_2_BET_ROUND:
            #print('since round: %d >= %d, this is big-bet street' % (round, DRAW_2_BET_ROUND))
            min_bet_this_street = SMALL_BET_SIZE # BIG_BET_SIZE # Can bet 100 into dry pot in NLH
            is_small_street = False

        # D. max bet (size of allin)
        
        # Debug, and see if we need to add more? [Like track last bet, etc]
        print('Player %s to act. He has bet %d (this street) %d (this round)' % (self.action_on.name, 
                                                                                 self.action_on.bet_this_street,
                                                                                 self.action_on.bet_this_hand))
        print('Oppn %s off act. He has bet %d (this street) %d (this round)' % (self.action_off.name, 
                                                                                self.action_off.bet_this_street,
                                                                                self.action_off.bet_this_hand))
        # Use hard-wired stack limit, same for both players. NOTE: Easy to make this per-hand, per-player.
        stack_limit = BIG_BET_FULL_STACK 
        allin_bet = BIG_BET_FULL_STACK - self.action_on.bet_this_hand # max we can put in (allin)
        facing_bet = self.action_off.bet_this_street - self.action_on.bet_this_street # bet to call?
        min_raise = min(allin_bet, facing_bet * 2) # minimum to raise the action
        pot_size = self.action_on.bet_this_hand + self.action_off.bet_this_hand + facing_bet # pot... for PLO uses

        print('Start stack %d\tallin_bet %d\tfacing_bet %d\tmin_raise %d\tpot_size(plo) %d' % 
              (stack_limit, allin_bet, facing_bet, min_raise, pot_size))
        
        # Now, based on situation, collect actions are allowed for active play...
        allowed_actions = set([])
        if bet_on_action > bet_off_action:
            assert bet_on_action <= bet_off_action, ('On action %s, but somehow has bet %d > %d' %
                                                     (self.action_on.name, bet_on_action, bet_off_action))
        elif bet_on_action == bet_off_action:
            # If more betting left, option to check or bet
            if allin_bet >= 0:
                # TODO: How to handle already allin? Treat is as check-down, or special move? I guess check.
                allowed_actions.add(CHECK_HAND)
                if bet_on_action == 0 and allin_bet > 0:
                    allowed_actions.add(BET_NO_LIMIT) # BET_SMALL_STREET if is_small_street else BET_BIG_STREET)
                elif bet_on_action > 0 and allin_bet > 0:
                    # If already put money in, always a raise... BB for example.
                    allowed_actions.add(RAISE_NO_LIMIT) # RAISE_SMALL_STREET if is_small_street else RAISE_BIG_STREET)
        else:
            # If we're facing a bet, always option to call or fold.
            allowed_actions.add(FOLD_HAND)
            allowed_actions.add(CALL_NO_LIMIT) # CALL_SMALL_STREET if is_small_street else CALL_BIG_STREET)

            # If opponent's bet hasn't topped the (limit) max.... we can also raise.
            if min_raise > facing_bet:
                allowed_actions.add(RAISE_NO_LIMIT) # RAISE_SMALL_STREET if is_small_street else RAISE_BIG_STREET)

        # Exit quickly... if there are no actions (thus street is capped out)
        if not allowed_actions:
            print('No more allowed actions! Street must be capped betting.')
            return
        
        # If still here... ther are legal actions that a player may take!
        print('Allowed big-bet actions for player %s [%s]: %s' % (self.action_on.name, self.action_on.player_tag(), [actionName[action] for action in allowed_actions]))
        
        ###########################
        # Ok, now ask an action, from big-bet model. 
        # NOTE: Logically, we want big-bet model to return "bet size" parameter, with single value, if bet/raise is best move.
        # This is a bit messy, but returning distribution and sampling here, is messier still. 
        # Encapsulate the bet-size logic elsewhere. But make bet size facing easy to see.
        ###########################

        # array, just of bet sizes. [0, 200] -> ['check', 'bet 200']
        bets_sequence = [h.bet_size  for h in self.hand_history_this_round]
        print('bets this street -> %s' % bets_sequence)
        
        # Here the agent... would choose a good action.
        # ...unless we have declared a desired "bet_string" from which we already know desired bet_size and action
        if bet_string:
            # Ensure that action is legal, and map it to "best_action"
            assert bet_type, 'Unknown/unparsable next bet_type, for bet_string |%s|' % bet_string
            if bet_type == 'c':
                # Backward compatible. 'c' can mean check or call
                if CALL_NO_LIMIT in allowed_actions:
                    best_action = CALL_NO_LIMIT
                elif CHECK_HAND in allowed_actions:
                    best_action = CHECK_HAND 
                else:
                    assert False, 'impossible situation with bet action. bet_string |%s|' % bet_string
            elif bet_type == 'k':
                best_action = CHECK_HAND
            elif bet_type == 'f':
                best_action = FOLD_HAND
            elif bet_type == 'b' or bet_type == 'r':
                if BET_NO_LIMIT in allowed_actions:
                    best_action = BET_NO_LIMIT
                elif RAISE_NO_LIMIT in allowed_actions:
                    best_action = RAISE_NO_LIMIT
                else:
                    assert False, 'impossible situation with bet action. bet_string |%s|' % bet_string
            else:
                assert False, 'impossible situation with bet action. bet_string |%s|' % bet_string

            # If a bet/raise, bet_amount is a little tricky.
            # Our agents decide *marginal bet size*, while ACPC gives total this street
            if best_action in ALL_BETS_SET:
                # Another backward compatibility issue. pre-2014 ACPC makes bets... in full stack sick. So weird.
                # STATE:28:r250c/cr500r1250c/r5000c/cr20000f:Ac5d|5hKs/5s5cQh/8h/Kh:-5000|5000:tartanian6|slumbot
                if USE_2013_ACPC_BETS_FORMAT:
                    bet_amount = bet_amount - self.action_on.bet_this_hand
                else:
                    bet_amount = bet_amount - bet_on_action
        else:
            best_action, bet_amount = self.action_on.choose_action(actions=allowed_actions, 
                                                                   round=round, 
                                                                   bets_this_round = max(bet_on_action, bet_off_action) / (2.0 * min_bet_this_street),
                                                                   bets_sequence=bets_sequence,
                                                                   chip_stack = allin_bet,
                                                                   has_button = (self.action_on == self.player_button),
                                                                   pot_size=self.pot_size, 
                                                                   actions_this_round=self.hand_history_this_round,
                                                                   actions_whole_hand=self.hand_history,
                                                                   cards_kept=self.action_on.num_cards_kept, 
                                                                   opponent_cards_kept=self.action_off.num_cards_kept)
        # What is the best action?
        # If action returned, complete the action... and keep going
        if (best_action):
            print('\nBest action for nlh chose ->  %s\nSuggested bet amount -> %s\n' % (actionName[best_action], bet_amount))
            # Create the action
            # We keep betting after this action... as long last action allows it.
            keep_betting = True

            # In case we get illegal "bet_amount", just cap it with bounds.
            if best_action in ALL_BETS_SET and (bet_amount <= min_bet_this_street or bet_amount > allin_bet):
                if bet_amount <= min_bet_this_street:
                    print('WARNING: resetting illegal bet |%s| to |%s|' % (bet_amount, max(min_raise, min_bet_this_street)))
                    bet_amount = max(min_raise, min_bet_this_street)
                    
                    if best_action in ALL_RAISES_SET:
                        bet_amount = max(bet_amount, min_bet_this_street * 2)

                    # As a hack, tweak the bet amount semi-randomly.
                    # TODO: Remove before production!!
                    #bet_amount *= 3.0 * np.random.random_sample() + 1.0
                    
                if bet_amount > allin_bet:
                    print('WARNING: resetting illegal bet |%s| to |%s|' % (bet_amount, allin_bet))
                    bet_amount = allin_bet
            bet_amount = math.floor(bet_amount)

            # For ACPC compatibility... we need to encode 3-bet as b100b300b1200 [totals this street]
            # Thus, sum amount we already bet, with this current bet size... 
            bet_faced = bet_off_action - bet_on_action # how much is bet into us?
            bet_this_street = bet_on_action
            # If betting, total amount (this street) is previous bet + new bet
            # If calling, previous bet + opponent's bet
            # Else, we are folding, etc, so it's previous bet
            if best_action in ALL_BETS_SET:
                bet_this_street += bet_amount
            elif best_action in ALL_CALLS_SET:
                bet_this_street += bet_faced
            
            if best_action in ALL_CALLS_SET:
                #action = CallSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action), format=self.format)
                action = CallNoLimit(self.action_on.name, self.pot_size, bet_faced, format=self.format, chip_stack = allin_bet, bet_this_street=bet_this_street, bet_faced=bet_faced)
                if not(round == PRE_DRAW_BET_ROUND and bet_off_action == BIG_BLIND_SIZE):
                    print('\nchosen CALL action, closes the action')
                    keep_betting = False
            elif best_action in ALL_RAISES_SET:
                action = RaiseNoLimit(self.action_on.name, self.pot_size, bet_amount, format=self.format, chip_stack = allin_bet, bet_this_street=bet_this_street, bet_faced=bet_faced)
            elif best_action in ALL_BETS_SET:
                action = BetNoLimit(self.action_on.name, self.pot_size, bet_amount, format=self.format, chip_stack = allin_bet, bet_this_street=bet_this_street, bet_faced=bet_faced)
            elif best_action == FOLD_HAND:
                action = FoldStreet(self.action_on.name, self.pot_size, format=self.format, chip_stack = allin_bet, bet_this_street=bet_this_street, bet_faced=bet_faced)
                print('\nchosen FOLD action, closes the action')
                keep_betting = False
            elif best_action == CHECK_HAND:
                action = CheckStreet(self.action_on.name, self.pot_size, format=self.format, chip_stack = allin_bet, bet_this_street=bet_this_street, bet_faced=bet_faced)

                # Logic for checking is a bit tricky. Action ended if button checks... except on first, when F player checks ends it.
                if (round == PRE_DRAW_BET_ROUND and self.action_on == self.player_blind) or (round != PRE_DRAW_BET_ROUND and self.action_on == self.player_button):
                    print('\nchosen CHECK action, closes the action')
                    keep_betting = False
                else:
                    print('\nthis check... does not end the action.')
            else:
                assert False, 'Unknown best_action %s' % actionName[best_action]

            # HACK: Use "best_draw" and "hand_after" fields, to record the flop, turn and river in Holdem games.
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

            # TODO: in 'bet_string' situation... check that we keep_betting and don't keep_betting as string requires.
            # NOTE: Special case for allin! Handled with check-down or hand just ends??
            if keep_betting:
                #print('chosen action, allows further betting.')
                self.play_betting_round(round, bet_string=bet_string_remainder)

    # Play full betting round on a loop... until action ends.
    def play_betting_round(self, round, bet_string = None):
        # Split out for big-bet game... since logic is different! [Even if we call same decision-makers for now!]
        # NOTE: Why not merge with if/thens? Because logic for limit works, so let's not mess it up. 
        if self.format == 'nlh':
            self.play_big_betting_round(round, bet_string=bet_string)

            # Once play_big_betting_round works... just return control.
            return

        # Check for conditions that must be met, to continue.
        if not(self.player_blind.live and self.player_button.live):
            print('Exiting betting round %d. Since one of the players is not live (folded)' % round)
            return
        
        # Debug/check if we are playing the right game.
        print('play_betting_round %d for game type %s' % (round, self.format))

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

            # If opponent's bet hasn't topped the (limit) max.... we can also raise.
            if bet_off_action < max_bet:
                allowed_actions.add(RAISE_SMALL_STREET if is_small_street else RAISE_BIG_STREET)

        # Exit quickly... if there are no actions (thus street is capped out)
        if not allowed_actions:
            print('No more allowed actions! Street must be capped betting.')
            return
        
        # If still here... ther are legal actions that a player may take!
        print('Allowed actions for player %s: %s' % (self.action_on.name, [actionName[action] for action in allowed_actions]))

        # Here the agent... would choose a good action.
        # NOTE: Now tuned for NLH. Need to test for non-big bet games (ignore bet amount, etc)
        allin_bet = BIG_BET_FULL_STACK - self.action_on.bet_this_hand # max we can put in (allin) 
        best_action, bet_amount = self.action_on.choose_action(actions=allowed_actions, 
                                                               round=round, 
                                                               bets_this_round = max(bet_on_action, bet_off_action) / bet_this_street,
                                                               chip_stack = allin_bet,
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
            if self.format == 'holdem' or self.format == 'nlh':
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
    # NOTE: We can also supply "bets_string" to declare player bets (re-create from logs). NLH only!
    def play_single_hand(self, bets_string = None):
        # If community card game, create blank community object first.
        if self.format == 'holdem' or self.format == 'nlh':
            self.community = HoldemCommunityHand()
        else:
            self.community = None

        # If supplied bets string, chop into bets by street.
        street_bets_array = [None, None, None, None] # default == no bets supplied
        if bets_string:
            assert self.format == 'nlh', 'bets_string for play_single_hand supported for NLH games only!'
            street_bets_array = bets_string.split('/')
            print('bets_string (%d rounds) supplied: %s' % (len(street_bets_array), street_bets_array))

        # Deal initial hands to players
        # NOTE: Nice thing with hold'em hands... w're done with player hands, except to update shared community cards.
        if self.format == 'holdem' or self.format == 'nlh':
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

        if self.format == 'holdem' or self.format == 'nlh':
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
        self.play_betting_round(round = PRE_DRAW_BET_ROUND, bet_string = (street_bets_array[0] if len(street_bets_array) > 0 else None))

        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if not(self.format == 'holdem' or self.format == 'nlh'):
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
        if self.format == 'holdem' or self.format == 'nlh':
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

        self.play_betting_round(round = DRAW_1_BET_ROUND, bet_string = (street_bets_array[1] if len(street_bets_array) > 1 else None))
        
        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if not(self.format == 'holdem' or self.format == 'nlh'):
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
        if self.format == 'holdem' or self.format == 'nlh':
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

        self.play_betting_round(round = DRAW_2_BET_ROUND, bet_string = (street_bets_array[2] if len(street_bets_array) > 2 else None))
        
        # print(self.hand_history)

        if not(self.player_blind.live and self.player_button.live):
            return

        # Make draws for each player, in turn
        if not(self.format == 'holdem' or self.format == 'nlh'):
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
        if self.format == 'holdem' or self.format == 'nlh':
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

        self.play_betting_round(round = DRAW_3_BET_ROUND, bet_string = (street_bets_array[3] if len(street_bets_array) > 3 else None))
        
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

