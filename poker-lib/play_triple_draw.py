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

"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing heads-up triple-draw poker, with both players simulated by AI.

As such, the game is run by a dealer, who controls actions, a deck, which can be drawn from,
player hands, and player agents... which make decisions, when propted, ultimately by querying an AI system.

The game is hard-coded to implement triple draw. But should be possible, to switch rules,
models and final hand evaluations, to accomodate any other draw game.
"""

# Heuristics, to evaluate hand actions. On 0-1000 scale, where wheel is 1000 points, and bad hand is 50-100 points.
# Meant to map to rough % of winning at showdown. Tuned for ring game, so random hand << 500.
RANDOM_HAND_HEURISTIC_BASELINE = 300 # baseline, before looking at any cards.

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
        return best_hand

# Should inherit from more general player... when we nee one.
class TripleDrawAIPlayer():
    # TODO: Initialize model to use, etc.
    def __init__(self):
        self.draw_hand = None
        # Current 0-1000 value, based on cards held, and approximation of value from draw model.
        # For example, if no more draws... heuristic is actual hand.
        self.heuristic_value = RANDOM_HAND_HEURISTIC_BASELINE 

        # TODO: Use this to track number of cards discarded, etc. Obviously, don't look at opponent's cards.
        self.opponent_hand = None

    # Apply current model, based on known information, to draw 0-5 cards from the deck.
    # TODO: We should log this, and output information useful for tracking.
    def draw(self, deck=None, round=1):
        # Total placeholder. Draw random cards.
        draw_string = ''
        for i in range(0,5):
            if random.random() > 0.50:
                draw_string += '%d' % i

        discards = self.draw_hand.draw(draw_string)
        deck.take_discards(discards)
        new_cards = deck.deal(len(discards))
        self.draw_hand.deal(new_cards, final_hand=True)

    # TODO: This should choose an action policy...
    def choose_action(self, actions, round):
        return self.choose_random_action(actions, round)

    # Nobody said that... some actions can't be more random than others!
    def choose_random_action(self, actions, round):
        if actions:
            random_choice = random.sample(actions, 1)
            # Act here, if we re-sample, to fold less, etc
            return random_choice[0]
        return None


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
            print('since round: %d >= %d, this is big-bet street' % (round, DRAW_2_BET_ROUND))
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
        best_action = self.action_on.choose_action(actions=allowed_actions, round=round)

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
            elif best_action == BET_SMALL_STREET:
                action = BetSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
            elif best_action == RAISE_SMALL_STREET:
                action = RaiseSmallStreet(self.action_on.name, self.pot_size, bet_on_action, max(bet_on_action, bet_off_action))
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
                                                                                             
        
        # Play out a full round of betting.
        # Will go back & forth between players betting, until
        # A. Player calls (instead of raise or fold)
        # B. Player folds (thus concedes the hand)
        self.play_betting_round(round = PRE_DRAW_BET_ROUND)

        print(self.hand_history)

        # Make draws for each player, in turn
        if (self.live):
            # Similar to "player.move()" in the single-draw video poker context
            # NOTE: Player already knows his own hand.
            # TODO: We should also integrate context, like hand history, pot size, opponent's actions.
            self.player_blind.draw(deck=self.deck, round=1)
            self.player_button.draw(deck=self.deck, round=1)

        # TODO: Next round. For now... just evalute these hands & declare a winner.

    # Declare a winner... assuming hand ends now.
    def get_hand_result(self, cashier):
        if self.player_blind.live and not self.player_button.live:
            print('Player blind wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_blind.draw_hand))
        elif not self.player_blind.live and self.player_button.live:
            print('Player button wins by default. %d chips in the pot. %s' % (self.pot_size, self.player_button.draw_hand))
        elif not self.player_blind.live and not self.player_button.live:
            print('Error! both players are dead.')
        else:
            # TODO: Handle ties & split pots!
            best_hand = cashier.showdown([self.player_blind.draw_hand, self.player_button.draw_hand])
            if best_hand == self.player_blind.draw_hand:
                print('Blind player wins on showdown! %d chips in the pot.' % self.pot_size)
            elif best_hand == self.player_button.draw_hand:
                print('Button player wins on showdown! %d chips in the pot.' % self.pot_size)
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

    # We initialize deck, and dealer, every round. But players kept constant, and reset for each trial.
    # NOTE: This can, and will change, if we do repetative simulation, etc.
    player_one = TripleDrawAIPlayer()
    player_two = TripleDrawAIPlayer()


    round = 0
    try:
        while round < sample_size:
            # TODO: Implement human player, switch button.
            game_round(round, cashier, player_button=player_one, player_blind=player_two)

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
