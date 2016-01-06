import io, re, socket, sys
import math
import time
import argparse # command line arguements parsing
import os.path # checking file existence, etc
import numpy as np
import scipy.stats as ss
import lasagne
import theano
import theano.tensor as T

# Functions to interpret hand, run neurnal network lookups.
# NOTE: This will be *way slow* by ACPC metrics. Need to change timeouts in the dealer code, to make this work.
from poker_lib import *
from holdem_lib import * # if we want to support Holdem hands and game sim
from poker_util import *

# Import only the draw-functions that we need.
from draw_poker import holdem_cards_input_from_string
from triple_draw_poker_full_output import build_model
from triple_draw_poker_full_output import predict_model # outputs result for [BATCH x data]
from triple_draw_poker_full_output import evaluate_single_hand # single hand... returns 32-point vector
from triple_draw_poker_full_output import evaluate_single_event # just give it the 26x17x17 bits... and get a vector back
from triple_draw_poker_full_output import evaluate_single_holdem_hand # for a holdem hand, returns 0-1.0 value vs random, and some odds
from triple_draw_poker_full_output import expand_parameters_input_to_match # expand older models, to work on larger input (just zero-fill layer)

# We can't just import all of play_triple_draw
from play_triple_draw import TripleDrawAIPlayer

parser = argparse.ArgumentParser(description='Play heads-up triple draw against a convolutional network. Or see two networks battle it out.')

# To connect with the server!
parser.add_argument('-address', '--address', default='localhost', help='ACPC server/dealer address')
parser.add_argument('-port', '--port', default='48777', help='ACPC server/dealer PORT')
# ------------------------------------------------------------------------------------
parser.add_argument('-draw_model', '--draw_model', default=None, help='neural net model for draws, or simulate betting if no bet model') # draws, from 32-length array
parser.add_argument('-holdem_model', '--holdem_model', default=None, help='neural net model for Holdem hands, with first value 0-1.0 value vs random hand, other values the odds of making specific hand types. Baseline value for any valid hand and flop, turn or river')
parser.add_argument('-CNN_model', '--CNN_model', default=None, help='neural net model for betting') # Optional CNN model. If not supplied, uses draw model to "sim" decent play
parser.add_argument('-output', '--output', help='output CSV') # CSV output file, in append mode.
parser.add_argument('--human_player', action='store_true', help='pass for p2 = human player') # Declare if we want a human competitor? (as player_2)
parser.add_argument('-CNN_old_model', '--CNN_old_model', default=None, help='pass for p2 = old model (or second model)') # useful, if we want to test two CNN models against each other.
parser.add_argument('-CNN_other_old_model', '--CNN_other_old_model', default=None, help='pass for p2 = other old model (or 3rd model)') # and a third model, 
args = parser.parse_args()

# For testing CNN models (Holdem only)
BATCH_SIZE = 100 # Across all cases
FORMAT = 'nlh'
test_cases = [['Ad,Ac', '[]', '', ''], # AA preflop
              ['[8d,5h]','[Qh,9d,3d]','[Ad]','[7c]'], # missed draw
              ['4d,5d', '[6d,7d,3c]', 'Ad', ''], # made flush
              ['7c,9h', '[8s,6c,Qh]', '', ''], # open ended straight draw
              ['Ad,Qd', '[Kd,Td,2s]', '3s', ''], # big draw
              ['7s,2h', '', '', ''], # weak hand preflop
              ['Ts,Th', '', '', ''], # good hand preflop
              ['9s,Qh', '', '', ''], # average hand preflop
              ]
for i in range(BATCH_SIZE - len(test_cases)):
    test_cases.append(test_cases[1])
test_batch = np.array([holdem_cards_input_from_string(case[0], case[1], case[2], case[3]) for case in test_cases], np.int32)

"""
Sample client to play ACPC Limit Holdem in Python!
"""

#player  = kuhn3p.players.Chump(1, 1, 1)
player  = None # no player yet, need to initialize...
address = args.address # 'localhost' # sys.argv[1] 
port    = args.port # 48777 #16177 # 48777 # int(sys.argv[2]) 16177

# Given something like "r302c/r465c/", 
# A. return string with per-round numbers adjusted
# B. return bet size progression
def reformat_bets_string(bets_string):
    bet_size_progression = [50, 100]
    bet_progression_this_round = [100, 50]
    bets_string_reformat = ''
    street_bets_array = bets_string.split('/')
    previous_round_bet = 0

    for street_bets in street_bets_array:
        # Add '/' if new row
        if bets_string_reformat:
            bets_string_reformat += '/'
            previous_round_bet = bet_size_progression[-1]
            bet_progression_this_round = []

        for bet in re.finditer('\S[0-9]*', street_bets):
            #print(bet.span(), bet.group(0))
            # Each bet should be encoded, in turn.
            bet_type = bet.group(0)[0]
            bets_string_reformat += bet_type
            bet_amount = bet.group(0)[1:]
            if not bet_amount:
                bet_amount = 0
            else:
                bet_amount_this_street = int(bet_amount) - previous_round_bet
                bets_string_reformat += '%s' % bet_amount_this_street
                bet_size_progression.append(int(bet_amount))
                
                # bets made this round...
                # NOTE: Very tricky. We expect this to be the order of bet sizes made. 
                # So "b200" prelop would be [100, 50, 150] and then "b200b500" continues [100, 50, 150, 400]
                #already_bet_this_street = 0
                if len(bet_progression_this_round) == 2 and bet_progression_this_round[0:2] == [100, 50]:
                    already_bet_this_street = bet_progression_this_round[1] # $50 small blind
                elif len(bet_progression_this_round) >=2:
                    already_bet_this_street = bet_progression_this_round[-2] # two bets ago
                else:
                    already_bet_this_street = 0
                bet_progression_this_round.append(bet_amount_this_street - already_bet_this_street)

    # Do we end with a '/'?
    if bets_string and bets_string[-1] == '/':
        #bets_string_reformat += '/'
        previous_round_bet = bet_size_progression[-1]
        bet_progression_this_round = []

    print([bets_string_reformat, bet_size_progression, previous_round_bet, bet_progression_this_round])
    return (bets_string_reformat, bet_size_progression, previous_round_bet, bet_progression_this_round)

# Load model for Holdem heuristic player.
holdem_model_filename = None
holdem_output_layer = None
holdem_input_layer = None
if args:
    holdem_model_filename = args.holdem_model
    bets_model_filename = args.CNN_model

    print('Attempting to read holdem_model from %s' % holdem_model_filename)
    if holdem_model_filename and os.path.isfile(holdem_model_filename):
        print('\nExisting holdem model in file %s. Attempt to load it!\n' % holdem_model_filename)
        all_param_values_from_file = np.load(holdem_model_filename)
        expand_parameters_input_to_match(all_param_values_from_file, zero_fill = True)

        for layer_param in all_param_values_from_file:
            print(layer_param)
            print(layer_param.shape)
            print('---------------')

        # Size must match exactly!
        holdem_output_layer, holdem_input_layer, holdem_layers  = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )

        #print('filling model with shape %s, with %d params' % (str(output_layer.get_output_shape()), len(all_param_values_from_file)))
        lasagne.layers.set_all_param_values(holdem_output_layer, all_param_values_from_file)

        predict_model(output_layer=holdem_output_layer, test_batch=test_batch, format = FORMAT)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled model...')
    else:
        print('No model provided or loaded. Expect error if model required. %s', holdem_model_filename)

    # If supplied, also load the bets model. conv(xCards + xNumDraws + xContext) --> values for all betting actions
    bets_output_layer = None
    bets_input_layer = None
    print('Attempting to read CNN_model from %s' % bets_model_filename)
    if bets_model_filename and os.path.isfile(bets_model_filename):
        print('\nExisting *bets* model in file %s. Attempt to load it!\n' % bets_model_filename)
        bets_all_param_values_from_file = np.load(bets_model_filename)
        expand_parameters_input_to_match(bets_all_param_values_from_file, zero_fill = True)

        # Size must match exactly!
        bets_output_layer, bets_input_layer, bets_layers  = build_model(
            HAND_TO_MATRIX_PAD_SIZE, 
            HAND_TO_MATRIX_PAD_SIZE,
            32,
        )

        #print('filling model with shape %s, with %d params' % (str(bets_output_layer.get_output_shape()), len(bets_all_param_values_from_file)))
        lasagne.layers.set_all_param_values(bets_output_layer, bets_all_param_values_from_file)
        predict_model(output_layer=bets_output_layer, test_batch=test_batch)
        print('Cases again %s' % str(test_cases))
        print('Creating player, based on this pickled *bets* model...')
    else:
        print('No *bets* model provided or loaded. Expect error if model required. %s', bets_model_filename)

    #sys.exit(-3)

# Create neural network AI player.
player_one = TripleDrawAIPlayer()

# If holdem layer exists, this will be good enough.
player_one.holdem_output_layer = holdem_output_layer
player_one.holdem_input_layer = holdem_input_layer
player_one.bets_output_layer = bets_output_layer
player_one.bets_input_layer = bets_input_layer

# enable, to make betting decisions with learned model (instead of heurstics)
if bets_output_layer:
    player_one.use_learning_action_model = True

# Begin talking to the protocol...
sock    = socket.create_connection((address, port))
sockin  = sock.makefile(mode='rb')

sock.send('VERSION:2.0.0\r\n')

#state_regex = re.compile(r"MATCHSTATE:(\d):(\d+):([^:]*):([^|]*)\|([^|]*)\|(.*)")

# MATCHSTATE:1:0::|9hQd
matchstate_string = "MATCHSTATE:(\d*):(\d*):([^:]*)"
state_regex = re.compile(r"MATCHSTATE:(\d*):(\d*):([^:]*):([^|]*)\|([^|/]*)/?([^|/]*)/?([^|/]*)/?([^|/]*)")


position = None
hand     = None
now = time.time()
while 1:
    line = sockin.readline().strip()

    if not line:
        break


    print('\n-------------------')

    print line

    state = state_regex.match(line)
    assert state, 'Unable to regexp from server: |%s|' % line
   
    position, this_hand = map(lambda x: int(x), state.group(1, 2))
    betting                  = state.group(3)
    cards                    = map(card_array_from_string, state.group(4, 5, 6, 7, 8))

    print([position, this_hand, betting])
    if this_hand:
        time_diff = time.time() - now
        print('Took %.1f sec to process %d hands (%.2f average)' % (time_diff, this_hand, time_diff/this_hand))

    # Parse the cards. What is flop, turn, and river?
    oop_hand = cards[0]
    pos_hand = cards[1]
    flop = cards[2]
    turn = cards[3]
    river = cards[4]

    #print([oop_hand, pos_hand, flop, turn, river])
    print([hand_string(x) for x in [oop_hand, pos_hand, flop, turn, river]])

    # Now turn this hand into our hand, and our community cards.
    community = HoldemCommunityHand(flop=flop, turn=turn, river=river)
    if position:
        hand = HoldemHand(cards=pos_hand, community=community)
    else:
        hand = HoldemHand(cards=oop_hand, community=community)

    # Ok, we got the f-ing hand. Now just take a random action.

    # TODO: Count positions, # of bets/raises, to determine if it's our move to act?

    # Break up the betting into rounds
    rounds_left = 3
    bets_this_round = 0
    bets = ['']
    folded = False # Opponent folded

    print('parsing betting string: |%s|' % betting)

    all_bets_string = betting

    for char in betting:
        #print(char)
        if char == 'c':
            bets[-1] += char
        elif char == 'r':
            bets_this_round += 1
            bets[-1] += char
        elif char == '/':
            rounds_left -= 1
            bets.append('')
            bets_this_round = 0
        elif char == 'f':
            bets[-1] += char
            folded = True
        #else:
        #   bets[-1] += char
            #print('skipping numerical character |%s|' % char)
            #assert False, 'Unknown character %s' % char

    # Go through the string, round by round, and re-format it for 
    bets_string_adjusted, bets_progression, previous_round_bet, bet_sequence_this_round = reformat_bets_string(betting)

    # Now go through bets and compute pot size.
    pot_size = 0.0
    print(bets)
    # Special case for $150 pot to start (before we call a blind)
    if rounds_left == 3 and bets[-1] == '':
        pot_size = BIG_BLIND_SIZE + SMALL_BLIND_SIZE
        print('No bets yet, so pot %.1f' % pot_size)

        # Collect or encode other information
        chip_stack = MAX_STACK_SIZE - SMALL_BLIND_SIZE
    else:
        # TODO: pot_size --> Track both player size

        # TODO: Subtract out uncalled bet... if pending.
        pot_size = 2 * bets_progression[-1]
        chip_stack = MAX_STACK_SIZE - bets_progression[-1] 
        if betting and betting[-1].isdigit():
            chip_pending = bets_progression[-1] - bets_progression[-2]
            pot_size -= chip_pending
            chip_stack += chip_pending
        
    
        

    print([position, rounds_left, bets_this_round, bets, pot_size, folded])

    # Normally, it's our action. However, sometimes it is not (shown dealt cards, flop or turn... before our turn to act
    our_action = True
    actions = []
    
    # A few cases... (assuming that dealer makes no mistakes)
    # A. No bets this round --> ''. Are we in position to bet first?
    # B. Facing a bet. Are we allowed to raise?
    # C. Hand is over. 
    if len(cards) >= 2 and cards[0] and cards[1]:
        print('--> showdown! %s vs %s' % (hand_string(cards[0]), hand_string(cards[1])))
        # Showdown. We get to see opponent hand.
        our_action = False
    elif bets[-1] and bets[-1][-1] == 'f':
        our_action = False
        actions = []
    elif bets[-1] == '' and rounds_left == 3:
        if not position:
            our_action = False
        else:
            actions = ['r', 'c', 'f']
    elif bets[-1] == '':
        if position:
            our_action = False
        else:
            actions = ['r', 'c']
    elif bets[-1] and bets[-1][-1] == 'c':
        # TODO: Check # of actions this round. Odd or even? Should match position = 
        num_acts = len(bets[-1])
        if rounds_left == 3:
            num_acts += 1
        if num_acts % 2 == position:
            actions = ['r', 'c']
        else:
            our_action = False
    elif bets[-1] and bets[-1][-1] == 'r':
        # TODO: Check # of actions this round. Odd or even? Should match position = 
        num_acts = len(bets[-1])
        if rounds_left == 3:
            num_acts += 1

        print('num_acts: %s' % num_acts)
        print('position: %d' % position)

        if num_acts % 2 == position:
            actions = ['r', 'c']
        else:
            our_action = False

        if our_action and ((rounds_left < 3) or (rounds_left == 3)):
            actions = ['r', 'c', 'f']
        else:
            actions = ['c', 'f']
    else:
        print(bets)
        assert False, 'Unknown game state.'
            
    print(actions)


    # MATCHSTATE:0:998:r:Kc6h|:c


    # Now re-map actions to FOLD_HAND, etc
    action_enums = []
    for action in actions:
        if action == 'f':
            action_enums.append(FOLD_HAND)
        elif action == 'r' and not turn:
            if ('f' not in set(actions)) and not (rounds_left == 3 and (bets[-1] == '0' or bets[-1] == 'c')):
                action_enums.append(BET_NO_LIMIT)
            else:
                action_enums.append(RAISE_NO_LIMIT)
        elif action == 'r' and turn:
            if 'f' not in set(actions):
                action_enums.append(BET_NO_LIMIT)
            else:
                action_enums.append(RAISE_NO_LIMIT)
        elif action == 'c' and ('f' not in set(actions)):
            action_enums.append(CHECK_HAND)
        elif action == 'c' and ('f' in set(actions)):
            if not turn:
                action_enums.append(CALL_NO_LIMIT)
            else:
                action_enums.append(CALL_NO_LIMIT)
        else:
            assert False, 'Unknown action |%s|' % action

    print(action_enums)

    
    # Split up bet strings
    street_bets_array = bets_string_adjusted.split('/')
    if street_bets_array:
        current_round_bets_string = street_bets_array[-1]
    else:
        current_round_bets_string = ''
    if all_bets_string and all_bets_string[-1] == '/':
        current_round_bets_string = ''

    # Choose action with AI player.
    if our_action:
        player_one.holdem_hand = hand
        player_one.imitate_CFR_betting = True # TODO flip coin to decide rate?

        # Use CNN player. But this means translating actions to 0/1 strings, and computing the size of the pot...
        # NOTE: It's ridiculous that dealer won't give us pot size. (or winners, or hand results, for that matter)
        if player_one.use_learning_action_model:
            AI_action, bet_size =  player_one.choose_action(actions = action_enums, 
                                                            round = community.round, 
                                                            # bets_this_round = bets_this_round, 
                                                            bets_sequence = bet_sequence_this_round,
                                                            bets_this_round = bet_sequence_this_round,
                                                            chip_stack = chip_stack,
                                                            has_button = (position == '1' or position == 1), 
                                                            pot_size=pot_size, 
                                                            actions_this_round= current_round_bets_string, # bets[-1], 
                                                            actions_whole_hand= bets_string_adjusted ) #all_bets_string) # ''.join(bets))

            print(AI_action, bet_size)

        else:
            print('We need learning_action model. Else, what is the point?')
            sys.exit(-3)

        print('suggested AI_action: %s' % actionName[AI_action])

        # Translate something like '304' to usable action
        if AI_action == FOLD_HAND:
            readable_AI_action = 'f'
        elif AI_action == CHECK_HAND:
            readable_AI_action = 'c'
        elif AI_action in ALL_CALLS_SET:
            readable_AI_action = 'c'
        elif AI_action in ALL_BETS_SET:
            bet_this_hand = BIG_BET_FULL_STACK - chip_stack
            readable_AI_action = 'r%d' % (bet_size + bet_this_hand)
        else:
            #assert False, 'Unreadable action! %s' % actionName[AI_action]
            readable_AI_action = 'c'

    # TODO: Take account of the flop, turn and river. Bet 100% if our hand is good.

    if our_action:
        # response = '%s:%s\r\n' % (line, best_action) # random action
        response = '%s:%s\r\n' % (line, readable_AI_action)
        sock.send(response)
    else:
        print('--> Not our move to respond. TODO: examine & record this action')

