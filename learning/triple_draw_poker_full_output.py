from __future__ import print_function

import lasagne
import theano
import theano.tensor as T
import theano.printing as tp # theano.printing.pprint(variable) [tp.pprint(var)]
import time
import pickle
import os.path # checking file existence, etc
import numpy as np

from poker_lib import *
from holdem_lib import *
from draw_poker import _load_poker_csv
from draw_poker import cards_input_from_string
from draw_poker import holdem_cards_input_from_string
from draw_poker import create_iter_functions
from draw_poker import train

"""
Use similar network... to learn triple draw poker!!

First, need new data import functins.
"""

TRAINING_FORMAT = 'holdem_events' # 'holdem' # 'deuce_events' # 'deuce' # 'video'
# fat model == 5x5 bottom layer, and remove a maxpool. Better visualization?
USE_FAT_MODEL = False # True # False # True
USE_FULLY_CONNECTED_MODEL = False # True # False
# TODO: Is leaky units compatible with normal ReLu? Old models won't be 100% correct... but can we load them?
DEFAULT_LEAKY_UNITS = True # False # Use leaky ReLU to avoid saturation at 0.0? [default leakiness 0.01]

DATA_FILENAME = None
if TRAINING_FORMAT == 'deuce_events':
    DATA_FILENAME = '../data/100k_hands_triple_draw_events.csv' # 'deuce_events' 4M hands, of the latest model (and some human play)
elif TRAINING_FORMAT == 'holdem_events':
    DATA_FILENAME = '../data/holdem/100k_CNN_holdem_hands.csv' # 'holdem_events' trained on actually CNN hands (at least some poker ability)
elif TRAINING_FORMAT == 'holdem':
    DATA_FILENAME = '../data/holdem/500k_holdem_values.csv' # 'holdem' 500k holdem hand values. Cards, possible flop, turn and river.
elif TRAINING_FORMAT == 'deuce':
    DATA_FILENAME = '../data/500k_hands_sample_details_all.csv' # all 32 values for 'deuce' (draws)
elif TRAINING_FORMAT == 'video':
    DATA_FILENAME = '../data/250k_full_sim_combined.csv' # 250k hands (exactly) for 32-item sim for video poker (Jacks or better) [from April]

MAX_INPUT_SIZE = 740000 # 700000 # 110000 # 120000 # 10000000 # Remove this constraint, as needed
VALIDATION_SIZE = 40000
TEST_SIZE = 0 # 5000
NUM_EPOCHS = 20 # 100 # 100 # 20 # 50 # 100 # 500
BATCH_SIZE = 100 # 50 #100
BORDER_SHAPE = "valid" # "full" = pads to prev shape "valid" = shrinks [bad for small input sizes]
NUM_FILTERS = 24 # 16 # 32 # 16 # increases 2x at higher level
NUM_HIDDEN_UNITS = 1024 # 512 # 256 #512
LEARNING_RATE = 0.02 # 0.1 # 0.02 # 0.1 # 0.05
MOMENTUM = 0.9
# Fix and test, before epoch switch...
EPOCH_SWITCH_ADAPT = 20 # 12 # 10 # 30 # switch to adaptive training after X epochs of learning rate & momentum with Nesterov

# Model initialization if we use adaptive learning (to start, to to switch to...)
ADA_DELTA_LEARNING_RATE = 1.0 # 0.001 #  # 1.0 learning rate for AdaDelta... 0.01 recommended for RMSProp (very sensitive)
ADA_DELTA_RHO = 0.9 # 0.95 recommended from the AdaDelta paper, 0.9 for RMSprop
ADA_DELTA_EPSILON = 1e-6 # 1e-4 # 1e-6 # default from the paper (MNIST dataset) is small. We can be more aggressive... if data is not noisy (but it's really just a constant)

# Default to adaptive learning rate. Recommended to train with fixed learning rate first. Don't do adaptive on clean model (too noisy)
DEFAULT_ADAPTIVE = False # True # Set on to train adapative. 
ADAPTIVE_USE_RMSPROP = False # If use adaaptive, use RMSprop instead of AdaDelta?
if ADAPTIVE_USE_RMSPROP:
    ADA_DELTA_LEARNING_RATE = 0.001 # needs tiny learning rate for RMSprop, else gradient goes crazy

NUM_FAT_FILTERS = NUM_FILTERS / 2
if USE_FAT_MODEL:
    NUM_FILTERS = NUM_FAT_FILTERS

if (USE_FAT_MODEL or USE_FULLY_CONNECTED_MODEL) and not (TRAINING_FORMAT == 'deuce_events' or TRAINING_FORMAT == 'deuce'):
    LEARNING_RATE = 0.1 # *= 5

# Here, we get into growing input information, beyond the 5-card hand.
INCLUDE_NUM_DRAWS = True # 3 "cards" to encode number of draws left. ex. 2 draws: [0], [1], [1]
INCLUDE_FULL_HAND = True # add 6th "card", including all 5-card hand... in a single matrix [Good for detecting str8, pair, etc]

CONTEXT_LENGTH = 2 + 5 + 5 + 5 + 5 # Fast way to see how many zero's to add, if needed. [xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept, xPreviousRoundBetting]
FULL_INPUT_LENGTH = 5 + 1 + 3 + CONTEXT_LENGTH

#######################
## HACK for 'video' ###
if TRAINING_FORMAT == 'video':
    LEARNING_RATE = 0.1
    CONTEXT_LENGTH = 0
    FULL_INPUT_LENGTH = 5 + CONTEXT_LENGTH

# Do we use linear loss? Why? If cases uncertain or small sample, might be better to approximate the average...
LINEAR_LOSS_FOR_MASKED_OBJECTIVE = False # True # False # True

# If we are trainging on poker events (bets, raises and folds) instead of draw value,
# input and output shape will be the same. But the way it's uses is totally different. 
# NOTE: We keep shape the same... so we can use the really good "draw value" model as initialization.
INCLUDE_HAND_CONTEXT = True # False 17 or so extra "bits" of context. Could be set, could be zero'ed out.
DISABLE_EVENTS_EPOCH_SWITCH = True # False # Is system stable enough, to switch to adaptive training?

# Train on masked objective? 
# NOTE: load_data needs to be already producing such masks.
# NOTE: Default mask isn't all 1's... it's best_output == 1, others == 0
# WARINING: Theano will complain if we pass a mask and don't use it!
TRAIN_MASKED_OBJECTIVE = True
if not(TRAINING_FORMAT == 'deuce_events' or TRAINING_FORMAT == 'holdem_events'):
    TRAIN_MASKED_OBJECTIVE = False

# Helps speed up inputs?
TRAINING_INPUT_TYPE = theano.config.floatX # np.int32
# Question: does deterministic/not deterministic create errors with "RuntimeError: GpuCorrMM failed to allocate working memory of 234 x 225" after a while?
DETERMINISTIC_MODEL_RUN = True # False # True # Do we evaluate bet/raise/draw model in deterministic mode? (dropout, etc)

# HACK linear error
def linear_error(x, t):
    return abs(x - t)

# Hack a more complex error function.
# A. must return same shape matrix as input...
# B. but frist 5 rows, straight difference
# C. next five rows, action*value
# D. special row = (sum of actions) - 1.0
# E. special row = (value/action sum)

first_five_vector = np.zeros(32)
only_first_five_vector = np.zeros(32)
only_first_five_with_actions_vector = np.zeros(32) # First five, and action%
only_second_five_vector = np.zeros(32) # only draws
only_holdem_category_vector = np.zeros(32) # only valid categories for holdem hands
for i in ALL_ACTION_CATEGORY_SET: 
    first_five_vector[i] = 1.0
    only_first_five_vector[i] = 1.0
    only_first_five_with_actions_vector[i] = 1.0
for i in ALL_ACTION_PERCENT_CATEGORY_SET:
    only_first_five_with_actions_vector[i] = 1.0
for i in DRAW_CATEGORY_SET:
    first_five_vector[i] = 1.0
    only_second_five_vector[i] = 1.0
for i in range(1, len(HOLDEM_VALUE_KEYS)):
    only_holdem_category_vector[i] = 1.0
first_five_matrix = [first_five_vector for i in xrange(BATCH_SIZE)]
first_five_mask = np.array(first_five_matrix)
only_first_five_matrix = [only_first_five_vector for i in xrange(BATCH_SIZE)]
only_first_five_mask = np.array(only_first_five_matrix)
only_holdem_category_matrix = [only_holdem_category_vector for i in xrange(BATCH_SIZE)]
only_holdem_category_mask =  np.array(only_holdem_category_matrix)

print('first_five_mask:')
print(first_five_mask)

# Depending on your game type, default mask where 1 = category that matters (for prediction debug) 0 = non-category
if TRAINING_FORMAT == 'deuce_events':
    OUTPUT_CATEGORY_DEFAULT_MASK = first_five_mask
elif TRAINING_FORMAT == 'holdem':
    OUTPUT_CATEGORY_DEFAULT_MASK = only_holdem_category_mask
else:
    OUTPUT_CATEGORY_DEFAULT_MASK = np.ones((32,100))


# Automatically computes the correct mask, from target matrix (see what values exist for data)
def set_mask_at_row_from_target(row_num, output_matrix, target_matrix):
    zeros = T.zeros_like(output_matrix)
    zeros_subtensor = zeros[row_num[0],:] # row in 2D matrix
    # Do we have any (target) values for bets?
    target_bets_row_sum = (target_matrix[row_num[0],:] * only_first_five_vector).sum()
    # If so, choose "first_five" vector. Otherwise, "second_five" will do nicely, thank you.
    #row_value = T.switch(T.gt(target_bets_row_sum, 0.0), only_first_five_vector, only_second_five_vector)
    row_value = T.switch(T.gt(target_bets_row_sum, 0.0), only_first_five_with_actions_vector, only_second_five_vector)
    return T.set_subtensor(zeros_subtensor, row_value) 

# For row with output value and (sparse) matrix values, combine (output) + X (real value) [X is a learning rate]
# NOTE: Would be better to connect this to true RL, and then can increase observation significantly.
TARGET_ACTION_LEARNING_RATE = 0.1 # 0.05
def set_values_at_row_from_target(row_num, output_rows, target_rows):
    zeros = T.zeros_like(target_rows)
    zeros_subtensor = zeros[row_num[0],:] # row in 2D matrix
    output_row = output_rows[row_num[0]]
    target_row = target_rows[row_num[0]]
    target_mask = T.ceil(0.05 * target_row)
    output_mask = T.ones_like(target_mask) - target_mask
    
    # Learning rate should be 0.05? 0.1? No point making it too low, since cases we care about 
    # are typically -200 (big bet) for bad call. Even at 0.1 rate, that's 0.02 change in underlying value...
    average_row = (1.0 - TARGET_ACTION_LEARNING_RATE) * output_row + (TARGET_ACTION_LEARNING_RATE) * (target_row * target_mask  + output_row * output_mask)
    return T.set_subtensor(zeros_subtensor, average_row)

# Can we at least use outside mask??
INCREASE_VALUES_SUM_INVERSE = 1.0 # default 1.0 (1/3 or so)
def value_action_error(output_matrix, target_matrix):
    # Compute a mask... which per-row takes 5 bits of values, if target matrix has values,
    # and the five (six) bits of draws if target matrix is all about draws. 
    # Why? We don't want to action% model (for bets) on inputs to num_draws output.
    category_mask_results, mask_updates = theano.scan(fn=set_mask_at_row_from_target,
                                                      outputs_info=None,
                                                      sequences=[np.asarray([[index, 0] for index in range(BATCH_SIZE)] , dtype=np.int32)],
                                                      non_sequences=[np.zeros((BATCH_SIZE,32), dtype=np.float32), target_matrix])
    output_category_mask = category_mask_results.sum(axis=1)

    # Apply mask to values that matter.
    output_matrix_masked = output_matrix * first_five_mask 
    
    # Now, use matrix manipulation to tease out the current action vector, and current value vector.
    # All values should be positive. Or a hard zero.
    # NOTE: Take the absolute value of everthing. Why? Just in case! If we used leaky ReLU, etc... values could get negative, leading to divide by zero.
    action_matrix = abs(output_matrix[:,5:10]) # Take action % from output matrix. It's all we got!
    value_matrix_output = abs(theano.gradient.disconnected_grad(output_matrix[:,0:5])) # disconnect gradient -- so values aren't changed by action% model
    value_matrix_target = abs(target_matrix[:,0:5]) # directly from observation
    
    #value_matrix = value_matrix_output # Try to learn values from the network. 
    #value_matrix = value_matrix_target # Use real values. Doesn't work, since too much bouncing around. Learns conservative moves (folds alot)

    # Alternatively, learn a mix of network values, and real results (with a learning rate)
    # This will push the action % a little bit into adapting from real values, but not enough that we over-write predicted values, which are good
    average_value_matrix_results, average_value_updates = theano.scan(fn=set_values_at_row_from_target,
                                                                      outputs_info=None,
                                                                      sequences=[np.asarray([[index, 0] for index in range(BATCH_SIZE)] , dtype=np.int32)],
                                                                      non_sequences=[value_matrix_output, value_matrix_target])
    value_matrix = average_value_matrix_results.sum(axis=1)

    # create a mask, for non-zero values in the observed (values) space
    value_matrix_mask = T.ceil(0.05 * value_matrix) # Ends up with reasonable (available) values --> 1.0, zero values --> 0.0

    # This is the action-weighted sum of the values. Don't worry, we normalize by action sum.
    # A mask is needed, so that we ignore the unknown values inherent. 
    # NOTE: As an alternative... we can take the max of known, and network value. To try this, need to sever connection to network, so gradient isn't distorted.
    weighted_value_matrix = value_matrix * action_matrix * value_matrix_mask 

    # action-weighted value average for values
    # Average value will be ~2.0 [zero-value action]
    # We use the mask, so that action-weights on unknown values are ignored. In both the sum, and the average.
    # NOTE: 0.05 is our regularization "epsilon" term
    values_sum_vector = weighted_value_matrix.sum(axis=1) / ((action_matrix * value_matrix_mask).sum(axis=1) + 0.05) 

    # minimize this, to maximize average value!
    # Average value will be ~1/3.0 = 0.33 [since normal/worst value of a normal spot is all folds]
    # Further reduce this, if we want the network to learn it slowly, not change values, etc.
    values_sum_inverse_vector = INCREASE_VALUES_SUM_INVERSE * 1.0 / (values_sum_vector + 1.0) # We need to make sure that gradient is never crazy

    # sum of all probabilities...
    # We want the probabilities to sum to 1.0...  but this should not be a huge consideration.
    # Therefore, dampen the value. But also make sure that this matches the target.
    probabilities_sum_vector = 0.10 * action_matrix.sum(axis=1) 
    
    # not sure if this is correct, but try it... 
    #values_output_matrix_masked = T.set_subtensor(output_matrix_masked[:,BET_ACTIONS_VALUE_CATEGORY], values_sum_vector)
    values_output_matrix_masked = T.set_subtensor(output_matrix_masked[:,BET_ACTIONS_VALUE_CATEGORY], values_sum_inverse_vector)
    probability_sum_output_matrix_masked = T.set_subtensor(values_output_matrix_masked[:,BET_ACTIONS_SUM_CATEGORY], probabilities_sum_vector)
    new_output_matrix_masked = probability_sum_output_matrix_masked
    #new_output_matrix_masked = T.set_subtensor(probability_sum_output_matrix_masked[:,BET_ACTIONS_SUM_VARIETY_CATEGORY], probabilities_square_vector)

    # Apply mask, so that we only consider:
    # A. for bets, bet values + bet-action values
    # B. for draws, draw values only
    simple_error = (new_output_matrix_masked - target_matrix) * output_category_mask

    return simple_error ** 2
    #return new_simple_error ** 2


# TODO: Include "empty" bits... so we can get a model started... which can be used as basis for next data?
def load_data():
    print('About to load up to %d items of data, for training format %s' % (MAX_INPUT_SIZE, TRAINING_FORMAT))
    # Do *not* bias the data, or smooth out big weight values, as we would for video poker.
    # 'deuce' has its own adjustments...
    data = _load_poker_csv(filename=DATA_FILENAME, max_input = MAX_INPUT_SIZE, keep_all_data=(TRAINING_FORMAT != 'video'), format=TRAINING_FORMAT, include_num_draws = INCLUDE_NUM_DRAWS, include_full_hand = INCLUDE_FULL_HAND, include_hand_context = INCLUDE_HAND_CONTEXT)

    # num_hands = total loaded, X = input, y = best cateogy, z = all categories, m = mask on all categories (if applicable)
    num_hands, X_all, y_all, z_all, m_all = data

    # Split into train (remainder), valid (1000), test (1000)
    X_split = np.split(X_all[:num_hands], [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    X_valid = X_split[0]
    X_test = X_split[1]
    X_train = X_split[2]

    print('X_valid %s %s' % (type(X_valid), X_valid.shape))
    print('X_test %s %s' % (type(X_test), X_test.shape))
    print('X_train %s %s' % (type(X_train), X_train.shape))

    # And same for Y
    y_split = np.split(y_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    y_valid = y_split[0]
    y_test = y_split[1]
    y_train = y_split[2]

    print('y_valid %s %s' % (type(y_valid), y_valid.shape))
    print('y_test %s %s' % (type(y_test), y_test.shape))
    print('y_train %s %s' % (type(y_train), y_train.shape))

    # And for Z
    z_split = np.split(z_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    z_valid = z_split[0]
    z_test = z_split[1]
    z_train = z_split[2]

    print('z_valid %s %s' % (type(z_valid), z_valid.shape))
    print('z_test %s %s' % (type(z_test), z_test.shape))
    print('z_train %s %s' % (type(z_train), z_train.shape))

    # And for M
    m_split = np.split(m_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
    m_valid = m_split[0]
    m_test = m_split[1]
    m_train = m_split[2]

    print('m_valid %s %s' % (type(m_valid), m_valid.shape))
    print('m_test %s %s' % (type(m_test), m_test.shape))
    print('m_train %s %s' % (type(m_train), m_train.shape))

    #sys.exit(0)

    # We ignore validation & test for now.
    #X_valid, y_valid = data[1]
    #X_test, y_test = data[2]

    return dict(
        # theano.shared() can't support huge amounts of data (more than 100k-300k examples (depending on size)
        #X_train=theano.shared(lasagne.utils.floatX(X_train), borrow=True),
        #y_train=T.cast(theano.shared(y_train, borrow=True), 'int32'),
        #z_train=theano.shared(lasagne.utils.floatX(z_train), borrow=True),
        #m_train=theano.shared(lasagne.utils.floatX(m_train), borrow=True),
        X_train=lasagne.utils.floatX(X_train),
        y_train=y_train,
        z_train=lasagne.utils.floatX(z_train),
        m_train=lasagne.utils.floatX(m_train),

        #X_valid=theano.shared(lasagne.utils.floatX(X_valid), borrow=True),
        #y_valid=T.cast(theano.shared(y_valid, borrow=True), 'int32'),
        #z_valid=theano.shared(lasagne.utils.floatX(z_valid), borrow=True),
        #m_valid=theano.shared(lasagne.utils.floatX(m_valid), borrow=True),
        X_valid=lasagne.utils.floatX(X_valid),
        y_valid=y_valid.astype(np.int32),
        z_valid=lasagne.utils.floatX(z_valid),
        m_valid=lasagne.utils.floatX(m_valid),

        X_test=theano.shared(lasagne.utils.floatX(X_test), borrow=True),
        y_test=T.cast(theano.shared(y_test, borrow=True), 'int32'),
        z_test=theano.shared(lasagne.utils.floatX(z_test), borrow=True),
        m_test=theano.shared(lasagne.utils.floatX(m_test), borrow=True),

        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_height=X_train.shape[2],
        input_width=X_train.shape[3],
        #input_dim=X_train.shape[1] * X_train.shape[2] * X_train.shape[3], # How much size per input?? 5x4x13 data (cards, suits, ranks)
        output_dim=32, # output cases
    )

# Alternatively, compare to same input... but two fully connected layers
def build_fully_connected_model(input_width, input_height, output_dim,
                                batch_size=BATCH_SIZE, input_var = None):
    print('building fat model, layer by layer...')
    num_input_cards = FULL_INPUT_LENGTH

    # Track all layers created, and return the full stack
    layers = []
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, num_input_cards, input_height, input_width),
        input_var = input_var,
        )
    layers.append(l_in)
    print('input layer shape %d x %d x %d x %d' % (batch_size, num_input_cards, input_height, input_width))

    l_hidden0 = lasagne.layers.DenseLayer(
        l_in, # l_conv2_2, #l_pool2,
        num_units=NUM_HIDDEN_UNITS, # NUM_HIDDEN_UNITS/2,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_hidden0)
    print('hidden layer l_hidden0. Shape %s' % str(l_hidden0.output_shape))

    l_hidden1 = lasagne.layers.DenseLayer(
        l_hidden0, # l_conv2_2, #l_pool2,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_hidden1)
    print('hidden layer l_hidden1. Shape %s' % str(l_hidden1.output_shape))

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    layers.append(l_hidden1_dropout)
    print('dropout layer l_hidden1_dropout. Shape %s' % str(l_hidden1_dropout.output_shape))

    l_out = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify, # Don't return softmax! #nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_out)

    print('final layer l_out, into %d dimension. Shape %s' % (output_dim, str(l_out.output_shape)))
    print('produced network of %d layers. TODO: name \'em!' % len(layers))

    # Don't really need l_out... but easy to access that way
    return (l_out, l_in, layers)

# Alternatively, but model with 5x5 filter on the bottom. Better visualization(?)
def build_fat_model(input_width, input_height, output_dim,
                    batch_size=BATCH_SIZE, input_var = None):
    print('building fat model, layer by layer...')
    num_input_cards = FULL_INPUT_LENGTH

    # Track all layers created, and return the full stack
    layers = []
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, num_input_cards, input_height, input_width),
        input_var = input_var,
        )
    layers.append(l_in)
    print('input layer shape %d x %d x %d x %d' % (batch_size, num_input_cards, input_height, input_width))
    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=NUM_FAT_FILTERS,
        filter_size=(5,5),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv1)
    print('convolution layer l_conv1. Shape %s' % str(l_conv1.output_shape))
    l_conv1_1 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=NUM_FAT_FILTERS,
        filter_size=(3,3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv1_1)
    print('convolution layer l_conv1_1. Shape %s' % str(l_conv1_1.output_shape))

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, pool_size=(2, 2), ignore_border=False)
    layers.append(l_pool1)
    print('maxPool layer l_pool1. Shape %s' % str(l_pool1.output_shape))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=NUM_FAT_FILTERS*2, 
        filter_size=(3,3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv2)
    print('convolution layer l_conv2. Shape %s' % str(l_conv2.output_shape))

    # Add 4th convolution layer...
    l_conv2_2 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=NUM_FAT_FILTERS*2,
        filter_size=(3,3),
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv2_2)
    print('convolution layer l_conv2_2. Shape %s' % str(l_conv2_2.output_shape))

    # OK now skip second max-pool.

    l_hidden1 = lasagne.layers.DenseLayer(
        l_conv2_2, #l_pool2,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_hidden1)

    print('hidden layer l_hidden1. Shape %s' % str(l_hidden1.output_shape))

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    layers.append(l_hidden1_dropout)

    print('dropout layer l_hidden1_dropout. Shape %s' % str(l_hidden1_dropout.output_shape))

    l_out = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify, # Don't return softmax! #nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_out)

    print('final layer l_out, into %d dimension. Shape %s' % (output_dim, str(l_out.output_shape)))
    print('produced network of %d layers. TODO: name \'em!' % len(layers))

    # Don't really need l_out... but easy to access that way
    return (l_out, l_in, layers)

# Need to explicitly pass input_var... if we want to feed input into the network, without putting that input into "shared"
def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE, input_var = None,
                use_leaky_units=DEFAULT_LEAKY_UNITS # small "leak" in ReLu units, to avoid saturation at 0.0?
                ):
    print('building model, layer by layer...')
    num_input_cards = FULL_INPUT_LENGTH

    # Track all layers created, and return the full stack
    layers = []

    # Shape is [cards + bits] x height x width
    if input_var:
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_input_cards, input_height, input_width),
            input_var = input_var,
            )
    else:
        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, num_input_cards, input_height, input_width),
            )
    layers.append(l_in)

    print('input layer shape %d x %d x %d x %d' % (batch_size, num_input_cards, input_height, input_width))

    # Do we use rectified linear units, or a leaky version thereof? 
    # NOTE: Could mix this layer by layer... but better to keep it consistent.
    nonlinearity = lasagne.nonlinearities.rectify # default. Linear for all values >= 0.0. Negative values to go 0.0
    if use_leaky_units:
        print('Initializing with \'leaky\' ReLU units. Leakiness is default (0.01)');
        nonlinearity = lasagne.nonlinearities.leaky_rectify # default 0.01 leakiness

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=NUM_FILTERS, #16, #32,
        filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv1)

    print('convolution layer l_conv1. Shape %s' % str(l_conv1.output_shape))

    # No hard rule that we need to pool after every 3x3!
    # l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))
    l_conv1_1 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=NUM_FILTERS, #16, #32,
        filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv1_1)
    print('convolution layer l_conv1_1. Shape %s' % str(l_conv1_1.output_shape))

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, pool_size=(2, 2), ignore_border=False)
    layers.append(l_pool1)
    # Try *not pooling* in the suit layer...
    #l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, ds=(2, 1))

    print('maxPool layer l_pool1. Shape %s' % str(l_pool1.output_shape))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=NUM_FILTERS*2, #16, #32,
        filter_size=(3,3), #(5,5), # (3,3), #(5, 5),
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv2)
    print('convolution layer l_conv2. Shape %s' % str(l_conv2.output_shape))

    # Add 4th convolution layer...
    # l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))
    l_conv2_2 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=NUM_FILTERS*2, #16, #32,
        filter_size=(3,3), #(5,5), # (3,3), #(5, 5),
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_conv2_2)
    print('convolution layer l_conv2_2. Shape %s' % str(l_conv2_2.output_shape))

    # Question? No need for Max-pool for already narrow network... NO
    # Try *not pooling* in the suit layer...
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, pool_size=(2, 2), ignore_border=False)
    layers.append(l_pool2)
    #l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, ds=(2, 1))

    print('maxPool layer l_pool2. Shape %s' % str(l_pool2.output_shape))

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=nonlinearity,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_hidden1)

    print('hidden layer l_hidden1. Shape %s' % str(l_hidden1.output_shape))

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)
    layers.append(l_hidden1_dropout)

    print('dropout layer l_hidden1_dropout. Shape %s' % str(l_hidden1_dropout.output_shape))

    l_out = lasagne.layers.DenseLayer(
        l_hidden1_dropout, #l_hidden2_dropout, # l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=nonlinearity, # Don't return softmax! #nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )
    layers.append(l_out)

    print('final layer l_out, into %d dimension. Shape %s' % (output_dim, str(l_out.output_shape)))
    print('produced network of %d layers. TODO: name \'em!' % len(layers))

    # Don't really need l_out... but easy to access that way
    return (l_out, l_in, layers)


# Adjust from the standard create_iter_functions, to deal with z_batch being vector of values.
def create_iter_functions_full_output(dataset, output_layer,
                                      input_var = None, # optionally supply input layer, model execution w/o theano.shared()
                                      input_layer = None,
                                      X_tensor_type=T.tensor4, # T.matrix,
                                      batch_size=BATCH_SIZE,
                                      learning_rate=LEARNING_RATE, momentum=MOMENTUM,
                                      ada_learning_rate=ADA_DELTA_LEARNING_RATE, ada_rho = ADA_DELTA_RHO, ada_epsilon = ADA_DELTA_EPSILON,
                                      default_adaptive=DEFAULT_ADAPTIVE):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    print('creating iter funtions')

    #print('input dataset %s' % dataset)

    batch_index = T.iscalar('batch_index')
    #if not input_var:
    #X_batch = X_tensor_type('x') # inputs to the network
    y_batch = T.ivector('y') # "best class" for the network
    z_batch = T.matrix('z') # all values for the network
    #if TRAIN_MASKED_OBJECTIVE:
    m_batch = T.matrix('m') # mask for all values, if some are relevant, others are N/A or ?

    # TODO: Replace the "batch index" input, with input of actual batch... (so data doesn't need to be stored in theano.shared())
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    if not TRAIN_MASKED_OBJECTIVE:
        # For training, not determinisitc
        output_batch = lasagne.layers.get_output(output_layer) # , deterministic=DETERMINISTIC_MODEL_RUN)
        loss_train_mask = lasagne.objectives.squared_error(output_batch, z_batch)
        loss_train_no_mask = loss_train_mask.mean()

        # For eval, deterministic!
        output_batch_eval = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
        loss_eval_mask = lasagne.objectives.squared_error(output_batch, z_batch)
        loss_eval_no_mask = loss_eval_mask.mean()
    else:                              
        # error is computed same as un-masked objective... but we also supply a mask for each output. 1 = output matters 0 = N/A or ?
        # NOTE: X_batch will be loaded into "shared". To not do this... do function.eval({input_layer.var_in: data})

        # For training, not determinisitc
        output_batch = lasagne.layers.get_output(output_layer) # , deterministic=DETERMINISTIC_MODEL_RUN)
        loss_train_mask = value_action_error(output_batch, z_batch) * m_batch
        loss_train_mask = loss_train_mask.mean()

        # For eval, deterministic!
        output_batch_eval = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
        loss_eval_mask = value_action_error(output_batch_eval, z_batch) * m_batch
        loss_eval_mask = loss_eval_mask.mean()

    if TRAIN_MASKED_OBJECTIVE:
        print('--> We are told to use \'masked\' loss function. So training & validation loss will be computed on inputs with mask == 1 only')
        #objective = objective_mask
        loss_train = loss_train_mask
        loss_eval = loss_eval_mask
    else:
        #objective = objective_no_mask
        loss_train = loss_train_no_mask
        loss_eval = loss_eval_no_mask
    
    # Prediction actually stays the same! Since we still want biggest value in the array... and compare to Y
    # NOTE: If accuracy really supposed to be first 5... use [:,0:5] below [multiply by mask, before argmax]
    if TRAIN_MASKED_OBJECTIVE:
        # Apply [0:5] only mask, to consider accuracy (for bet alues)
        # TODO: Allow predictions to consider both cases with output == bet/raise and ouput == num_cards draw

        # Builds a mask, row by row, of either 1 == bets or 1 == draws, depending on the output category. 
        # TODO: Put this in a function... but not a theano.function. 
        category_mask_results, mask_updates = theano.scan(fn=set_mask_at_row_from_target,
                                                          outputs_info=None,
                                                          sequences=[np.asarray([[index, 0] for index in range(BATCH_SIZE)] , dtype=np.int32)],
                                                          non_sequences=[np.zeros((BATCH_SIZE,32), dtype=np.float32), z_batch])
        output_category_mask = category_mask_results.sum(axis=1)


        pred = T.argmax(lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN) * output_category_mask, axis=1)
    else:
        # If not masked output... then just mask which categories matter for this input class (outside constant)
        pred = T.argmax(lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN) * OUTPUT_CATEGORY_DEFAULT_MASK, axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    # Default: Nesterov momentum. Try something else?
    print('Building updates.nesterov_momentum with learning rate %.8ff, momentum %.2f' % (learning_rate, momentum))
    updates_nesterov = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

    # Try adaptive training -- varies learning rate in reaction to how data reacts.
    print('Building updates.adadelta with learning rate %.8f, rho %.3f, epsilon %.8f' % (ada_learning_rate, ada_rho, ada_epsilon))
    updates_adadelta = lasagne.updates.adadelta(loss_train, all_params, learning_rate=ada_learning_rate, rho=ada_rho, epsilon=ada_epsilon)

    # RMSprop = very similar to AdaDelta. Need to read more.
    print('Building updates.rmsprop with learning rate %.8f, rho %.3f, epsilon %.8f' % (ada_learning_rate, ada_rho, ada_epsilon))
    updates_rmsprop = lasagne.updates.rmsprop(loss_train, all_params, learning_rate=ada_learning_rate, rho=ada_rho, epsilon=ada_epsilon)

    # Be careful not to include in givens, what won't be used. Theano will complain!
    if TRAIN_MASKED_OBJECTIVE:
        givens_train = {}
    else:
         givens_train = {}

    # Function to train with nesterov momentum...
    iter_train_nesterov = theano.function(
        [input_layer.input_var, z_batch, m_batch], loss_train,
        updates=updates_nesterov,
        givens=givens_train,
        on_unused_input='warn', # We might not need "m_batch" if unmasked input... but pain to deal with conditional compiling
        )

    # Also, function for adaptive learning, which we may or may not use
    iter_train_ada_delta = theano.function(
        [input_layer.input_var, z_batch, m_batch], loss_train,
        updates=updates_adadelta,
        givens=givens_train,
        on_unused_input='warn',
        )
    iter_train_rmsprop = theano.function(
        [input_layer.input_var, z_batch, m_batch], loss_train,
        updates=updates_rmsprop,
        givens=givens_train,
        on_unused_input='warn', 
        )

    # Fixed learning rate with Nesterov momentum, is still the default training function
    iter_train = iter_train_nesterov

    # Don't do adaptive training on fresh model. needs to run with learning & momentum first. Then... we delta
    if default_adaptive:
        if ADAPTIVE_USE_RMSPROP:
            print('--> Using adaptive learning (RMSprop) as default training!')
            iter_train = iter_train_rmsprop
        else:
            print('Using adaptive learning (AdaDelta) as default training!')
            iter_train = iter_train_ada_delta 
    else:
        print('--> Using fixed learning rate with Nesterov momentum as default training.')

    # Be careful not to include in givens, what won't be used. Theano will complain!
    if TRAIN_MASKED_OBJECTIVE:
        givens_valid = {}
    else:
        givens_valid = {}
    iter_valid = theano.function(
        [input_layer.input_var, y_batch, z_batch, m_batch], [loss_eval, accuracy],
        givens=givens_valid,
        on_unused_input='warn', # We might not need "m_batch" if unmasked input... but pain to deal with conditional compiling
    )

    return dict(
        train=iter_train,
        train_nesterov=iter_train_nesterov,
        train_ada_delta=iter_train_ada_delta,
        valid=iter_valid,
        #test=iter_test,
    )

# Now how do I return theano function to predict, from my given thing? Should be simple.
# TODO: output second-best choice
# TODO: show choices, alongside inputs
def predict_model(output_layer, test_batch, format = 'deuce', input_layer = None):
    print('Computing predictions on test_batch: %s %s' % (type(test_batch), test_batch.shape))
    if input_layer:
        #print('evaluating test_batch using explicit input_layer (no theano.shared())!')
        pred_all = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
    else:
        pred_all = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=DETERMINISTIC_MODEL_RUN)

    if format == 'deuce_events' or format == 'holdem_events':
        # Slice it, so only relevant rows are looked at
        pred = pred_all[:17,:10] # pred_all[:17,:5] # hack: show first 17 examples only
    elif format == 'holdem':
        print(HOLDEM_VALUE_KEYS)
        pred = pred_all[:17,:len(HOLDEM_VALUE_KEYS)] # show first 17 examples only
    else:
        pred = pred_all        

    # Print out entire predictions array
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    print('Prediciton: %s' % pred) 
    # Return options to previous settings...
    np.set_printoptions(**opt)

    # TODO: For "deuce_events" format... show best action inside of events actions only.

    #print(tp.pprint(pred))
    if input_layer:
        # Or is it {l_in.input_var: xx}
        softmax_values = pred.eval({input_layer.input_var: lasagne.utils.floatX(test_batch)})
    else:
        softmax_values = pred.eval()
    print(softmax_values)

    if format == 'deuce_events' or format == 'holdem_events':
        pred_max = T.argmax(pred[:,0:5], axis=1) # 0-5 bets
    elif format == 'holdem':
        # TODO: Ignore first element... (overall value)
        """
        #pred[:,0] = 0.0 # zero out first row, which is "best_value" column
        zeros = T.zeros_like(pred)
        zeros_subtensor = zeros[:,0:1]
        pred_values = T.set_subtensor(zeros_subtensor, pred)
        """
        pred_max = T.argmax(pred[:,0:len(HOLDEM_VALUE_KEYS)], axis=1) # values of actions
    else:
        pred_max = T.argmax(pred, axis=1)

    print('Maximums %s' % pred_max)
    #print(tp.pprint(pred_max))

    if input_layer:
        # Or is it {l_in.input_var: xx}
        softmax_choices = pred_max.eval({input_layer.input_var: lasagne.utils.floatX(test_batch)})
    else:
        softmax_choices = pred_max.eval()
    print(softmax_choices)

    # now debug the softmax choices...
    if format != 'deuce_events':
        softmax_debug = [DRAW_VALUE_KEYS[i] for i in softmax_choices]
    else:
        softmax_debug = [eventCategoryName[i] for i in softmax_choices]
    print(softmax_debug)


# Get 0-BATCH_SIZE hands, evaluate, and return matrix of vectors
def evaluate_batch_hands(output_layer, test_cases, include_hand_context = INCLUDE_HAND_CONTEXT, input_layer = None): 
    now = time.time()
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[0])
    
    # case = [hand_string, int(num_draws)]
    #print(test_cases)
    if TRAINING_FORMAT == 'video' and isinstance(test_cases[0], basestring):
        # for 'video', consider case where 'case' just includes a card string.
        test_batch = np.array([cards_input_from_string(hand_string=case,
                                                       include_num_draws = False,
                                                       include_full_hand = False,
                                                       include_hand_context = False) for case in test_cases], TRAINING_INPUT_TYPE)
    else:
        test_batch = np.array([cards_input_from_string(hand_string=case[0],
                                                       include_num_draws=INCLUDE_NUM_DRAWS, num_draws=case[1], 
                                                       include_full_hand = INCLUDE_FULL_HAND,
                                                       include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], TRAINING_INPUT_TYPE)

    # print('%.2fs to create BATCH_SIZE input' % (time.time() - now))
    #now = time.time()
    if input_layer:
        #print('evaluating batch with input layer, so no grow in theano.shared()!')
        pred = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
    else:
        pred = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=DETERMINISTIC_MODEL_RUN)
    # print('%.2fs to get_output' % (time.time() - now))
    #now = time.time()

    #print('Prediciton: %s' % pred)
    #print(tp.pprint(pred))
    if input_layer:
        softmax_values = pred.eval({input_layer.input_var: lasagne.utils.floatX(test_batch)})
    else:
        softmax_values = pred.eval()
    print('%.2fs to eval() output' % (time.time() - now))
    now = time.time()

    return softmax_values


# Return 32-point vector.
# Make a batch, to evaluate single hand. Expensive!!
def evaluate_single_hand(output_layer, hand_string_dealt, num_draws = 1, 
                         include_hand_context = INCLUDE_HAND_CONTEXT,
                         input_layer = None):
    test_cases = [[hand_string_dealt, num_draws]]
    softmax_values = evaluate_batch_hands(output_layer, test_cases, include_hand_context = include_hand_context, input_layer = input_layer)
    return softmax_values[0]

# Similar for Holdem. 
def evaluate_single_holdem_hand(output_layer, input_layer, cards, flop, turn, river):
    now = time.time()

    # Just one case, and all zeros to fit the input.
    # TODO: Create side function to avoid this tostring, then back to array nonsense...
    test_case = holdem_cards_input_from_string(hand_string(cards), hand_string(flop), hand_string(turn), hand_string(river))
    test_cases = [test_case]
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(np.zeros_like(test_case))
    test_batch = np.array(test_cases, TRAINING_INPUT_TYPE)

    # Get model prediction. Requires input layer.
    pred = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
    softmax_values = pred.eval({input_layer.input_var: lasagne.utils.floatX(test_batch)})
    softmax_single_vector = softmax_values[0]
    print('%.2fs to eval() output' % (time.time() - now))

    # Return output vector for our single input.
    return softmax_single_vector

# Just give us the bits... expect 26x17x17 matrix...
def evaluate_single_event(output_layer, event_input, input_layer = None):
    now = time.time()
    test_batch = np.array([event_input for i in range(BATCH_SIZE)], TRAINING_INPUT_TYPE)
    # print('%.2fs to create BATCH_SIZE input' % (time.time() - now))
    now = time.time()
    if input_layer:
        #print('evaluating batch with input layer, so no grow in theano.shared()!')
        pred = lasagne.layers.get_output(output_layer, deterministic=DETERMINISTIC_MODEL_RUN)
    else:
        pred = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=DETERMINISTIC_MODEL_RUN)
    # print('%.2fs to get_output' % (time.time() - now))
    now = time.time()
    if input_layer:
        softmax_values = pred.eval({input_layer.input_var: lasagne.utils.floatX(test_batch)})
    else:
        softmax_values = pred.eval()
    print('%.2fs to eval() output' % (time.time() - now))
    now = time.time()

    return softmax_values[0]

# Pickle the model. Can be un-pickled, if building same network.
def save_model(out_file=None, output_layer=None):
    # TODO: Should save layer throughout
    # Save the layer to file...
    if out_file and output_layer:
        # Get all param values (for fixed network)
        all_param_values = lasagne.layers.get_all_param_values(output_layer)

        # save values to output file!
        print('pickling model %d param values to %s' % (len(all_param_values), out_file))
        with open(out_file, 'wb') as f:
            pickle.dump(all_param_values, f, -1)

# If input is < expected input... fill remaining rows with noise (for training)... or with zeros (production)
def expand_parameters_input_to_match(all_param_values, zero_fill = False):
    # HACK: If input size doesn't match... pad the input (first) layer with random noise from working layers of the model.
    # NOTE: This should warn, and crash. Very hacky. But also necessary if we grow (or shrink) the model and don't want to lose knowledge.
    first_layer_params = all_param_values[0]
    first_layer_shape = first_layer_params.shape
    input_len = first_layer_shape[1]
    if input_len < FULL_INPUT_LENGTH:
        print('too few inputs in model from disk! %d < %d' % (input_len, FULL_INPUT_LENGTH))
        fill_len = FULL_INPUT_LENGTH - input_len
        for i in range(fill_len):
            # Take a slice for random existing input
            # TODO: In executing model... should fill with zeros!
            # TODO: all of this in function!
            random_row = np.random.randint(input_len)
            params_per_input = first_layer_params[:,random_row:random_row+1,:,:]
            # Use this to fill with zeros! Instead of random noise from which to train on...
            if zero_fill:
                params_per_input = np.zeros_like(params_per_input, dtype=TRAINING_INPUT_TYPE)
            print(params_per_input.shape)
            print('adding input row...')
            # Append it, to serve as noise for missing input in model
            first_layer_params = np.concatenate((first_layer_params, params_per_input), axis = 1)
            print(first_layer_params.shape)
        all_param_values[0] = first_layer_params
        print(all_param_values[0].shape)

def main(num_epochs=NUM_EPOCHS, out_file=None):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    sys.stdout.flush() # Helps keep track of output live in re-directed out

    # Pass this reference, to compute theano graph
    input_var = T.tensor4('inputs')
    if USE_FULLY_CONNECTED_MODEL:
        output_layer, input_layer, layers = build_fully_connected_model(
            input_height=dataset['input_height'],
            input_width=dataset['input_width'],
            output_dim=dataset['output_dim'],
            input_var = input_var,
            )
    elif USE_FAT_MODEL:
        output_layer, input_layer, layers = build_fat_model(
            input_height=dataset['input_height'],
            input_width=dataset['input_width'],
            output_dim=dataset['output_dim'],
            input_var = input_var,
            )
    else:
        output_layer, input_layer, layers = build_model(
            input_height=dataset['input_height'],
            input_width=dataset['input_width'],
            output_dim=dataset['output_dim'],
            input_var = input_var,
            )

    # Optionally, load in previous model parameters, from pickle!
    # NOTE: Network shape must match *exactly*
    if os.path.isfile(out_file):
        print('Existing model in file %s. Attempt to load it!' % out_file)
        all_param_values_from_file = np.load(out_file) # Use GPU utils! # np.load(out_file)
        all_param_values_from_file_with_type = []
        for value in all_param_values_from_file:
            all_param_values_from_file_with_type.append(lasagne.utils.floatX(value))
        print('Loaded values %d' % len(all_param_values_from_file_with_type))
        for layer_param in all_param_values_from_file_with_type:
            #print(layer_param)
            print(layer_param.shape)
            print('---------------')
        expand_parameters_input_to_match(all_param_values_from_file_with_type, zero_fill=False)
        #print(all_param_values_from_file_with_type)
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file_with_type)
        print('Successfully initialized model with previous saved params!')
    else:
        print('No existing model file (or skipping intialization) %s' % out_file)

    # Define iter functions differently. Since we now want to predict the entire vector. 
    iter_funcs = create_iter_functions_full_output(
        dataset,
        output_layer,
        input_layer = input_layer,
        X_tensor_type=T.tensor4,
        input_var = input_var
        )

    print("Starting training...")
    now = time.time()
    try:
        # When do we switch to adaptive training? Problems with adapative training and events, so disable that.
        switch_adaptive_after = EPOCH_SWITCH_ADAPT
        switch_adaptive_after = 10000 # never
        if TRAINING_FORMAT == 'deuce_events' and DISABLE_EVENTS_EPOCH_SWITCH:
            switch_adaptive_after = 10000 # never
        for epoch in train(iter_funcs, dataset, epoch_switch_adapt=switch_adaptive_after):
            sys.stdout.flush() # Helps keep track of output live in re-directed out
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.8f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.8f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                    epoch['valid_accuracy'] * 100))

            # Save model, after every epoch.
            save_model(out_file=out_file, output_layer=output_layer)

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass

    # Can we do something with final output model? Like show a couple of moves...
    #test_batch = dataset['X_test']

    # Test cases -- for Deuce. Can it keep good hands, break pairs, etc? 
    # NOTE: These cases, and entire training... do *not* include number of draw rounds left. Add that!
    test_cases_draw = [['3h,3s,3d,5c,6d', 3], ['2h,3s,4d,6c,5s', 1], ['3s,2h,4d,8c,5s', 0],
                       ['3h,3s,3d,5c,6d', 0], ['2h,3s,4d,6c,5s', 2], ['3s,2h,4d,8c,5s', 3],
                       ['As,Ad,4d,3s,2c', 1], ['As,Ks,Qs,Js,Ts', 2], ['3h,4s,3d,5c,6d', 2],
                       ['8s,Ad,Kd,8c,Jd', 3], ['8s,Ad,2d,7c,Jd', 2], ['2d,7d,8d,9d,4d', 1], 
                       ['7c,8c,Tc,Js,Qh', 3], ['2c,8s,5h,8d,2s', 2],
                       ['[8s,9c,8c,Kd,7h]', 2], ['[Qh,3h,6c,5s,4s]', 1], ['[Jh,Td,9s,Ks,5s]', 1],
                       ['[6c,4d,Ts,Jc,6s]', 3], ['[4h,8h,2c,7d,3h]', 2], ['[2c,Ac,Tc,6d,3d]', 1], 
                       ['[Ad,3c,Tc,4d,5d]', 1], ['3d,Jc,7d,Ac,6s', 2],
                       ['7h,Kc,5s,2d,Tc', 3], ['5c,6h,Jc,7h,2d', 1], ['Ts,As,3s,2d,4h', 3]] 

    # hand (2 cards), flop (0 or 3), turn (0 or 1), river (0 or 1). All we need.
    test_cases_holdem = [['Ad,Ac', '[]', '', ''], # AA preflop
                         ['[8d,5h]','[Qh,9d,3d]','[Ad]','[7c]'], # missed draw
                         ['4d,5d', '[6d,7d,3c]', 'Ad', ''], # made flush
                         ['7c,9h', '[8s,6c,Qh]', '', ''], # open ended straight draw
                         ['Ad,Qd', '[Kd,Td,2s]', '3s', ''], # big draw
                         ['7s,2h', '', '', ''], # weak hand preflop
                         ['Ts,Th', '', '', ''], # good hand preflop
                         ['9s,Qh', '', '', ''], # average hand preflop
                         ]
                         
    if TRAINING_FORMAT == 'holdem' or TRAINING_FORMAT == 'holdem_events':
        test_cases = test_cases_holdem
    else:
        test_cases = test_cases_draw
    test_cases_size = len(test_cases)
                         
    print('looking at some test cases: %s' % test_cases)

    # Fill in test cases to get to batch size?
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[0])
    # Test_batch... 5 cards, no 3-card "round" encoding.
    # test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)
    if TRAINING_FORMAT == 'video' or TRAINING_FORMAT == 'deuce':
        test_batch = np.array([cards_input_from_string(hand_string=case[0], include_num_draws=(INCLUDE_NUM_DRAWS and TRAINING_FORMAT != 'video'), num_draws=case[1], include_full_hand = (INCLUDE_FULL_HAND and TRAINING_FORMAT != 'video'), include_hand_context = (INCLUDE_HAND_CONTEXT and TRAINING_FORMAT != 'video')) for case in test_cases], np.int32)
    elif TRAINING_FORMAT == 'deuce_events':
        # TODO:  Add context, if made available...
        test_batch = np.array([cards_input_from_string(hand_string=case[0], include_num_draws=INCLUDE_NUM_DRAWS, num_draws=case[1], include_full_hand = INCLUDE_FULL_HAND, include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], np.int32)
    elif TRAINING_FORMAT == 'holdem' or TRAINING_FORMAT == 'holdem_events':
        test_batch = np.array([holdem_cards_input_from_string(case[0], case[1], case[2], case[3]) for case in test_cases], np.int32)
        
    predict_model(output_layer=output_layer, test_batch=test_batch, format = TRAINING_FORMAT, input_layer = input_layer)

    print('again, the test cases: \n%s' % test_cases[:test_cases_size])    

    return output_layer


if __name__ == '__main__':
    # Additional flags... only if we inlcude in model
    extra_flags = ''
    if INCLUDE_FULL_HAND:
        extra_flags += '_full_hand'
    if INCLUDE_HAND_CONTEXT:
        extra_flags += '_hand_context'

    if TRAINING_FORMAT != 'holdem':
        output_layer_filename = '%striple_draw_conv_%.2f_learn_rate_%d_epoch_adaptive_%d_filters_%s_border_%d_num_draws%s_model.pickle' % (TRAINING_FORMAT, LEARNING_RATE, EPOCH_SWITCH_ADAPT, NUM_FILTERS, BORDER_SHAPE, INCLUDE_NUM_DRAWS, extra_flags)
    else:
        output_layer_filename = '%s_conv_%.2f_learn_rate_%d_epoch_adaptive_%d_filters_%s_border_%d_num_draws%s_model.pickle' % (TRAINING_FORMAT, LEARNING_RATE, EPOCH_SWITCH_ADAPT, NUM_FILTERS, BORDER_SHAPE, INCLUDE_NUM_DRAWS, extra_flags)
    main(out_file=output_layer_filename)
