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
from draw_poker import _load_poker_csv
from draw_poker import cards_input_from_string
from draw_poker import create_iter_functions
from draw_poker import train

"""
Use similar network... to learn triple draw poker!!

First, need new data import functins.
"""

DATA_FILENAME = '../data/40k_hands_triple_draw_events.csv' # 40k hands (a lot more events) from man vs CNN, CNN vs sim, and sim vs sim [need more CNN vs CNN]
# '../data/60k_triple_draw_events.csv' # 60k 'event's from a few thousand full hands.
# '../data/500k_hands_sample_details_all.csv' # all 32 values for 'deuce' (draws)
# '../data/500k_hands_sample_details_all.csv' # all 32 values.
# '../data/200k_hands_sample_details_all.csv' # all 32 values. Cases for 1, 2 & 3 draws left
# '../data/60000_hands_sample_details.csv' # 60k triple draw hands... best draw output only

MAX_INPUT_SIZE = 110000 # 120000 # 10000000 # Remove this constraint, as needed
VALIDATION_SIZE = 10000
TEST_SIZE = 0 # 5000
NUM_EPOCHS = 50 # 100 # 500 # 500 # 20 # 20 # 100
BATCH_SIZE = 100 # 50 #100
BORDER_SHAPE = "valid" # "same" # "valid" # "full" = pads to prev shape "valid" = shrinks [bad for small input sizes]
NUM_FILTERS = 24 # 16 # 32 # 16 # increases 2x at higher level
NUM_HIDDEN_UNITS = 1024 # 512 # 256 #512
LEARNING_RATE = 0.02 # 0.1 # 0.1 #  0.05 # 0.01 # 0.02 # 0.01
MOMENTUM = 0.9
EPOCH_SWITCH_ADAPT = 20 # 12 # 10 # 30 # switch to adaptive training after X epochs of learning rate & momentum with Nesterov
ADA_DELTA_EPSILON = 1e-4 # 1e-6 # default is smaller, be more aggressive...
ADA_LEARNING_RATE = 1.0 # 0.5 # algorithm confuses this

# Here, we get into growing input information, beyond the 5-card hand.
INCLUDE_NUM_DRAWS = True # 3 "cards" to encode number of draws left. ex. 2 draws: [0], [1], [1]
INCLUDE_FULL_HAND = True # add 6th "card", including all 5-card hand... in a single matrix [Good for detecting str8, pair, etc]

# Train on masked objective? 
# NOTE: load_data needs to be already producing such masks.
# NOTE: Default mask isn't all 1's... it's best_output == 1, others == 0
# WARINING: Theano will complain if we pass a mask and don't use it!
TRAIN_MASKED_OBJECTIVE = True # False # True # False # True 

# Do we use linear loss? Why? If cases uncertain or small sample, might be better to approximate the average...
LINEAR_LOSS_FOR_MASKED_OBJECTIVE = False # True # False # True

# If we are trainging on poker events (bets, raises and folds) instead of draw value,
# input and output shape will be the same. But the way it's uses is totally different. 
# NOTE: We keep shape the same... so we can use the really good "draw value" model as initialization.
TRAINING_FORMAT = 'deuce_events' # 'deuce' # 'video'
INCLUDE_HAND_CONTEXT = True # False 17 or so extra "bits" of context. Could be set, could be zero'ed out.
DISABLE_EVENTS_EPOCH_SWITCH = True # False # Is system stable enough, to switch to adaptive training?

# Helps speed up inputs?
TRAINING_INPUT_TYPE = theano.config.floatX # np.int32

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
for i in [0,1,2,3,4]: 
    first_five_vector[i] = 1.0
first_five_matrix = [first_five_vector for i in xrange(BATCH_SIZE)]
first_five_mask = np.array(first_five_matrix)

print('first_five_mask:')
print(first_five_mask)

# Can we at least use outside mask??
def value_action_error(output_matrix, target_matrix):
    # Apply mask to values that matter.
    output_matrix_masked = output_matrix * first_five_mask 

    
    # Now, use matrix manipulation to tease out the current action vector, and current value vector
    action_matrix = output_matrix[:,5:10] # Always output matrix. It's all we got!
    value_matrix_output = output_matrix[:,0:5] # implied from the current model.
    value_matrix_target = target_matrix[:,0:5] # directly from observation
    # value_matrix = value_matrix_output # Try to learn values from the network. Warning! This creates a gradient, and network will change.
    value_matrix = value_matrix_target # Use real values. And reduce/remove pressure to tweak values, which is bad.

    # create a mask, for non-zero values in the observed (values) space
    value_matrix_mask = T.ceil(0.1 * value_matrix) # Ends up with reasonable (available) values --> 1.0, zero values --> 0.0

    # This is the action-weighted sum of the values. Don't worry, we normalize by action sum.
    # A mask is needed, so that we ignore the unknown values inherent. 
    # NOTE: As an alternative... we can take the max of known, and network value. To try this, need to sever connection to network, so gradient isn't distorted.
    weighted_value_matrix = value_matrix * action_matrix * value_matrix_mask 

    # action-weighted value average for values
    # Average value will be ~2.0 [zero-value action]
    # We use the mask, so that action-weights on unknown values are ignored. In both the sum, and the average.
    values_sum_vector = weighted_value_matrix.sum(axis=1) / ((action_matrix * value_matrix_mask).sum(axis=1) + 0.05) 

    # minimize this, to maximize average value!
    # Average value will be ~1/3.0 = 0.33 [since normal/worst value of a normal spot is all folds]
    # Further reduce this, if we want the network to learn it slowly, not change values, etc.
    values_sum_inverse_vector = 1.0 / (values_sum_vector + 1.0) # We need to make sure that gradient is never crazy

    # sum of all probabilities...
    # We want the probabilities to sum to 1.0...  but this should not be a huge consideration.
    # Therefore, dampen the value. But also make sure that this matches the target.
    probabilities_sum_vector = 0.05 * action_matrix.sum(axis=1) 
    #probabilities_square_vector = 0.05 * (action_matrix ** 2).sum(axis=1) 
    
    # not sure if this is correct, but try it... 
    #values_output_matrix_masked = T.set_subtensor(output_matrix_masked[:,10], values_sum_vector)
    values_output_matrix_masked = T.set_subtensor(output_matrix_masked[:,10], values_sum_inverse_vector)
    probability_sum_output_matrix_masked = T.set_subtensor(values_output_matrix_masked[:,11], probabilities_sum_vector)
    new_output_matrix_masked = probability_sum_output_matrix_masked
    #new_output_matrix_masked = T.set_subtensor(probability_sum_output_matrix_masked[:,12], probabilities_square_vector)

    # Values that can't be controlled... won't be.
    #simple_error = output_matrix_masked - target_matrix
    simple_error = new_output_matrix_masked - target_matrix

    return simple_error ** 2
    #return new_simple_error ** 2


# TODO: Include "empty" bits... so we can get a model started... which can be used as basis for next data?
def load_data():
    print('About to load up to %d items of data, for training format %s' % (MAX_INPUT_SIZE, TRAINING_FORMAT))
    # Do *not* bias the data, or smooth out big weight values, as we would for video poker.
    # 'deuce' has its own adjustments...
    data = _load_poker_csv(filename=DATA_FILENAME, max_input = MAX_INPUT_SIZE, keep_all_data=True, format=TRAINING_FORMAT, include_num_draws = INCLUDE_NUM_DRAWS, include_full_hand = INCLUDE_FULL_HAND, include_hand_context = INCLUDE_HAND_CONTEXT)

    # X = input, y = best cateogy, z = all categories, m = mask on all categories (if applicable)
    X_all, y_all, z_all, m_all = data

    # Split into train (remainder), valid (1000), test (1000)
    X_split = np.split(X_all, [VALIDATION_SIZE, VALIDATION_SIZE + TEST_SIZE])
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
        X_train=theano.shared(lasagne.utils.floatX(X_train), borrow=True),
        y_train=T.cast(theano.shared(y_train, borrow=True), 'int32'),
        z_train=theano.shared(lasagne.utils.floatX(z_train), borrow=True),
        m_train=theano.shared(lasagne.utils.floatX(m_train), borrow=True),

        X_valid=theano.shared(lasagne.utils.floatX(X_valid), borrow=True),
        y_valid=T.cast(theano.shared(y_valid, borrow=True), 'int32'),
        z_valid=theano.shared(lasagne.utils.floatX(z_valid), borrow=True),
        m_valid=theano.shared(lasagne.utils.floatX(m_valid), borrow=True),

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

def build_model(input_width, input_height, output_dim,
                batch_size=BATCH_SIZE):
    print('building model, layer by layer...')

    # Our input consists of 5 cards, strictly in order... and other inputs.
    # For inspiration to this approach, consider DeepMind's GO CNN. http://arxiv.org/abs/1412.6564v1
    # They use multiple fake "boards" to encode player strength, in 7 'bits'
    # Perhaps not efficient.... but easy to understand.
    # TODO: How to you encode entire hand history? Or even opponent draws & bets?
    num_input_cards = 5 
    # One "card" with all five card matrix
    if INCLUDE_FULL_HAND:
        num_input_cards += 1
    # 3 "bits" to encode where we are in the hand, in terms of number of draws.
    if INCLUDE_NUM_DRAWS:
        num_input_cards += 3

    # [xPosition, xPot, xBets [this street], xCardsKept, xOpponentKept] -- or just 0's...
    if INCLUDE_HAND_CONTEXT:
        num_input_cards += 17

    # Shape is [cards + bits] x height x width
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, num_input_cards, input_height, input_width),
        )

    print('input layer shape %d x %d x %d x %d' % (batch_size, num_input_cards, input_height, input_width))

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=NUM_FILTERS, #16, #32,
        filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
        border_mode=BORDER_SHAPE, # full = pads to prev shape "valid" = shrinks [bad for small input sizes]
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    print('convolution layer l_conv1. Shape %s' % str(l_conv1.output_shape))

    # No hard rule that we need to pool after every 3x3!
    # l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1, ds=(2, 2))
    l_conv1_1 = lasagne.layers.Conv2DLayer(
        l_conv1,
        num_filters=NUM_FILTERS, #16, #32,
        filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
        border_mode=BORDER_SHAPE, # full = pads to prev shape "valid" = shrinks [bad for small input sizes]
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )
    print('convolution layer l_conv1_1. Shape %s' % str(l_conv1_1.output_shape))

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, pool_size=(2, 2))
    # Try *not pooling* in the suit layer...
    #l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, ds=(2, 1))

    print('maxPool layer l_pool1. Shape %s' % str(l_pool1.output_shape))

    # try 3rd conv layer
    #l_conv1_2 = lasagne.layers.Conv2DLayer(
    #    l_conv1_1,
    #    num_filters=16, #16, #32,
    #    filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
    #    nonlinearity=lasagne.nonlinearities.rectify,
    #    W=lasagne.init.GlorotUniform(),
    #    )
    #l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_2, ds=(2, 2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_pool1,
        num_filters=NUM_FILTERS*2, #16, #32,
        filter_size=(3,3), #(5,5), # (3,3), #(5, 5),
        border_mode=BORDER_SHAPE, # full = pads to prev shape "valid" = shrinks [bad for small input sizes]
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    print('convolution layer l_conv2. Shape %s' % str(l_conv2.output_shape))

    # Add 4th convolution layer...
    # l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2, ds=(2, 2))
    l_conv2_2 = lasagne.layers.Conv2DLayer(
        l_conv2,
        num_filters=NUM_FILTERS*2, #16, #32,
        filter_size=(3,3), #(5,5), # (3,3), #(5, 5),
        border_mode=BORDER_SHAPE, # full = pads to prev shape "valid" = shrinks [bad for small input sizes]
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    print('convolution layer l_conv2_2. Shape %s' % str(l_conv2_2.output_shape))

    # Question? No need for Max-pool for already narrow network... NO
    # Try *not pooling* in the suit layer...
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, pool_size=(2, 2))
    #l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, ds=(2, 1))

    print('maxPool layer l_pool2. Shape %s' % str(l_pool2.output_shape))

    # Add 3rd convolution layer!
    #l_conv3 = lasagne.layers.Conv2DLayer(
    #    l_pool2,
    #    num_filters=16, #16, #32,
    #    filter_size=(2,2), #(5,5), # (3,3), #(5, 5),
    #    nonlinearity=lasagne.nonlinearities.rectify,
    #    W=lasagne.init.GlorotUniform(),
    #    )
    #l_pool3 = lasagne.layers.MaxPool2DLayer(l_conv3, ds=(2, 2))

    l_hidden1 = lasagne.layers.DenseLayer(
        l_pool2, # l_pool3, # l_pool2,
        num_units=NUM_HIDDEN_UNITS,
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    print('hidden layer l_hidden1. Shape %s' % str(l_hidden1.output_shape))

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    print('dropout layer l_hidden1_dropout. Shape %s' % str(l_hidden1_dropout.output_shape))

    #l_hidden2 = lasagne.layers.DenseLayer(
    #     l_hidden1_dropout,
    #     num_units=NUM_HIDDEN_UNITS,
    #     nonlinearity=lasagne.nonlinearities.rectify,
    #     )
    #l_hidden2_dropout = lasagne.layers.DropoutLayer(l_hidden2, p=0.5)

    l_out = lasagne.layers.DenseLayer(
        l_hidden1_dropout, #l_hidden2_dropout, # l_hidden1_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.rectify, # Don't return softmax! #nonlinearity=lasagne.nonlinearities.softmax,
        W=lasagne.init.GlorotUniform(),
        )

    print('final layer l_out, into %d dimension. Shape %s' % (output_dim, str(l_out.output_shape)))

    return l_out


# Adjust from the standard create_iter_functions, to deal with z_batch being vector of values.
def create_iter_functions_full_output(dataset, output_layer,
                                      X_tensor_type=T.tensor4, # T.matrix,
                                      batch_size=BATCH_SIZE,
                                      learning_rate=LEARNING_RATE, momentum=MOMENTUM):
    """Create functions for training, validation and testing to iterate one
       epoch.
    """
    print('creating iter funtions')

    print('input dataset %s' % dataset)

    batch_index = T.iscalar('batch_index')
    X_batch = X_tensor_type('x') # inputs to the network
    y_batch = T.ivector('y') # "best class" for the network
    z_batch = T.matrix('z') # all values for the network
    if TRAIN_MASKED_OBJECTIVE:
        m_batch = T.matrix('m') # mask for all values, if some are relevant, others are N/A or ?

    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    
    # Also, attempt to multiply the 5 [bet, raise, check, call, fold] values 
    # by the 5 [bet, raise, check, call, fold] action percentages.
    # Can we do this with slicing?
    #vals_batch = T.matrix('vals') # 100x5 matrix of value estimates
    #actions_batch = T.matrix('actions') # 100x5 matric of value estimates



    # Use categorical_crossentropy objective, if we want to just predict best class, and not class values.
    #objective = lasagne.objectives.Objective(output_layer,
    #    loss_function=lasagne.objectives.categorical_crossentropy)
    #loss_train = objective.get_loss(X_batch, target=y_batch)
    #loss_eval = objective.get_loss(X_batch, target=y_batch,
    #                               deterministic=True)


    if not TRAIN_MASKED_OBJECTIVE:
        # compute loss on mean squared error
        objective_no_mask = lasagne.objectives.Objective(output_layer,
                                                         #loss_function=linear_error) # just not as good...
                                                         loss_function=lasagne.objectives.mse)

        # error is comparing output to z-vector.
        loss_train_no_mask = objective_no_mask.get_loss(X_batch, target=z_batch)
        loss_eval_no_mask = objective_no_mask.get_loss(X_batch, target=z_batch,
                                                       deterministic=True)
    else:
        # Alternatively, train only on some values! We do this by supplying a mask.
        # This means that some values matter, others do not. For example... 
        # A. Train on "best value" for 32-draws only (or random value, etc)
        # B. If we get records of actual hands played, train only on moves actually made (but output_layer produces value for all moves)
        if LINEAR_LOSS_FOR_MASKED_OBJECTIVE:
            # Use this, if cases highly random, and better to optimize for the average...
            masked_loss_function = linear_error
        else:
            # Use this, if results correct, or mostly correct, and we need this well reflected.
            #masked_loss_function = lasagne.objectives.mse

            # Test this hack...
            masked_loss_function=value_action_error

        objective_mask = lasagne.objectives.MaskedObjective(output_layer, masked_loss_function)
                                                  
        # error is computed same as un-masked objective... but we also supply a mask for each output. 1 = output matters 0 = N/A or ?
        loss_train_mask = objective_mask.get_loss(X_batch, target=z_batch, mask=m_batch)
        loss_eval_mask = objective_mask.get_loss(X_batch, target=z_batch, mask=m_batch,
                                                 deterministic=True)

    if TRAIN_MASKED_OBJECTIVE:
        print('--> We are told to use \'masked\' loss function. So training & validation loss will be computed on inputs with mask == 1 only')
        objective = objective_mask
        loss_train = loss_train_mask
        loss_eval = loss_eval_mask
    else:
        objective = objective_no_mask
        loss_train = loss_train_no_mask
        loss_eval = loss_eval_no_mask
    
    # Prediction actually stays the same! Since we still want biggest value in the array... and compare to Y
    # NOTE: If accuracy really supposed to be first 5... use [:,0:5] below.
    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    # Default: Nesterov momentum. Try something else?
    print('Using updates.nesterov_momentum with learning rate %.2f, momentum %.2f' % (learning_rate, momentum))
    updates_nesterov = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

    # "AdaDelta" by Matt Zeiler -- no learning rate or momentum...
    print('Using AdaDelta adaptive learning after %d epochs, with epsilon %s, learning rate %.2f!' % (EPOCH_SWITCH_ADAPT, str(ADA_DELTA_EPSILON), ADA_LEARNING_RATE))
    updates_ada_delta = lasagne.updates.adadelta(loss_train, all_params, learning_rate=ADA_LEARNING_RATE, epsilon=ADA_DELTA_EPSILON) #learning_rate=learning_rate)

    # Be careful not to include in givens, what won't be used. Theano will complain!
    if TRAIN_MASKED_OBJECTIVE:
        givens_train = {
            X_batch: dataset['X_train'][batch_slice],
            # Not computing 'accuracy' on training... [though we should]
            #y_batch: dataset['y_train'][batch_slice],
            z_batch: dataset['z_train'][batch_slice],
            m_batch: dataset['m_train'][batch_slice],
            }
    else:
         givens_train = {
            X_batch: dataset['X_train'][batch_slice],
            # Not computing 'accuracy' on training... [though we should]
            #y_batch: dataset['y_train'][batch_slice],
            z_batch: dataset['z_train'][batch_slice],
            #m_batch: dataset['m_train'][batch_slice],
            }

    # Function to train with nesterov momentum...
    iter_train_nesterov = theano.function(
        [batch_index], loss_train,
        updates=updates_nesterov,
        givens=givens_train,
        )

    # Still the default training function
    iter_train = iter_train_nesterov

    # Try training with AdaDelta (adaptive training)
    iter_train_ada_delta= theano.function(
        [batch_index], loss_train,
        updates=updates_ada_delta,
        givens=givens_train,
        )

    # Be careful not to include in givens, what won't be used. Theano will complain!
    if TRAIN_MASKED_OBJECTIVE:
        givens_valid = {
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            z_batch: dataset['z_valid'][batch_slice],
            m_batch: dataset['m_valid'][batch_slice],
        }
    else:
        givens_valid = {
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            z_batch: dataset['z_valid'][batch_slice],
            #m_batch: dataset['m_valid'][batch_slice],
        }
    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens=givens_valid,
    )

    # Be careful not to include in givens, what won't be used. Theano will complain!
    if TRAIN_MASKED_OBJECTIVE:
        givens_test = {
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            z_batch: dataset['z_test'][batch_slice],
            m_batch: dataset['m_test'][batch_slice],
        }
    else:
        givens_test = {
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            z_batch: dataset['z_test'][batch_slice],
            #m_batch: dataset['m_test'][batch_slice],
        }
    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens=givens_test,
    )

    return dict(
        train=iter_train,
        train_nesterov=iter_train_nesterov,
        train_ada_delta=iter_train_ada_delta,
        valid=iter_valid,
        test=iter_test,
    )

# Now how do I return theano function to predict, from my given thing? Should be simple.
# TODO: output second-best choice
# TODO: show choices, alongside inputs
def predict_model(output_layer, test_batch, format = 'deuce'):
    print('Computing predictions on test_batch: %s %s' % (type(test_batch), test_batch.shape))
    #pred = T.argmax(output_layer.get_output(test_batch, deterministic=True), axis=1)
    #pred_all = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True) # deprecated
    pred_all = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=True)
    if format != 'deuce_events':
        pred = pred_all
    else:
        # Slice it, so only relevant rows are looked at
        pred = pred_all[:17,:10] # pred_all[:17,:5] # hack: show first 17 examples only

    # Print out entire predictions array
    opt = np.get_printoptions()
    np.set_printoptions(threshold='nan')
    print('Prediciton: %s' % pred) 
    # Return options to previous settings...
    np.set_printoptions(**opt)

    # TODO: For "deuce_events" format... show best action inside of events actions only.

    #print(tp.pprint(pred))
    softmax_values = pred.eval()
    print(softmax_values)

    pred_max = T.argmax(pred[:,0:5], axis=1)

    print('Maximums %s' % pred_max)
    #print(tp.pprint(pred_max))

    softmax_choices = pred_max.eval()
    print(softmax_choices)

    # now debug the softmax choices...
    if format != 'deuce_events':
        softmax_debug = [DRAW_VALUE_KEYS[i] for i in softmax_choices]
    else:
        softmax_debug = [eventCategoryName[i] for i in softmax_choices]
    print(softmax_debug)


# Get 0-BATCH_SIZE hands, evaluate, and return matrix of vectors
def evaluate_batch_hands(output_layer, test_cases, include_hand_context = INCLUDE_HAND_CONTEXT): 
    now = time.time()
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[0])
    
    # case = [hand_string, int(num_draws)]
    test_batch = np.array([cards_input_from_string(hand_string=case[0],
                                                   include_num_draws=INCLUDE_NUM_DRAWS, num_draws=case[1], 
                                                   include_full_hand = INCLUDE_FULL_HAND,
                                                   include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], TRAINING_INPUT_TYPE)

    # print('%.2fs to create BATCH_SIZE input' % (time.time() - now))
    now = time.time()

    pred = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=True)
    # print('%.2fs to get_output' % (time.time() - now))
    now = time.time()

    #print('Prediciton: %s' % pred)
    #print(tp.pprint(pred))
    softmax_values = pred.eval()
    print('%.2fs to eval() output' % (time.time() - now))
    now = time.time()

    return softmax_values


# Return 32-point vector.
# Make a batch, to evaluate single hand. Expensive!!
def evaluate_single_hand(output_layer, hand_string_dealt, num_draws = 1, 
                         include_hand_context = INCLUDE_HAND_CONTEXT):
    test_cases = [[hand_string_dealt, num_draws]]
    softmax_values = evaluate_batch_hands(output_layer, test_cases, include_hand_context = include_hand_context)
    return softmax_values[0]

# Just give us the bits... expect 26x17x17 matrix...
def evaluate_single_event(output_layer, event_input):
    now = time.time()
    test_batch = np.array([event_input for i in range(BATCH_SIZE)], TRAINING_INPUT_TYPE)
    # print('%.2fs to create BATCH_SIZE input' % (time.time() - now))
    now = time.time()

    #pred = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True) # deprecated...
    pred = lasagne.layers.get_output(output_layer, lasagne.utils.floatX(test_batch), deterministic=True)
    # print('%.2fs to get_output' % (time.time() - now))
    now = time.time()

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
        #print('all param values: %d' % len(all_param_values))

        # save values to output file!
        print('pickling model %d param values to %s' % (len(all_param_values), out_file))
        with open(out_file, 'wb') as f:
            pickle.dump(all_param_values, f, -1)

def main(num_epochs=NUM_EPOCHS, out_file=None):
    print("Loading data...")
    dataset = load_data()

    print("Building model and compiling functions...")
    sys.stdout.flush() # Helps keep track of output live in re-directed out
    output_layer = build_model(
        input_height=dataset['input_height'],
        input_width=dataset['input_width'],
        output_dim=dataset['output_dim'],
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
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file_with_type)
        print('Successfully initialized model with previous saved params!')
    else:
        print('No existing model file (or skipping intialization) %s' % out_file)


    #iter_funcs = create_iter_functions(
    #    dataset,
    #    output_layer,
    #    X_tensor_type=T.tensor4,
    #    )

    # Define iter functions differently. Since we now want to predict the entire vector. 
    iter_funcs = create_iter_functions_full_output(
        dataset,
        output_layer,
        X_tensor_type=T.tensor4,
        )

    print("Starting training...")
    now = time.time()
    try:
        # When do we switch to adaptive training? Problems with adapative training and events, so disable that.
        switch_adaptive_after = EPOCH_SWITCH_ADAPT
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
    test_cases = [['3h,3s,3d,5c,6d', 3], ['2h,3s,4d,6c,5s', 1], ['3s,2h,4d,8c,5s', 0],
                  ['3h,3s,3d,5c,6d', 0], ['2h,3s,4d,6c,5s', 2], ['3s,2h,4d,8c,5s', 3],
                  ['As,Ad,4d,3s,2c', 1], ['As,Ks,Qs,Js,Ts', 2], ['3h,4s,3d,5c,6d', 2],
                  ['8s,Ad,Kd,8c,Jd', 3], ['8s,Ad,2d,7c,Jd', 2], ['2d,7d,8d,9d,4d', 1], 
                  ['7c,8c,Tc,Js,Qh', 3], ['2c,8s,5h,8d,2s', 2],
                  ['[8s,9c,8c,Kd,7h]', 2], ['[Qh,3h,6c,5s,4s]', 1], ['[Jh,Td,9s,Ks,5s]', 1],
                  ['[6c,4d,Ts,Jc,6s]', 3], ['[4h,8h,2c,7d,3h]', 2], ['[2c,Ac,Tc,6d,3d]', 1], 
                  ['[Ad,3c,Tc,4d,5d]', 1], ['3d,Jc,7d,Ac,6s', 2],
                  ['7h,Kc,5s,2d,Tc', 3], ['5c,6h,Jc,7h,2d', 1], ['Ts,As,3s,2d,4h', 3]] 

    print('looking at some test cases: %s' % test_cases)

    # Fill in test cases to get to batch size?
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[1])
    # Test_batch... 5 cards, no 3-card "round" encoding.
    # test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)
    if TRAINING_FORMAT == 'video' or TRAINING_FORMAT == 'deuce':
        test_batch = np.array([cards_input_from_string(hand_string=case[0], include_num_draws=INCLUDE_NUM_DRAWS, num_draws=case[1], include_full_hand = INCLUDE_FULL_HAND, include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], np.int32)
    
    elif TRAINING_FORMAT == 'deuce_events':
        # TODO:  Add context, if made available...
        test_batch = np.array([cards_input_from_string(hand_string=case[0], include_num_draws=INCLUDE_NUM_DRAWS, num_draws=case[1], include_full_hand = INCLUDE_FULL_HAND, include_hand_context = INCLUDE_HAND_CONTEXT) for case in test_cases], np.int32)
        
    predict_model(output_layer=output_layer, test_batch=test_batch, format = TRAINING_FORMAT)

    print('again, the test cases: \n%s' % test_cases)    

    return output_layer


if __name__ == '__main__':
    # Additional flags... only if we inlcude in model
    extra_flags = ''
    if INCLUDE_FULL_HAND:
        extra_flags += '_full_hand'
    if INCLUDE_HAND_CONTEXT:
        extra_flags += '_hand_context'

    output_layer_filename = '%striple_draw_conv_%.2f_learn_rate_%d_epoch_adaptive_%d_filters_%s_border_%d_num_draws%s_model.pickle' % (TRAINING_FORMAT, LEARNING_RATE, EPOCH_SWITCH_ADAPT, NUM_FILTERS, BORDER_SHAPE, INCLUDE_NUM_DRAWS, extra_flags)
    main(out_file=output_layer_filename)
