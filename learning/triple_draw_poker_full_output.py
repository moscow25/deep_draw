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

DATA_FILENAME = '../data/200k_hands_sample_details_all.csv' # all 32 values. Cases for 1, 2 & 3 draws left
# '../data/60000_hands_sample_details.csv' # 60k triple draw hands... best draw output only

###MAX_INPUT_SIZE = 10000 # 10000000 # Remove this constraint, as needed
VALIDATION_SIZE = 5000
TEST_SIZE = 5000
NUM_EPOCHS = 500 # 20 # 20 # 100
BATCH_SIZE = 100 # 50 #100
BORDER_SHAPE = "valid" # "same" # "valid" # "full" = pads to prev shape "valid" = shrinks [bad for small input sizes]
NUM_FILTERS = 16 # 32 # 16 # increases 2x at higher level
NUM_HIDDEN_UNITS = 1024 # 512 # 256 #512
LEARNING_RATE = 0.1 # 0.1 #  0.05 # 0.01 # 0.02 # 0.01
MOMENTUM = 0.9
EPOCH_SWITCH_ADAPT = 10 # 30 # switch to adaptive training after X epochs of learning rate & momentum with Nesterov
ADA_DELTA_EPSILON = 1e-4 # 1e-6 # default is smaller, be more aggressive...
ADA_LEARNING_RATE = 0.5 # algorithm confuses this

# Here, we get into growing input information, beyond the 5-card hand.
INCLUDE_NUM_DRAWS = True # 3 "cards" to encode number of draws left. ex. 2 draws: [0], [1], [1]

def load_data():
    # Do *not* bias the data, or smooth out big weight values, as we would for video poker.
    # 'deuce' has its own adjustments...
    data = _load_poker_csv(filename=DATA_FILENAME, keep_all_data=True, adjust_floats='deuce', include_num_draws = INCLUDE_NUM_DRAWS)

    # X = input, y = best cateogy, z = all categories.
    X_all, y_all, z_all = data

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

    #sys.exit(0)

    # We ignore validation & test for now.
    #X_valid, y_valid = data[1]
    #X_test, y_test = data[2]

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        z_train=theano.shared(lasagne.utils.floatX(z_train)),

        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        z_valid=theano.shared(lasagne.utils.floatX(z_valid)),

        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        z_test=theano.shared(lasagne.utils.floatX(z_test)),

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
    if INCLUDE_NUM_DRAWS:
        num_input_cards += 3

    l_in = lasagne.layers.InputLayer(
        # Shape is *5* x width x height 
        shape=(batch_size, num_input_cards, input_width, input_height),
        )

    print('input layer shape %d x %d x %d x %d' % (batch_size, num_input_cards, input_width, input_height))

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in,
        num_filters=NUM_FILTERS, #16, #32,
        filter_size=(3,3), #(5,5), #(3,3), #(5, 5),
        border_mode=BORDER_SHAPE, # full = pads to prev shape "valid" = shrinks [bad for small input sizes]
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform(),
        )

    print('convolution layer l_conv1. Shape %s' % str(l_conv1.get_output_shape()))

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
    print('convolution layer l_conv1_1. Shape %s' % str(l_conv1_1.get_output_shape()))

    l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, ds=(2, 2))
    # Try *not pooling* in the suit layer...
    #l_pool1 = lasagne.layers.MaxPool2DLayer(l_conv1_1, ds=(2, 1))

    print('maxPool layer l_pool1. Shape %s' % str(l_pool1.get_output_shape()))

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

    print('convolution layer l_conv2. Shape %s' % str(l_conv2.get_output_shape()))

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

    print('convolution layer l_conv2_2. Shape %s' % str(l_conv2_2.get_output_shape()))

    # Question? No need for Max-pool for already narrow network... NO
    # Try *not pooling* in the suit layer...
    l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, ds=(2, 2))
    #l_pool2 = lasagne.layers.MaxPool2DLayer(l_conv2_2, ds=(2, 1))

    print('maxPool layer l_pool2. Shape %s' % str(l_pool2.get_output_shape()))

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

    print('hidden layer l_hidden1. Shape %s' % str(l_hidden1.get_output_shape()))

    l_hidden1_dropout = lasagne.layers.DropoutLayer(l_hidden1, p=0.5)

    print('dropout layer l_hidden1_dropout. Shape %s' % str(l_hidden1_dropout.get_output_shape()))

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

    print('final layer l_out, into %d dimension. Shape %s' % (output_dim, str(l_out.get_output_shape())))

    return l_out

# HACK linear error
def linear_error(x, t):
    return abs(x - t)

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
    X_batch = X_tensor_type('x')
    y_batch = T.ivector('y') # We don't replace Y. Just compute loss from
    z_batch = T.matrix('z') # 2D vector

    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    #objective = lasagne.objectives.Objective(output_layer,
    #    loss_function=lasagne.objectives.categorical_crossentropy)
    #loss_train = objective.get_loss(X_batch, target=y_batch)
    #loss_eval = objective.get_loss(X_batch, target=y_batch,
    #                               deterministic=True)

    # compute loss on mean squared error
    objective = lasagne.objectives.Objective(output_layer,
                                             #loss_function=linear_error) # just not as good...
                                             loss_function=lasagne.objectives.mse)

    # error is comparing output to z-vector.
    loss_train = objective.get_loss(X_batch, target=z_batch)
    loss_eval = objective.get_loss(X_batch, target=z_batch,
                                   deterministic=True)
    
    # Prediction actually stays the same! Since we still want biggest value in the array... and compare to Y
    pred = T.argmax(output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    # Default: Nesterov momentum. Try something else?
    print('Using updates.nesterov_momentum with learning rate %.2f, momentum %.2f' % (learning_rate, momentum))
    updates_nesterov = lasagne.updates.nesterov_momentum(loss_train, all_params, learning_rate, momentum)

    # "AdaDelta" by Matt Zeiler -- no learning rate or momentum...
    print('Using AdaDelta adaptive learning after %d epochs, with epsilon %s, learning rate %.2f!' % (EPOCH_SWITCH_ADAPT, str(ADA_DELTA_EPSILON), ADA_LEARNING_RATE))
    updates_ada_delta = lasagne.updates.adadelta(loss_train, all_params, learning_rate=ADA_LEARNING_RATE, epsilon=ADA_DELTA_EPSILON) #learning_rate=learning_rate)

    # Function to train with nesterov momentum...
    iter_train_nesterov = theano.function(
        [batch_index], loss_train,
        updates=updates_nesterov,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            # Not computing 'accuracy' on training... [though we should]
            #y_batch: dataset['y_train'][batch_slice],
            z_batch: dataset['z_train'][batch_slice],
        },
    )

    # Still the default training function
    iter_train = iter_train_nesterov

    # Try training with AdaDelta (adaptive training)
    iter_train_ada_delta= theano.function(
        [batch_index], loss_train,
        updates=updates_ada_delta,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            # Not computing 'accuracy' on training... [though we should]
            #y_batch: dataset['y_train'][batch_slice],
            z_batch: dataset['z_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
            z_batch: dataset['z_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
            z_batch: dataset['z_test'][batch_slice],
        },
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
def predict_model(output_layer, test_batch):
    print('Computing predictions on test_batch: %s %s' % (type(test_batch), test_batch.shape))
    #pred = T.argmax(output_layer.get_output(test_batch, deterministic=True), axis=1)
    pred = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True)
    print('Prediciton: %s' % pred)
    #print(tp.pprint(pred))
    softmax_values = pred.eval()
    print(softmax_values)

    pred_max = T.argmax(pred, axis=1)

    print('Maximums %s' % pred_max)
    #print(tp.pprint(pred_max))

    softmax_choices = pred_max.eval()
    print(softmax_choices)

    # now debug the softmax choices...
    softmax_debug = [DRAW_VALUE_KEYS[i] for i in softmax_choices]
    print(softmax_debug)


# Get 0-BATCH_SIZE hands, evaluate, and return matrix of vectors
def evaluate_batch_hands(output_layer, test_cases): 
    now = time.time()
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[0])
    test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)

    print('%.2fs to create BATCH_SIZE input' % (time.time() - now))
    now = time.time()

    pred = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True)
    print('%.2fs to get_output' % (time.time() - now))
    now = time.time()

    #print('Prediciton: %s' % pred)
    #print(tp.pprint(pred))
    softmax_values = pred.eval()
    print('%.2fs to eval() output' % (time.time() - now))
    now = time.time()

    return softmax_values


# Return 32-point vector.
# Make a batch, to evaluate single hand. Expensive!!
def evaluate_single_hand(output_layer, hand_string_dealt): #, test_batch = None, test_cases = TEST_CASES):
    test_cases = [hand_string_dealt]
    softmax_values = evaluate_batch_hands(output_layer, test_cases)
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
        all_param_values_from_file = np.load(out_file)
        print('Loaded values %d' % len(all_param_values_from_file))
        lasagne.layers.set_all_param_values(output_layer, all_param_values_from_file)
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
        for epoch in train(iter_funcs, dataset, epoch_switch_adapt=EPOCH_SWITCH_ADAPT):
            sys.stdout.flush() # Helps keep track of output live in re-directed out
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
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
    test_cases = ['As,Ad,4d,3s,2c', 'As,Ks,Qs,Js,Ts', '3h,3s,3d,5c,6d', '3h,4s,3d,5c,6d', '2h,3s,4d,6c,5s',
                  '8s,Ad,Kd,8c,Jd', '8s,Ad,2d,7c,Jd', '2d,7d,8d,9d,4d', '7c,8c,Tc,Js,Qh', '2c,8s,5h,8d,2s',
                  '[8s,9c,8c,Kd,7h]', '[Qh,3h,6c,5s,4s]', '[Jh,Td,9s,Ks,5s]', '[6c,4d,Ts,Jc,6s]', 
                  '[4h,8h,2c,7d,3h]', '[2c,Ac,Tc,6d,3d]', '[Ad,3c,Tc,4d,5d]', '3d,Jc,7d,Ac,6s',
                  '7h,Kc,5s,2d,Tc', '5c,6h,Jc,7h,2d', 'Ts,As,3s,2d,4h'] 

    print('looking at some test cases: %s' % test_cases)

    # Fill in test cases to get to batch size?
    for i in range(BATCH_SIZE - len(test_cases)):
        test_cases.append(test_cases[1])
    test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)
    predict_model(output_layer=output_layer, test_batch=test_batch)

    print('again, the test cases: \n%s' % test_cases)    

    return output_layer


if __name__ == '__main__':

    output_layer_filename = 'triple_draw_conv_%.2f_learn_rate_%d_epoch_adaptive_%d_filters_%s_border_%d_num_draws_model.pickle' % (LEARNING_RATE, EPOCH_SWITCH_ADAPT, NUM_FILTERS, BORDER_SHAPE, INCLUDE_NUM_DRAWS)
    main(out_file=output_layer_filename)
