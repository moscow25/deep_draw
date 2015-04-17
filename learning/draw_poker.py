"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

A system for playing basic video poker, and training/generating trianing data, for a neural network AI to learn how to play such games.

Hat tip to Erik of DeepPink, which implements a Theano system for chess.
https://github.com/erikbern/deep-pink

Here, we use Lasagne, a set of tools & examples for setting up a neural net with Theano, to train the network to predict values of video poker draws.
https://github.com/Lasagne/Lasagne

We based the first version of this code, on the Lasagne MNIST example.

Rather than trying to predict the best move of a hand, say [Kh,4s,5s,Qd,5c], we output the average value, for a given (hidden) payout table, for each draw option:

All 32 sim results for hand [Kh,4s,5s,Qd,5c]:
	[]:	1000 sample:	0.34 average	9.00 maximum
	[Kh]:	10000 sample:	0.47 average	25.00 maximum
	[4s]:	1000 sample:	0.28 average	9.00 maximum
	[5s]:	1000 sample:	0.22 average	9.00 maximum
	[Qd]:	10000 sample:	0.47 average	25.00 maximum
	[5c]:	1000 sample:	0.24 average	9.00 maximum
	[Kh,4s]:	1000 sample:	0.34 average	3.00 maximum
	[Kh,5s]:	1000 sample:	0.30 average	3.00 maximum
	[Kh,Qd]:	1000 sample:	0.52 average	4.00 maximum
	[Kh,5c]:	1000 sample:	0.28 average	3.00 maximum
	[4s,5s]:	1000 sample:	0.27 average	6.00 maximum
	[4s,Qd]:	1000 sample:	0.33 average	9.00 maximum
	[4s,5c]:	1000 sample:	0.21 average	9.00 maximum
	[5s,Qd]:	1000 sample:	0.32 average	9.00 maximum
	[5s,5c]:	1000 sample:	0.76 average	25.00 maximum
	[Qd,5c]:	1000 sample:	0.28 average	3.00 maximum
	[Kh,4s,5s]:	1000 sample:	0.19 average	3.00 maximum
	[Kh,4s,Qd]:	1000 sample:	0.32 average	3.00 maximum
	[Kh,4s,5c]:	1000 sample:	0.18 average	3.00 maximum
	[Kh,5s,Qd]:	1000 sample:	0.30 average	3.00 maximum
	[Kh,5s,5c]:	1000 sample:	0.58 average	9.00 maximum
	[Kh,Qd,5c]:	1000 sample:	0.29 average	3.00 maximum
	[4s,5s,Qd]:	1000 sample:	0.18 average	3.00 maximum
	[4s,5s,5c]:	1000 sample:	0.65 average	9.00 maximum
	[4s,Qd,5c]:	1000 sample:	0.19 average	3.00 maximum
	[5s,Qd,5c]:	1000 sample:	0.75 average	25.00 maximum
	[Kh,4s,5s,Qd]:	1000 sample:	0.14 average	1.00 maximum
	[Kh,4s,5s,5c]:	1000 sample:	0.39 average	3.00 maximum
	[Kh,4s,Qd,5c]:	1000 sample:	0.11 average	1.00 maximum
	[Kh,5s,Qd,5c]:	1000 sample:	0.38 average	3.00 maximum
	[4s,5s,Qd,5c]:	1000 sample:	0.41 average	3.00 maximum
	[Kh,4s,5s,Qd,5c]:	1000 sample:	0.00 average	0.00 maximum

best result:
	[5s,5c]:	1000 sample:	0.76 average	25.00 maximum
""" 

from __future__ import print_function

import gzip
import itertools
import pickle
import os
import sys
import numpy as np
import lasagne
import theano
import theano.tensor as T
import theano.printing as tp # theano.printing.pprint(variable) [tp.pprint(var)]
import time
import math
import csv
from poker_lib import *
from poker_util import *

PY2 = sys.version_info[0] == 2

if PY2:
    from urllib import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f)
else:
    from urllib.request import urlretrieve

    def pickle_load(f, encoding):
        return pickle.load(f, encoding=encoding)

DATA_URL = '' # 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
DATA_FILENAME = '../data/300k_full_sim_samples.csv' # '../data/100k_full_sim_samples.csv' # '../data/20000_full_sim_samples.csv' # '../data/100k_full_sim_samples.csv' #'../data/40000_full_sim_samples.csv'
# Not too much accuracy gain... in doubling the training data. And more than 2x as slow.
# '../data/20000_full_sim_samples.csv'

MAX_INPUT_SIZE = 1000000 # Remove this constraint, as needed
VALIDATION_SIZE = 1000
TEST_SIZE = 1000
NUM_EPOCHS = 100 # 20 # 100
BATCH_SIZE = 100 # 50 #100
NUM_HIDDEN_UNITS = 512
LEARNING_RATE = 0.02 # Try faster learning # 0.01
MOMENTUM = 0.9

# Do we include all data? Sure... but maybe not.
# If we keep everything, almost 60% of data falls into "keep 2" nodes. 
DATA_SAMPLING_KEEP_ALL = [1.0 for i in range(32)]
# Choose 50% of "keep two cards" moves, and 50% of "keep one card" moves
DATA_SAMPLING_REDUCE_KEEP_TWO = [1.0] + [0.5] * 5 + [0.5] * 10 + [1.0] * 10 + [1.0] * 5 + [1.0] 

# returns numpy array 5x4x13, for card hand string like '[Js,6c,Ac,4h,5c]' or 'Tc,6h,Kh,Qc,3s'
# if pad_to_fit... pass along to card input creator, to create 14x14 array instead of 4x13
def cards_input_from_string(hand_string, pad_to_fit = True):
    hand_array = hand_string_to_array(hand_string)

    # Now turn the array of Card abbreviations into numpy array of of input
    cards_array = [Card(suit=suitFromChar[card_str[1]], value=valueFromChar[card_str[0]]) for card_str in hand_array]
    assert(len(cards_array) == 5)
    cards_input = np.array([card_to_matrix(cards_array[0], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[1], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[2], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[3], pad_to_fit=pad_to_fit), card_to_matrix(cards_array[4], pad_to_fit=pad_to_fit)], np.int32)

    return cards_input

# a hack, since we aren't able to implement liner loss in Theano...
# To remove, or reduce problems with really big values... map values above 2.0 points to sqrt(remainder)
# 25.00 to 6.80
# 50.00 to 8.93
def adjust_float_value(hand_val):
    # Use small value if mean square error. Big value if linear error...
    base = 4.0 # 50.0 # 2.0
    if hand_val <= base:
        return hand_val
    else:
        remainder = hand_val - base
        remainder_reduce = math.sqrt(remainder)
        total = base + remainder_reduce
        #print('regressing high hand value %.2f to %.2f' % (hand_val, total))
        return total

# Turn each hand into an input (cards array) + output (32-line value)
# if output_best_class==TRUE, instead outputs index 0-32 of the best value (for softmax category output)
# Why? A. Easier to train B. Can re-use MNIST setup.
def read_poker_line(data_array, csv_key_map):
    # Get the input hand, and into an array
    """
    hand_array = hand_string_to_array(data_array[csv_key_map['hand']])

    # Now turn the array of Card abbreviations into numpy array of of input
    cards_array = [Card(suit=suitFromChar[card_str[1]], value=valueFromChar[card_str[0]]) for card_str in hand_array]
    assert(len(cards_array) == 5)
    cards_input = np.array([card_to_matrix(cards_array[0]), card_to_matrix(cards_array[1]), card_to_matrix(cards_array[2]), card_to_matrix(cards_array[3]), card_to_matrix(cards_array[4])], np.int32)
    """

    cards_input = cards_input_from_string(data_array[csv_key_map['hand']])

    # Ok now translate the 32-low output row.
    output_values = [adjust_float_value(float(data_array[csv_key_map[draw_value_key]])) for draw_value_key in DRAW_VALUE_KEYS] # np.array([float(data_array[csv_key_map[draw_value_key]]) for draw_value_key in DRAW_VALUE_KEYS], np.float32)
    best_output = max(output_values)
    output_category = output_values.index(best_output)

    """
    if output_best_class:
        return (cards_input, output_category)
    else:
        return (cards_input, output_values)   
        """

    # Output all three things. Cards input, best category (0-32) and 32-row vector, of the weights
    return (cards_input, output_category, output_values) 
    
# Read CSV lines, create giant numpy arrays of the input & output values.
def _load_poker_csv(filename=DATA_FILENAME, max_input=MAX_INPUT_SIZE, output_best_class=True):
    csv_reader = csv.reader(open(filename, 'rU'))

    csv_key = None
    csv_key_map = None

    # Can't grow numpy arrays. So instead, be lazy and grow array to be turned into numpy arrays.
    X_train_not_np = []
    y_train_not_np = []
    z_train_not_np = [] # 32-length vectors for all weights
    hands = 0
    sampling_policy = DATA_SAMPLING_REDUCE_KEEP_TWO # DATA_SAMPLING_KEEP_ALL
    # compute histogram of how many hands, output "correct draw" to each of 32 choices
    y_count_by_bucket = [0 for i in range(32)] 
    for line in csv_reader:
        # Read the CSV key, so we look for columns in the data, not fixed positions.
        if not csv_key:
            print('CSV key' + str(line))
            csv_key = line
            csv_key_map = CreateMapFromCSVKey(csv_key)
        else:
            # Skip any mail-formed lines.
            try:
                hand_input, output_class, output_array = read_poker_line(line, csv_key_map)
            except (IndexError, ValueError):
                print('\nskipping malformed input line:\n|%s|\n' % line)
                continue

	    if (hands % 10000) == 0:
		print('Loaded %d hands...' % hands)

            # Assumes that output_array is really just 0-31 for best class.
            # TODO: Just output array and class every time, choose which one to use!
            #output_class = output_array

            # If requested, sample out some too common cases. 
            # A. Better balance
            # B. Faster training
            if sampling_policy:
                output_percent = sampling_policy[output_class]
                if random.random() > output_percent:
                    #print('(item skipped for class %s' % DRAW_VALUE_KEYS[output_class])
                    continue

            # count class, if item chosen
            y_count_by_bucket[output_class] += 1

            X_train_not_np.append(hand_input)
            y_train_not_np.append(output_class)
            z_train_not_np.append(output_array)

            hands += 1

            if hands >= max_input:
                break

            #sys.exit(-3)

    # Show histogram... of counts by 32 categories.
    print('count ground truth for 32 categories:\n%s\n' % ('\n'.join([str([DRAW_VALUE_KEYS[i],y_count_by_bucket[i],'%.1f%%' % (y_count_by_bucket[i]*100.0/hands)]) for i in range(32)])))

    X_train = np.array(X_train_not_np)
    y_train = np.array(y_train_not_np)
    z_train = np.array(z_train_not_np) # 32-length vectors

    print('Read %d data points. Shape below:' % len(X_train_not_np))
    #print(X_train)
    #print(y_train)
    print('num_examples (train) %s' % X_train.shape[0])
    print('input_dimensions %s' % X_train.shape[1])
    print('X_train object is type %s of shape %s' % (type(X_train), X_train.shape))
    print('y_train object is type %s of shape %s' % (type(y_train), y_train.shape))
    print('z_train object is type %s of shape %s' % (type(z_train), z_train.shape))

    return (X_train, y_train, z_train)

def load_data():
    """Get data with labels, split into training, validation and test set."""
    data = _load_poker_csv()

    # We ignore z (all values array)
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

    #sys.exit(0)

    # We ignore validation & test for now.
    #X_valid, y_valid = data[1]
    #X_test, y_test = data[2]

    return dict(
        X_train=theano.shared(lasagne.utils.floatX(X_train)),
        y_train=T.cast(theano.shared(y_train), 'int32'),
        X_valid=theano.shared(lasagne.utils.floatX(X_valid)),
        y_valid=T.cast(theano.shared(y_valid), 'int32'),
        X_test=theano.shared(lasagne.utils.floatX(X_test)),
        y_test=T.cast(theano.shared(y_test), 'int32'),
        num_examples_train=X_train.shape[0],
        num_examples_valid=X_valid.shape[0],
        num_examples_test=X_test.shape[0],
        input_dim=X_train.shape[1] * X_train.shape[2] * X_train.shape[3], # How much size per input?? 5x4x13 data (cards, suits, ranks)
        output_dim=32, # output cases
    )

def build_model(input_dim, output_dim,
                batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
    """Create a symbolic representation of a neural network with `intput_dim`
    input nodes, `output_dim` output nodes and `num_hidden_units` per hidden
    layer.

    The training function of this model must have a mini-batch size of
    `batch_size`.

    A theano expression which represents such a network is returned.
    """
    print('building model for input_dim: %s, output_dim: %s, batch_size: %d, num_hidden_units: %d' % (input_dim, output_dim,
                                                                                                      batch_size, num_hidden_units))

    print('creating l_in layer')
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_dim),
    )
    print('creating l_hidden1 layer')
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    print('creating l_hidden1_dropout layer')
    l_hidden1_dropout = lasagne.layers.DropoutLayer(
        l_hidden1,
        p=0.5,
    )
    print('creating l_hidden2 layer')
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1_dropout,
        num_units=num_hidden_units,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    print('creating l_hidden2_dropout layer')
    l_hidden2_dropout = lasagne.layers.DropoutLayer(
        l_hidden2,
        p=0.5,
    )
    print('creating dense layer')
    l_out = lasagne.layers.DenseLayer(
        l_hidden2_dropout,
        num_units=output_dim,
        nonlinearity=lasagne.nonlinearities.softmax,
    )
    return l_out

# X_tensor_type=T.matrix --> for matrix (pixel) input
# need to change it... T.tensor4 [4-d matrix]
# http://deeplearning.net/software/theano/library/tensor/basic.html
def create_iter_functions(dataset, output_layer,
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

    y_batch = T.ivector('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)

    objective = lasagne.objectives.Objective(output_layer,
        loss_function=lasagne.objectives.categorical_crossentropy)

    loss_train = objective.get_loss(X_batch, target=y_batch)
    loss_eval = objective.get_loss(X_batch, target=y_batch,
                                   deterministic=True)

    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)

    all_params = lasagne.layers.get_all_params(output_layer)
    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    # Add a function.... to evaluate the result of an input!
    #evaluate = theano.function()

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: dataset['X_train'][batch_slice],
            y_batch: dataset['y_train'][batch_slice],
        },
    )

    iter_valid = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_valid'][batch_slice],
            y_batch: dataset['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_eval, accuracy],
        givens={
            X_batch: dataset['X_test'][batch_slice],
            y_batch: dataset['y_test'][batch_slice],
        },
    )

    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
    )


def train(iter_funcs, dataset, batch_size=BATCH_SIZE):
    """Train the model with `dataset` with mini-batch training. Each
       mini-batch has `batch_size` recordings.
    """
    num_batches_train = dataset['num_examples_train'] // batch_size
    num_batches_valid = dataset['num_examples_valid'] // batch_size

    for epoch in itertools.count(1):
        batch_train_losses = []
        for b in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](b)
            batch_train_losses.append(batch_train_loss)

        avg_train_loss = np.mean(batch_train_losses)

        batch_valid_losses = []
        batch_valid_accuracies = []
        for b in range(num_batches_valid):
            batch_valid_loss, batch_valid_accuracy = iter_funcs['valid'](b)
            batch_valid_losses.append(batch_valid_loss)
            batch_valid_accuracies.append(batch_valid_accuracy)

        avg_valid_loss = np.mean(batch_valid_losses)
        avg_valid_accuracy = np.mean(batch_valid_accuracies)

        yield {
            'number': epoch,
            'train_loss': avg_train_loss,
            'valid_loss': avg_valid_loss,
            'valid_accuracy': avg_valid_accuracy,
        }


"""
# Borrowed from DeepPink chess.
# Apply function to (single) input
def get_model_from_pickle(fn):
    f = open(fn)
    Ws, bs = pickle.load(f)
    
    Ws_s, bs_s = train.get_parameters(Ws=Ws, bs=bs)
    x, p = train.get_model(Ws_s, bs_s)
    
    predict = theano.function(
        inputs=[x],
        outputs=p)

    return predict
    """

#def get_predict_function(output_layer, 



# Now how do I return theano function to predict, from my given thing? Should be simple.
def predict_model(output_layer, test_batch):
    print('Computing predictions on test_batch: %s %s' % (type(test_batch), test_batch.shape))
    #pred = T.argmax(output_layer.get_output(test_batch, deterministic=True), axis=1)
    pred = output_layer.get_output(lasagne.utils.floatX(test_batch), deterministic=True)
    print('Prediciton: %s' % pred)
    print(tp.pprint(pred))
    softmax_values = pred.eval()
    print(softmax_values)

    #f = theano.function([output_layer.get_output], test_batch)

    #print(f)

    #print('Predition eval: %' % pred.eval

    """
    pred = T.argmax(
        output_layer.get_output(X_batch, deterministic=True), axis=1)
    accuracy = T.mean(T.eq(pred, y_batch), dtype=theano.config.floatX)
    """

    pred_max = T.argmax(pred, axis=1)

    print('Maximums %s' % pred_max)
    print(tp.pprint(pred_max))

    softmax_choices = pred_max.eval()
    print(softmax_choices)

    # now debug the softmax choices...
    softmax_debug = [DRAW_VALUE_KEYS[i] for i in softmax_choices]
    print(softmax_debug)
    


def main(num_epochs=NUM_EPOCHS):
    print("Loading data...")
    dataset = load_data()

    print(dataset)

    #sys.exit(-3)

    print("Building model and compiling functions...")
    output_layer = build_model(
        input_dim=dataset['input_dim'],
        output_dim=dataset['output_dim'],
    )
    iter_funcs = create_iter_functions(dataset, output_layer)

    print("\n\nStarting training...")
    now = time.time()
    try:
        for epoch in train(iter_funcs, dataset):
            print("Epoch {} of {} took {:.3f}s".format(
                epoch['number'], num_epochs, time.time() - now))
            now = time.time()
            print("  training loss:\t\t{:.6f}".format(epoch['train_loss']))
            print("  validation loss:\t\t{:.6f}".format(epoch['valid_loss']))
            print("  validation accuracy:\t\t{:.2f} %%".format(
                epoch['valid_accuracy'] * 100))

            if epoch['number'] >= num_epochs:
                break

    except KeyboardInterrupt:
        pass


    # Can we do something with final output model? Like show a couple of moves...
    #test_batch = dataset['X_test']

    # Test cases
    test_cases = ['As,Ad,4d,3s,2c', 'As,Ks,Qs,Js,Ts', '3h,4s,3d,5c,6d', '2h,3s,4d,6c,5s',
                  '8s,Ad,Kd,8c,Jd', '8s,Ad,2d,7c,Jd', '[8s,9c,8c,Kd,7h]', '[Qh,3h,6c,5s,4s]', '[Jh,Td,9s,Ks,5s]',
                  '[6c,4d,Ts,Jc,6s]', '[4h,8h,2c,7d,3h]', '[2c,Ac,Tc,6d,3d]'] 

    print('looking at some test cases: %s' % test_cases)

    # Fill in test cases to get to batch size?
    #for i in range(BATCH_SIZE - len(test_cases)):
    #    test_cases.append(test_cases[1])
    test_batch = np.array([cards_input_from_string(case) for case in test_cases], np.int32)
    predict_model(output_layer=output_layer, test_batch=test_batch)

    print('again, the test cases: \n%s' % test_cases)

    #print(predict(output_layer[x=test_batch]))

    return output_layer


if __name__ == '__main__':
    main()
