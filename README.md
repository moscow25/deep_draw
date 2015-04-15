# deep_draw
Convolution neural network... for draw video poker. Perhaps, we learn something useful for other poker, too.

Assume Python 2.7, and some modules require Theano.

Other modules depend on Lasagne (a wrapper for easier Theano building). https://github.com/Lasagne/Lasagne
Install Lasagne to access.

You'll also need to add poker_lib to your PYTHONPATH... since I don't have install script yet for this module.
export DRAW_POKER_PYTHON_PATH=$HOME/deep_draw/poker-lib
export PYTHONPATH=$DRAW_POKER_PYTHON_PATH:$PYTHONPATH:.

Basic use cases...

Generate data: python simulate_draw_values.py outfile.csv
Play draw hands with a decision process (random decision is default): python play_draw.py

Train a neural network (non-convolution): python draw_poker.py
Train a convolution network to predict best choice: 
Train a convolution network to predict average values for all 32 draw conditions: python draw_poker_conv_full_output.py 

