#!/bin/bash

# Shell script for user to play against the latest Poker-CNN triple draw player.
# A few things...
# -> Absolute path
# -> Record user name
# -> Save results in user's directory
# Beyond that... keep it simple.

DEEP_DRAW_PATH="/home/ubuntu/deep_draw"
LOCAL_DRAW_PATH="/Users/kolya/Desktop/ML/learning/deep_draw"
if [ -d "$LOCAL_DRAW_PATH" ]; then
  DEEP_DRAW_PATH=$LOCAL_DRAW_PATH
fi

# Holdem categories
DRAW_MODEL="$DEEP_DRAW_PATH/learning/holdem_conv_24_filter_xCards_xNumDraws_x0_9831_percent_basline_800k.pickle"

# *Fresh* train on 700k new examples, directly from holdem values model. Higher validation loss, but hopefully less over-training.
# 2.1... but actually quite a different model.
CNN_2_1_BETS="$DEEP_DRAW_PATH/learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_2_1_canonical_clean_start_20_epoch_700k.pickle"

# Quick train on (mostly) same data, but with all hands in canonical form. See how it plays?
CNN_2_0_BETS="$DEEP_DRAW_PATH/learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_2_0_canonical_20_epoch_700k.pickle"

# Latest model, trained on self-play with 50/50 action percent, on CNN_16 and CNN_15
# Plays pretty good... but two issues. A. More passive. B. Still betting-action 100% when checked to. 
# TODO: Train on huge number of cases... like 5M. That's what is needed here. 
CNN_1_7_BETS="$DEEP_DRAW_PATH/learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_1_7_trained_CNN_1_65_700k.pickle"

# First legit-good HE model.
CNN_1_6_BETS="$DEEP_DRAW_PATH/learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_1_6_trained_CNN_1_5_700k.pickle"
# Kind of ok model, right before. Focuses on good and bad hands.
CNN_1_5_BETS="$DEEP_DRAW_PATH/learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_1_5_special_cases_play_board_700k.pickle"

# Up to three models that we can use.
# TODO: It's ok to use CNN_8 with DNN moves..... if we do draws model with 1/3 choice also. Otherwise... the crazy pats are too predictable
LATEST_VALUES_MODEL=$CNN_2_1_BETS
OLDER_VALUES_MODEL=$CNN_2_0_BETS 
OTHER_VALUES_MODEL=$CNN_1_7_BETS

# TODO: Dense Holdem model.
#DENSE_MODEL="$DEEP_DRAW_PATH/learning/deuce_bets_dense_2_layer_dropout_x0_5308_init_w_deuce_draws_700k.pickle"

# Grep user's name. Pass it to the play_triple_draw.py for recording into CSV
USER_NAME=$USER
tmp_pass=`head -c 10 /dev/random | base64`
RANDOM_SESSION_ID="${tmp_pass:0:10}" #cut to 10 characters after base64 conversion
USER_OUTPUT="$USER_NAME-holdem-vs-CNN16-$RANDOM_SESSION_ID.csv"

# First (best) model
python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -holdem_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL --human_player -output=./$USER_OUTPUT"

# Mix in second-best
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -holdem_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL -CNN_old_model $OLDER_VALUES_MODEL --human_player -output=./$USER_OUTPUT"

# Include third, possibly bad, model
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -holdem_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL -CNN_old_model $OLDER_VALUES_MODEL -CNN_other_old_model $OTHER_VALUES_MODEL --human_player -output=./$USER_OUTPUT"

# Self-play DNN model.
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -holdem_model $DRAW_MODEL -CNN_model $DENSE_MODEL"

echo $python_cmd

# Run the command...
$python_cmd