#!/bin/bash

# Shell script for user to play against the latest Poker-CNN triple draw player.
# A few things...
# -> Absolute path
# -> Record user name
# -> Save results in user's directory
# Beyond that... keep it simple.

DEEP_DRAW_PATH="/home/ubuntu/deep_draw"
#DEEP_DRAW_PATH="/Users/kolya/Desktop/ML/learning/deep_draw"

# 0-32 draws model:
DRAW_MODEL="$DEEP_DRAW_PATH/learning/deuce_triple_draw_conv_24_filter_xCards_xNumDraws_x0_53_percent_baseline_low_hands_good.pickle"

# Multiple models outputting {action, value} for hands. Also suggests {num_draws, value} on draw rounds.
LATEST_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_prev_betting_context_500k.pickle"
OLDER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_6_mixed_draws_model_700k.pickle"
OTHER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_6_draws_model_300k.pickle"

# Grep user's name. Pass it to the play_triple_draw.py for recording into CSV
USER_NAME=$USER
tmp_pass=`head -c 10 /dev/random | base64`
RANDOM_SESSION_ID="${tmp_pass:0:10}" #cut to 10 characters after base64 conversion

python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL -CNN_old_model $OLDER_VALUES_MODEL -CNN_other_old_model $OTHER_VALUES_MODEL --human_player -output=./$USER_NAME-vs-CNN76-$RANDOM_SESSION_ID.csv"

echo $python_cmd

# Run the command...
$python_cmd