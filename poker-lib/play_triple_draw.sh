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

# 0-32 draws model:
DRAW_MODEL="$DEEP_DRAW_PATH/learning/deuce_triple_draw_conv_24_filter_xCards_xNumDraws_x0_53_percent_baseline_low_hands_good.pickle"

# Multiple models outputting {action, value} for hands. Also suggests {num_draws, value} on draw rounds.

# Original model that played vs Rep. Not good at action model at all, so not really usable going forward
#LATEST_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_prev_betting_context_500k.pickle"
#OLDER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_6_mixed_draws_model_700k.pickle"
#OTHER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_6_draws_model_300k.pickle"

# Useful choice for intermediate model. Much better than early CNN model, and also uses/outputs the full mix, including action% and num_draws.
# At the same time, gets beat handily by the later, more aggressive CNN models. 
CNN_6_PREV_BETS="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_6_testing_prev_betting_context_50k.pickle"
CNN_7_FIRST_PREV_BETS="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_prev_betting_context_500k.pickle" # July 20th. [before all other CNN_7+
CNN_7_DISCONNECT_GRAD="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_disconnect_grad_bet_percent_full_run_500k.pickle" # July 27th. First model to use full disconnected gradient (hence action% not over-fold)

# Three similar versions of model with disconnecte gradient actions. A bit passive, and pays off river too much. But plays good, and snows sometimes.
CNN_77_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_77_Randy_edits_action_percentage_700k.pickle" # Aggro. Very good draw model. Trained on games that play really well, with bounding of impossible actions. Kind of vulnerable to being bluffed though. Makes too many nitty folds.
CNN_7_ACTION_FULL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_important_river_action_percentage_700k.pickle" # Too passive. 
CNN_7_ACTION="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_important_river_bets_percent_overtrained_500k.pickle" # Folds too much (overtrained)

# CNN_8 -- trained with (some) examples vs DNN
CNN_8_VS_DNN="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_8_actions_DNN2_full_run_700k.pickle" # A bit messy
CNN_8_W_DNN_MOVES="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_8_inc_DNN2_actions_700k.pickle" # Big improvement on aggressive patting, and big pots. Calls down kind of loose, but generally a good thing.

# Up to three models that we can use.
# TODO: It's ok to use CNN_8 with DNN moves..... if we do draws model with 1/3 choice also. Otherwise... the crazy pats are too predictable
LATEST_VALUES_MODEL=$CNN_8_VS_DNN # $CNN_8_W_DNN_MOVES
OLDER_VALUES_MODEL=$CNN_77_MODEL
OTHER_VALUES_MODEL=$CNN_7_ACTION_FULL

# Two-layer DNN. Play kind-of ok, but very aggresssive, stands pat a lot, and overplays every hand. Need to handle that, to avoid getting run over.
DENSE_MODEL="$DEEP_DRAW_PATH/learning/deuce_bets_dense_2_layer_dropout_x0_5308_init_w_deuce_draws_700k.pickle"

# Older... 
#OTHER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_important_pat_river_500k.pickle" # Not too good. Too aggro, and misses subtlety. But we want it... to increase random betting.


#LATEST_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_important_pat_river_500k.pickle"
#OLDER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_disconnect_grad_bet_percent_full_run_500k.pickle"
#OTHER_VALUES_MODEL="$DEEP_DRAW_PATH/learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_over_trained_500k.pickle"

# Grep user's name. Pass it to the play_triple_draw.py for recording into CSV
USER_NAME=$USER
tmp_pass=`head -c 10 /dev/random | base64`
RANDOM_SESSION_ID="${tmp_pass:0:10}" #cut to 10 characters after base64 conversion

# First (best) model
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL --human_player -output=./$USER_NAME-vs-CNN87-$RANDOM_SESSION_ID.csv"

# Mix in second-best
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL -CNN_old_model $OLDER_VALUES_MODEL --human_player -output=./$USER_NAME-vs-CNN97-$RANDOM_SESSION_ID.csv"

# Include third, possibly bad, model
python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $LATEST_VALUES_MODEL -CNN_old_model $OLDER_VALUES_MODEL -CNN_other_old_model $OTHER_VALUES_MODEL --human_player -output=./$USER_NAME-vs-CNN87-$RANDOM_SESSION_ID.csv"

# Self-play DNN model.
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $DENSE_MODEL"

# Human play against "intermediate" CNN_6 model
#python_cmd="python $DEEP_DRAW_PATH/poker-lib/play_triple_draw.py -draw_model $DRAW_MODEL -CNN_model $CNN_7_FIRST_PREV_BETS --human_player -output=./$USER_NAME-vs-CNN87-$RANDOM_SESSION_ID.csv"

echo $python_cmd

# Run the command...
$python_cmd