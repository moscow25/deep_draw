import sys
import numpy as np
import pickle
import matplotlib.pyplot as plt

"""
Author: Nikolai Yakovenko

Visualize CNN filters for a trained model...
"""

deuce_bets_model = '../learning/deuce_events_conv_24_filter_xCards_xNumDraws_xContext_0.02_CNN_7_important_river_bets_percent_overtrained_500k.pickle'
deuce_draws_model = '../learning/deuce_triple_draw_conv_24_filter_xCards_xNumDraws_x0_53_percent_baseline_low_hands_good.pickle'
holdem_values_model = '../learning/holdem_conv_24_filter_xCards_xNumDraws_x0_9831_percent_basline_800k.pickle'
holdem_bets_model = '../learning/holdem_events_conv_24_filter_xCards_xCommunity_xContext_0.02_CNN_1_3_trained_on_CNN_1_2_700k.pickle'
video_poker_model = '../poker-lib/archive/draw_poker_conv_0.10_learn_rate_10_epoch_adaptive_16_filters_valid_border_model-81-percent.pickle'

fn = holdem_bets_model # video_poker_model # holdem_values_model # deuce_draws_model # deuce_bets_model
with open(fn,'rb') as fp:
	data = pickle.load(fp)
	print dir(data), type(data)

# Show what we got.
for i, l in enumerate(data):
	print 'layer ', i, type(l), l.shape

a = data[0]
print a.shape

ncols = 24 # 3 # how many filters to show (all random)
# For Holdem: [0] = 2 private cards, [1] = flop [2] = turn [3] = river, [4] = all public cards [5] = all cards
# 3x Draw: [0-4] = individual cards [5] = all cards
# Video poker: [0-4] = individual cards
nrow = 6 # bit (out of 31 inputs)

print a[np.random.randint(ncols),np.random.randint(nrow),:,:]
print a[np.random.randint(ncols),np.random.randint(nrow),:,:]
print a[np.random.randint(ncols),np.random.randint(nrow),:,:]

#n0, n1, d, d = a.shape

for ii in range(ncols):
	for jj in range(nrow):
		plt.subplot(ncols, nrow, ii*nrow + jj)
		#fig = plt.imshow(a[ii, jj,:,:])
		fig = plt.imshow(a[ii, jj,:,:],interpolation='nearest', cmap=plt.cm.winter) # cmap=plt.cm.copper) # cmap=plt.cm.hot)
		print([ii, jj])
		print(a[ii, jj,:,:])
		fig.axes.get_xaxis().set_visible(False)
		fig.axes.get_yaxis().set_visible(False)

plt.show()
