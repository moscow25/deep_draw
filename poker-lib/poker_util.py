"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Re-usable & utility functions, for poker game network.
"""

# Math functions
import numpy as np
from scipy.stats import beta 
# from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d

# Fill in dense features vector from sparse features map. Keys: 'key':row, Data: 'key':datum.
def VectorFromKeysAndSparseMap(keys, sparse_data_map, default_value = 0):
    dense_data_vector = [default_value] * len(keys)
    for data_key in sparse_data_map:
        dense_data_vector[keys[data_key]] = sparse_data_map[data_key]
    return dense_data_vector

# Given a map of keys, create corresponding vector (for CSV writer, etc). Keys: 'key':row
def KeysVectorFromKeysMap(keys):
    keys_vector = [0] * len(keys)
    for key in keys:
        keys_vector[keys[key]] = key
    return keys_vector

# return <row name: row #> map based on csv key
def CreateMapFromCSVKey(csv_play):
    csv_map = {}
    for i in range(len(csv_play)):
        csv_map[csv_play[i]] = i
    return csv_map

# And similarly, add debug & lookups as needed

# Used a lot. [card, crd] -> [Ks,2d]
def hand_string(cards_array):
    if not cards_array:
        return '[]'
    return '[%s]' % ','.join([str(card) for card in cards_array])

# And the other way. [Ks,2d] -> ['Ks', '2d']
# Still strings, but splits reasonably
def hand_string_to_array(hand_string):
    if not hand_string:
        return []
    hand_string = hand_string.replace('[', '')
    hand_string = hand_string.replace(']', '')
    hand_array = hand_string.split(',')
    #print hand_array
    # Return empty array for '[]' input.
    if len(hand_array) == 1 and not hand_array[0]:
        return []
    return hand_array


#####################################################################
## Include poker-independent math functions here, for sampling, etc
#####################################################################

# If we know the mean and stdev of our beta-distribution, can just compute & output
# return (alpha, beta, scale, loc) so matches beta-fit [can just scale=1.0, loc = 0.0]
# http://stats.stackexchange.com/questions/12232/calculating-the-parameters-of-a-beta-distribution-using-the-mean-and-variance
# alpha <- ((1 - mu) / var - 1 / mu) * mu ^ 2
# beta <- alpha * (1 / mu - 1)
# 
# Also, note the bounds (from same link)
# mu -> (0,1)
# var -> (0, 0.5**2]
def generate_beta(mean, stdev, scale = 1.0):
    mu = mean / scale
    var = (stdev / scale) ** 2
    var = min(var, 0.5**2)
    var = max(var, 0.00000001)
    print('mu %s, var %s' % (mu, var))
    alpha =  ((1.0 - mu) / var - 1.0 / mu) * (mu**2)
    beta = alpha * (1.0 / mu - 1.0)
    return (alpha, beta, scale, 0.0)
    

# ~12 point sample, related to pot_size, including min and max bets, and some points in between.
# As default, use bet amounts used by Slumbot 2014. https://www.aaai.org/ocs/index.php/WS/AAAIW13/paper/viewFile/7044/6479
# pot * [0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 15.0, 25.0, 50.0] (and allin)
# NOTE: We snip any bets outside of [min_bet, max_bet] range, so length might be variable
def sample_bets_range(pot_size, min_bet, max_bet):
    bets = np.array([0.10, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0, 8.0, 15.0, 25.0, 50.0]) * pot_size
    bets = np.clip(bets, min_bet, max_bet) # snip out-of-range bets
    bets = np.unique(np.append(bets, [min_bet, max_bet]))
    return bets
                   
# Given 1-x data points, use Gaussian smoothing (in both X and Y) to create a better curve.
# In our case, for (bet_size, bet_value) pairs.
# NOTE: Order matters. Since bet_size == $200 in position[0] is not quite the same output as bet_size == $200 in position[3]
# TODO: Create a stochastic version, but permuting outputs randomly, and re-fitting.
# Code example from: http://stackoverflow.com/questions/15178146/line-smoothing-algorithm-in-python
def best_bet_with_smoothing(bets, values, min_bet = 0.0, pot_size = 0.0, allin_win = 0.0,
                            risk_power = 0.5, max_bet_pot_factor = 4.0, debug = True):
    # bets_data
    x = bets
    y = values

    # Estimate min_bet and pot_size if not provided (should be given)
    if not min_bet:
        min_bet = bets[0]
    if not pot_size:
        max_bet = bets[-1]
        chips_bet = BIG_BET_FULL_STACK - max_bet
        pot_size = max(2 * chips_bet, 2 * min_bet)

    # If we're using a risk factor, divide (positive) values by a factor of the bet_size.    
    if risk_power and allin_win:
        # If we also know allin win%, then scale the risk factor... on marginal value over (or under) the expected
        # WHY? Suppose we have full house. All values will be high. We expect to win those chips. Risk is on the margin.
        # allin_value = allin_win * pot_size 
        # allin_win can be optimistic. 0.75 win is more like 0.5 pot, actually. Obviously 1.0 --> 1.0
        allin_win_regressed = max(allin_win * 2. - 1., 0.0)
        allin_value = max(allin_win_regressed * pot_size, 0.0)
        value_at_risk_factor = [((pot_size + min_bet)/(value + pot_size))**risk_power for value in bets]

        # Now, calculate the point values, as margins on the allin_value.
        marginal_bet_values = values - allin_value
        marginal_value_at_risk_factor = np.array([(factor if value >= 0 else 1.0/factor) for (factor, value) in zip(value_at_risk_factor, marginal_bet_values)])
        if debug:
            print('%.2f allin_value. Marginal values:\n%s\n------------' % (allin_value, marginal_bet_values))
            print(marginal_value_at_risk_factor)
        values = np.multiply(marginal_bet_values, marginal_value_at_risk_factor) + allin_value
        y = values
    elif risk_power:
        value_at_risk_factor = [((pot_size + min_bet)/(value + pot_size))**risk_power for value in bets]
        value_at_risk_factor = np.array([(factor if value >= 0 else 1.0/factor) for (factor, value) in zip(value_at_risk_factor, values)])
        if debug:
            print(value_at_risk_factor)
        values = np.multiply(values, value_at_risk_factor)
        y = values
    else:
        if debug:
            print('No risk factor given. Interpolating (bets, values) as-is')

    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100) # 100 points, for which we interpolate

    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)
    sigma = 10
    x3 = gaussian_filter1d(x2, sigma)
    y3 = gaussian_filter1d(y2, sigma)

    x4 = np.interp(t, t2, x3)
    y4 = np.interp(t, t2, y3)

    # X3 and Y3 are the final point estimates. What is the largest point?
    # print(zip(x3,y3))

    # If provided, use a cutoff for maximum bet we can ever make. Just to sanity-check 50x pot bets...
    if max_bet_pot_factor and max_bet_pot_factor > 0.0:
        max_bet = pot_size * max_bet_pot_factor
        y_min = np.amin(y3)
        max_arg = np.argmax([(y_t if x_t<=max_bet else y_min - 0.01) for (x_t,y_t) in zip(x3,y3)])
    else:
        max_arg = np.argmax(y3)
    if debug:
        print([x3[max_arg], y3[max_arg]])

    # X4 and Y4 are the regressed points...  [one each for prior bet size)
    if debug:
        print(zip(x4, y4))

    return (x3[max_arg], y3[max_arg])
