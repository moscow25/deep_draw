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
from scipy.interpolate import interp1d

# Try a few options or defaults for bet-sizing.

# Logic bug that spreads bet-size sampling to uniform, even if smaller size bets under-sampled 
# If True, massively over-samples from the large-bet buckets. If False, truer to what CFR does.
AGGRESSIVE_BETSIZING = False # True 

# For large bets that commit us, do we just go allin, instead of 90% allin, 80% allin, even 70% allin?
PUSH_ALLIN_COMMITTED_BETS = True
# Once odds (remainder) / (pot + bet) pretty good to call oppn allin,
# start to push probabilty mass toward the allin bet ourselves.
PUSH_ALLIN_COMMIT_MIN_ODDS = 1. / 2. # Note that we don't count opponent bet, in odds we'd get in callin the allin next.

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
## Include poker-independent math/Scipy functions here, for sampling, etc
#####################################################################

# Given a bet (or other value) and non-decreasing array of buckets, return [0.0-1.0] vector for each bucket.
# Linear interpolation. Copy weight, if buckets exactly the same (via clip, etc). Only on boundaries, etc.
def bet_to_buckets_vector(bet, buckets, debug=False):
    bet_size = bet
    bet_sizes_vector = np.array(buckets)
    bet_sizes_weights = [0.0 for item in buckets]

    # C. Apply closest value to bet size actually made... or whatever algorithm to split/project
    bet_difference_vector = np.abs(bet_sizes_vector - bet_size)
    if debug:
        print('subtract bet %.0f:\t%s' % (bet_size, bet_difference_vector))
    closest_bet_index = bet_difference_vector.argmin()
    closest_bet = bet_sizes_vector[closest_bet_index]
    if debug:
        print('closest bet size %.0f' % (closest_bet))

    # Exact match. Otherwise, spread weight linearly between closest values
    second_closest_bet_index = closest_bet_index
    if closest_bet == bet_size or (closest_bet_index == 0 and closest_bet > bet_size) or (closest_bet_index >= len(bet_sizes_vector) - 1 and closest_bet < bet_size):
        if debug:
            print('perfect match (or out of bounds)')
        # Apply to every other value that matches.
        for index in range(len(bet_sizes_vector)):
            if bet_difference_vector[index] == abs(bet_size - closest_bet):
                bet_sizes_weights[index] = 1.0
    elif bet_size < closest_bet:
        second_closest_bet_index = closest_bet_index - 1
        if debug:
            print('real bet smaller')
    else:
        second_closest_bet_index = closest_bet_index + 1
        if debug:
            print('real bet larger')

    second_closest_bet = bet_sizes_vector[second_closest_bet_index]
    if second_closest_bet_index != closest_bet_index:
        a = abs(closest_bet - bet_size)
        b = abs(second_closest_bet - bet_size)
        closest_bet_weight = max(a,b)/(a+b)
        second_closest_bet_weight = min(a,b)/(a+b)
        bet_sizes_weights[closest_bet_index] = closest_bet_weight
        bet_sizes_weights[second_closest_bet_index] = second_closest_bet_weight

        # Don't forget to spread weight to equal edge buckets... 
        # TODO: If small difference, would make sense to spread weight further...
        for index in range(len(bet_sizes_vector)):
            if bet_difference_vector[index] == (bet_size - closest_bet):
                bet_sizes_weights[index] = closest_bet_weight
            elif bet_difference_vector[index] == (bet_size - second_closest_bet):
                bet_sizes_weights[index] = second_closest_bet_weight
    
    # Final vector, with 0.0-1.0 weights for each bucket.
    return bet_sizes_weights

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
                            risk_power = 0.5*0.5, max_bet_pot_factor = 4.0, debug = True):
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



# Similarly to above, given point estimates for bet sizes, smooth (a little)
# and project to 500(?) points. Sample at projected probability. 
# Returns single bet size between [min_bet, max_bet]
# NOTE: Minimal error checking. Feed this guy good inputs.
def sample_smoothed_bet_probability_vector(bets, bet_size_probs, min_bet = 0.0, pot_size = 0.0, max_bet = 0.0, 
                                           aggressive_betting = AGGRESSIVE_BETSIZING, 
                                           push_allin_committed = PUSH_ALLIN_COMMITTED_BETS, debug = True):
    # cleanup
    bets = np.clip(bets, min_bet, max_bet)
    
    # Hack... pull down over-value of larger bets. Negatives get zero'ed out later.
    # 10x pot bet --> minus 5% [smoothing encourages huge bets too much]
    # If aggro bet sizing... tweak down a bit, on the high end.
    if aggressive_betting:
        adjustments = np.array([min(0.0, (bet / pot_size) * -0.005) for bet in bets])
        bet_size_probs += adjustments
    bet_size_probs = np.clip(bet_size_probs, -0.1, 1.0) # We don't want to eliminate negative values.
    
    x = bets
    y = bet_size_probs

    # Allins, etc.
    if min_bet == max_bet or len(x) < 2:
        return min_bet

    # Project into both X and Y space, to smooth each section.
    t = np.linspace(0, 1, len(x))
    t2 = np.linspace(0, 1, 100)
    x2 = np.interp(t2, t, x)
    y2 = np.interp(t2, t, y)

    # Now, apply a bit of smoothing.
    sigma = 5 # 10
    x3 = x2 # don't smooth in the x-direction
    y3 = gaussian_filter1d(y2, sigma)

    # Problem: On min-bet side... we might get repeated values. Take the last element, and move on.
    head_indices = len(np.where(x3==x3[0])[0])
    # print(head_indices)
    # print('%s head indices are the same!' % (head_indices))
    x3 = x3[head_indices-1:]
    y3 = y3[head_indices-1:]
    # Same for tail... maybe.
    tail_indices = len(np.where(x3==x3[-1])[0])
    # print(tail_indices)
    # print('%s tail indices are the same!' % (tail_indices))
    if tail_indices > 1:
        x3 = x3[: -1 * (tail_indices-1)]
        y3 = y3[: -1 * (tail_indices-1)]

    # If aggressive, re-sample evently in bet-size. For fair sample, keep distribution adjusting to input points.
    if aggressive_betting:
        # For histogram... sample again. This time, regular points betwen min-bet and max-bet
        interped_x = np.linspace(min_bet, max_bet, 700) # XXX points from minraise to allin (need large number, to make bet sizing non-discrete)
    else:
        # Just take the points, interpolated between unevent-spaced bet sizes. 
        interped_x = x3
    # print(interped_x)
    interped_histogram = interp1d(x3, y3)(interped_x) # Function! that iterpolates the XXX points
    interped_histogram = np.clip(interped_histogram, 0.0, 1.0)

    # Do we want to push large bets toward allin?
    # Why? Not great idea to bet 80% of the pot.
    # If we want to push large non-allin bets toward allin, here is the spot to do that.
    # In short, find a cutoff, then linearly interpolate what % of a bet's odds to push to the allin-bet instead.
    if push_allin_committed:
        # Linearly interpolate how much odds to push into allin: [cutoff, 0.0] X [0.0, 1.0]
        # print([pot, allin, PUSH_ALLIN_COMMIT_MIN_ODDS, (pot + allin),  PUSH_ALLIN_COMMIT_MIN_ODDS * (pot + allin), PUSH_ALLIN_COMMIT_MIN_ODDS * (pot + allin) / (1. + PUSH_ALLIN_COMMIT_MIN_ODDS)])
        remainder_cutoff = PUSH_ALLIN_COMMIT_MIN_ODDS * (pot_size + max_bet) / (1. + PUSH_ALLIN_COMMIT_MIN_ODDS)
        # print([remainder_cutoff, allin - remainder_cutoff])
        push_allin_percent_func = interp1d([remainder_cutoff, 0.0], [0.0, 1.0])
        for i in xrange(len(interped_histogram)):
            bet_size = interped_x[i]
            bet_odds = interped_histogram[i]
            remainder = max_bet - bet_size
            if remainder < remainder_cutoff:
                push_allin_shift = push_allin_percent_func(remainder)
                # If we're not allin, but kind of close, push some of the odds toward allin bet
                if remainder > 0.0:
                    odds_to_shift = push_allin_shift * bet_odds
                    # print('shifting %.4f%% odds from bet size %d to allin! (out of %.4f%%)' % (odds_to_shift * 100.0, int(bet_size), bet_odds*100))
                    interped_histogram[i] -= odds_to_shift
                    interped_histogram[-1] += odds_to_shift

    # interp_histogram will be the individual odds of each bet.
    if interped_histogram.sum() <= 0:
        return min_bet
    interped_histogram = interped_histogram/interped_histogram.sum()
    
    # OK, now for a single sample, just return np.choice() [or 10 sample and take one, if debug]
    if debug:
        bet_samples = []
        for _ in range(10):
            bet_size = np.random.choice(interped_x, p=interped_histogram)
            bet_samples.append(bet_size)
        bet_samples.sort()
        print([int(bet) for bet in bet_samples])
    else:
        bet_size = np.random.choice(interped_x, p=interped_histogram)
    return bet_size
