"""
Author: Nikolai Yakovenko
Copyright: PokerPoker, LLC 2015

Re-usable & utility functions, for poker game network.
"""

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
    hand_string = hand_string.replace('[', '')
    hand_string = hand_string.replace(']', '')
    hand_array = hand_string.split(',')
    return hand_array
