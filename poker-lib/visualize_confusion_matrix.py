import numpy as np
import matplotlib.pyplot as plt

"""
Author: Nikolai Yakovenko

Visualize confusion matrix of clustering, etc. Based on this example:
http://stackoverflow.com/questions/5821125/how-to-plot-confusion-matrix-with-string-axis-rather-than-integer-in-python
"""

"""
conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], 
            [3,31,0,0,0,0,0,0,0,0,0], 
            [0,4,41,0,0,0,0,0,0,0,1], 
            [0,1,0,30,0,6,0,0,0,0,1], 
            [0,0,0,0,38,10,0,0,0,0,0], 
            [0,0,0,3,1,39,0,0,0,0,4], 
            [0,2,2,0,4,1,31,0,0,0,2],
            [0,1,0,0,0,0,0,36,0,2,0], 
            [0,0,0,0,0,0,1,5,37,5,1], 
            [3,0,0,0,0,0,0,0,0,39,0], 
            [0,0,0,0,0,0,0,0,0,0,38]]
"""

# 2-A x 2-A graph from 2007 NIPS paper (8 hands clusters)
# Top half is suited, bottom half is not.
conf_arr = [[4,1,1,1,1,1,2,2,2,4,4,4,6],
            [1,6,1,1,1,1,2,2,3,4,4,6,6],
            [1,1,6,1,1,2,2,2,3,4,4,6,6],
            [1,1,1,6,3,3,3,3,3,4,4,6,6],
            [1,1,1,1,7,3,3,3,3,4,5,6,6],
            [1,1,1,2,3,7,3,3,5,5,5,6,7],
            [1,1,2,2,3,3,8,3,5,5,5,6,7],
            [2,2,2,2,3,3,3,8,5,5,5,7,7],
            [2,2,2,2,3,3,3,5,8,5,7,7,7],
            [2,2,4,4,4,4,5,5,5,8,7,7,7],
            [4,4,4,4,4,4,5,5,5,5,8,7,7],
            [4,4,4,6,6,6,6,6,7,7,7,8,7],
            [6,6,6,6,6,6,6,7,7,7,7,7,8]]

# Default: norms each row. Instead, we want entire matrix.
norm_conf = []
for i in conf_arr:
    a = 0
    tmp_arr = []
    # a = sum(i, 0)
    # hack to see what happens if we keep X constant
    # Good enough. Same color for every label. 
    a = 100.0 
    for j in i:
        tmp_arr.append(float(j)/float(a))
    norm_conf.append(tmp_arr)
#print norm_conf

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')

width = len(conf_arr)
height = len(conf_arr[0])

for x in xrange(width):
    for y in xrange(height):
        ax.annotate(str(conf_arr[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')

# Do we want color bar/legend on the side?
#cb = fig.colorbar(res)
#alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
alphabet = '23456789TJQKA'

# put X-ticks on top...
ax.xaxis.tick_top()

plt.xticks(range(width), alphabet[:width])
plt.yticks(range(height), alphabet[:height])
plt.savefig('custering_matrix_preflop.png', format='png')

# Text labels...
fig.suptitle('Suited', fontsize=20)
#plt.xlabel('xlabel', fontsize=18)
plt.ylabel('Unsuited', fontsize=20)


plt.show()
