import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

#2-population replicator dynamic for 2x2x2 signaling game with only separating strategies

#pop_vec has s1, r1 reflecting proportion playing sender1 and receiver1 strategies
def rep(pop_vec, t):
    s1, r1 = pop_vec
    return [ s1 * (r1 - (s1*r1 + (1-s1)*(1-r1))) , r1 * (s1 - (r1*s1 + (1-r1)*(1-s1))) ]

#conflict of interest instead of common interest
def rep_conflict(pop_vec, t):
    s1, r1 = pop_vec
    return [ s1 * ((1-r1) - (s1*(1-r1) + (1-s1)*r1)) , r1 * (s1 - (r1*s1 + (1-r1)*(1-s1))) ]

t = np.arange(0, 100, .001)

#y0 = [.3, .4]
#y = odeint(rep_conflict, y0, t)
#plt.plot(y[:,1], y[:,0]) 

for s0 in np.arange(0.05, 0.95, .1):
    for r0 in np.arange(0.05, 0.95, .1):
        y0 = [s0, r0]
        y = odeint(rep_conflict, y0, t)
        plt.plot(y[:,1], y[:,0]) 

plt.show()
