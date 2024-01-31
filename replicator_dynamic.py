import numpy as np
from scipy.integrate import odeint
from matplotlib import pyplot as plt

#1-population replicator dynamic


#comment-out all but the one game's utility function that you are interested in
#call plot1d() to visualize the dynamics for the 2-player games
#call plot2d() to visualize the dynamics for the 3-player games


#hawkdove game
#you should see rapid convergence to 1/2 (mixed pop of hawks and doves)
u = [ [0, 3],
        [1, 2] ]

#prisoner's dilemma
#you should see rapid convergence to 0 (all defect)
u = [ [3, 1],
        [4, 2] ]

#stag hunt
#you should see a 'knife edge': above a line, convergence to 1; below, convergence to 0
'''
u = [ [3, 3] ,
        [0, 4] ]
        '''

#rock-paper-scissors
#you should see 'cycles'
u = [ [1, 2, 0],
        [0, 1, 2],
        [2, 0, 1] ]

#rock-paper-scissors with small mod
#larger e makes 'inward spiral' easier to see
#with this modification, the point [1/3 , 1/3 , 1/3] is globally stable
#so, you should see all points 'spiraling in' towards 1/3, 1/3
'''
e = 0.1
u = [ [1-e, 2, 0],
        [0, 1-e, 2],
        [2, 0, 1-e] ]
        '''


#pop_vec has p1, ... , p_n-1, representing proportions of each strategy
#pn = 1 - sum_i < n pi
#pi = pi * ( fit(i) - avgfit )
def rep(pop_vec, t):
    allpops = np.concatenate((pop_vec , [ 1-sum(pop_vec) ]))
    return [ pop_vec[i] * ( fitness(i, allpops) - avg_fitness(allpops) )
            for i in range(len(pop_vec)) ]

#fit(i) = sum_j ( pj * u(i, j) )
def fitness(i, allpop):
    return sum([allpop[j] * u[i][j] for j in range(len(allpop))])

#avgfit = sum_j pj * fit(j)
def avg_fitness(allpop):
    return sum([allpop[j] * fitness(j, allpop) for j in range(len(allpop))])

#replicator-mutator dynamics
Q = [ [ 0.95, 0.05] ,
        [0.05, 0.95] ]

Q = [ [ 0.95, 0.025, 0.025 ],
        [0.025, 0.95, 0.025 ],
        [0.025, 0.025, 0.95 ] ]

#pi = ( sum_j fit(j) * pj * Q_ji ) - pi * avgfit
def rep_mut(pop_vec, t):
    allpops = np.concatenate((pop_vec , [ 1-sum(pop_vec) ]))
    return [ sum([ fitness(j, allpops) * allpops[j] * Q[j][i] for j in range(len(allpops))]) -
            pop_vec[i] * avg_fitness(allpops)
            for i in range(len(pop_vec)) ]

t = np.arange(0, 100, .1)

#use these functions to visualize the dynamics
#pass `rep' to odeint for replicator dynamics
#pass `rep_mut' to odeint for replicator-mutator dynamics
#note that Q and u must be the same shape

def plot1d():

    for y0 in np.arange(0.05, 0.95, .1):
        y = odeint(rep_mut, [ y0 ], t)
        plt.plot(y) 

    plt.show()


def plot2d():

    for s0 in np.arange(0.05, 0.95, .1):
        #1-s0 to ensure that population vector lies in simplex
        for r0 in np.arange(0.05, 1-s0, .1):
            y0 = [s0, r0]
            y = odeint(rep_mut, y0, t)
            plt.plot(y[:,1], y[:,0]) 

    plt.show()

# a general printing function, which might be useful for games of higher dimension
def general_print():
    y0 = [0.75, 0.1]
    print('y0 = ' + str(y0))
    y = odeint(rep, y0, t)
    print(y)

plot2d()
