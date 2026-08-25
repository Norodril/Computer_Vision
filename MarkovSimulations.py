import numpy as np
import matplotlib.pyplot as plt
import random
#creating the transition matrix
P = np.array ([[0, 2/5, 3/5],
              [1/2, 1/4, 1/4],
              [1/2, 1/6, 1/3]
              ])

print(P)

#given that im currently in state x, generate a random number and use the transition probabilities to decide where to go next
import random
def trans(x):
    u = random.random()
    if x == 1:
        if u < 0.4:
            return 2
        else:
            return 3
    if x == 2:
        if u < 0.5:
            return 1
        elif u < 0.75:
            return 2
        else:
            return 3
    if x == 3:
        if u < 0.5:
            return 1
        elif u < 0.5 + 1/6:
            return 2
        else:
            return 3
    
#Simulating 10,000 transitions
random.seed(11)
nsim = 10000
mc = [0] * nsim
initial = 1
mc[0] = initial
for i in range(nsim - 1):
    mc[i + 1] = trans(mc[i])
#mc contains the simulated Markov chain

#Comparing the simulation to the stationary distribution
counts = np.bincount(mc)[1:]
proportions = counts / nsim
print(proportions)

"""
output of 0.3357, 0.2683, 0.396 matches the theoretical distribution
of approximately pi = (1/3, 4/15, 2/5)
"""

def trans_hhh(x):
    u = random.random()
    if x == 0:
        if u < 0.5:
            return 0
        else: 
            return 1
    if x == 1:
        if u < 0.5:
            return 0
        else:
            return 2
    if x == 2:
        if u < 0.5:
            return 0
        else:
            return 3
    if x == 3:
        return 3

#simulating how long HHH takes
random.seed(11)
nsim = 10000
nfp = [0] * nsim #eventually will store the number of flips required in one simulation
for i in range(nsim):
    state = 0 
    count = 0
    while state != 3:
        state = trans_hhh(state)
        count += 1
    nfp[i] = count

#calculating mean and variance
print("Mean:", np.mean(nfp))
print("Variance:", np.var(nfp, ddof = 1))
"""
Mean: 14.0341
Variance: 147.325... 
Takes about 14 flips on average

"""

#Part 3: Player A vs Player B
def trans_game(x):
    u = random.random()
    #State 0: Starting state
    if x == 0:
        if u < 5/36:
            return 2
        else:
            return 1
    
    #State 1: A has not won yet
    if x == 1:
        if u < 1/6:
            return 3
        else:
            return 0
    #State 2: A has won
    if x == 2:
        return 2
    #State 3: B has won
    if x == 3:
        return 3
    
#Simulating 10000 games
random.seed(11)
nsim = 10000
ng = [0] * nsim
for i in range(nsim):
    state = 0
    count = 0

    #continu while game has not ended
    while state == 0 or state == 1:
        state = trans_game(state)
        count += 1
    ng[i] = count

#Calculating mean and var
print("Game mean:", np.mean(ng))
print("Game variance:", np.var(ng, ddof=1))

"""
Simulated mean and variance match approximately the theoretical mean and variance provided
6.5391 close to 6.59
34.63 close to 36.24

Now moving to Branching Process- Zn = number of individuals in generation n
- Each individual produces a random number of children according to a Poisson distribution
"""
#If there are z individuals, generate the number of children for each individual and add them together 
def offspring(z, lambda_var = 1):
    return np.sum(np.random.poisson(lambda_var, z))

#simulating one branching process
np.random.seed(11)
n = 10
lmd = 2 
zn = [1] * n 

for i in range(n - 1):
    zn[i+1] = offspring(zn[i], lmd)

print("Branching Process:")
print(zn)

#Plotting the branching process
plt.xlabel("n")
plt.ylabel("Z_n")
plt.title("Branching Process")
plt.show

#Simulating 100 different paths, each lastig 10 generations
def branch_paths(n, nsim, lambda_value):
    for j in range(nsim):
        zn = [1] * n 
        for i in range(n - 1):
            zn[i + 1] = offspring(zn[i], lambda_value)
        plt.plot(range(n), zn)
    plt.xlabel("n")
    plt.ylabel("Z_n")
    plt.title(f"Branching Process (lambda = {lambda_value})")
    plt.show()

np.random.seed(11)
branch_paths(10, 100, 2)
branch_paths(10, 100, 1.5)
branch_paths(10, 100, 1)
