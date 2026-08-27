"""
We have a population Z0, Z1, Z2, ... 
Poisson offspring distribution
"""
import numpy as np
import matplotlib.pyplot as plt 

#Given that there are z individuals, randomly generate the number of children for 
#each of those z individuals and add them together
def ud(z, lam = 1): 
    offspring = np.random.poisson(lam, size = z) # #of children is the event that is occuring randomly 
    return np.sum(offspring)

print(ud(4, 2)) #4 individuals , 2 children on average

#simulating one branching process
np.random.seed(12)
n = 10
lam = 2
zn = np.ones(n, dtype = int) #array of ones to store the population
for i in range(n - 1): #Z0-Z9
    zn[i + 1] = ud(zn[i], lam) #calculates for Z1-Z9, we know Z0 = 1 individual
print(zn)

#plotting one branching path
plt.plot(range(n), zn)
plt.xlabel("n")
plt.title("Simulating 1 Path")
plt.ylabel("Z_n")
plt.show()

#Simulating many branching processes
def branchpath(n, nsim, lam):
    for j in range(nsim):
        #starting with one individual
        zn = np.ones(n, dtype = int)

        #simulating the generations
        for i in range(n-1):
            zn[i + 1] = ud(zn[i], lam)
        
        #plotting simulation
        plt.plot(range(n), zn)

    plt.xlabel("n")
    plt.ylabel("Z_n")
    plt.title("Branching process simulations")
    plt.xlim(0, n - 1)
    plt.ylim(0, 50)
    plt.show()


#10 generations, 100 simulations, average child count of 2
branchpath(10, 100, 2)
#child count of 1.5
branchpath(10, 100, 1.5)
#child count of 1
branchpath(10, 100, 1)

#Estimating probability of extinction
def propextinct(n, nsim, lam):
    extinct = 0
    for j in range(nsim):
        #starting with one individual
        zn = np.ones(n, dtype = int)
        #simulating process
        for i in range(n - 1):
            zn[i + 1] = ud(zn[i], lam)
        if zn[n - 1] == 0: #the last element
            extinct += 1
    #estimated probability of extinction
    return extinct / nsim


prob = propextinct(10, 1000, 2)
print(prob) #this does not guarantee accuracy, must increase n (as n gets last, probability converges to true probability)
prob2 = propextinct(50, 10000, 2)
print(prob2)

#Custom offspring distribution
"""
For every individual
0 children - probability 1/6
1 child - probability 1/2
2 children - probability 1/3 

rdiscrete() function- Generate a random number between 0 and 1, and use it to choose one of the possible
offspring values according to their probabilities 
"""

def rdiscrete(v, a):
    return np.random.choice(v, p = a)
#ex
v = [0, 1, 2]
a = [1/6, 1/2, 1/3]
print(rdiscrete(v, a))

"""
ud function- if there are z individuals, randomly generate the number of children
for each of the individuals and add them together
"""

def ud_custom(z, v, a):
    total = 0
    for i in range(z):
        total += rdiscrete(v, a)
    return total 

#custom branching simulation 
def branchpath_custom(n, nsim, v, a):
    for j in range(nsim):
        zn = np.ones(n, dtype = int)
        for i in range(n - 1):
            zn[i + 1] = ud_custom(zn[i], v, a)
        plt.plot(range(n), zn)
    plt.xlabel("n")
    plt.ylabel("Z_n")
    plt.title("Branching process simulations")
    plt.xlim(0, n-1)
    plt.ylim(0, 50)
    plt.show()

v = [0, 1, 2]
a = [1/6, 1/2, 1/3]

branchpath_custom(20, 100, v, a)

#Extinction for the custom distribution
def propextinct_custom(n, nsim, v, a):
    extinct = 0
    for j in range(nsim):
        zn = np.ones(n, dtype = int)
        for i in range(n - 1):
            zn[i + 1] = ud_custom(zn[i], v, a)
        if zn[n - 1] == 0:
            extinct +=1
    return extinct/nsim 

prob = propextinct_custom(50, 3000, v, a)
print(prob)

#Geometric offspring distribution
#X ~ Geometric(p), where X represents the number of failures before the first success
#random.geometric(p) gives the # of trials until the first success, so to match
#R's rgeom(), we use np.random.geometric(p) - 1

def ud_geometric(z, p = 0.45):
    offspring = np.random.geometric(p, size = z) - 1
    return np.sum(offspring)

#Geometric Branching Process
def branchpath_geometric(n, nsim, p):
    for j in range(nsim):
        zn = np.ones(n, dtype = int)
        for i in range(n - 1):
            zn[i + 1] = ud_geometric(zn[i], p)
        plt.plot(range(n), zn)
    plt.xlabel("n")
    plt.ylabel("Z_n")
    plt.title("Branching process simulation")
    plt.xlim(0, n - 1)
    plt.ylim(0, 50)
    plt.show()

p = 0.45

branchpath_geometric(20, 100, p)

#Extinction Probability for Geometric Distribution
def propextinct_geometric(n, nsim, p):
    extinct = 0
    for j in range(nsim):
        zn = np.ones(n, dtype = int)
        for i in range(n - 1):
            zn[i + 1] = ud_geometric(zn[i], p)
        if zn[n - 1] == 0:
            extinct += 1
    return extinct / nsim


prob = propextinct_geometric(50, 3000, 0.45)
print(prob)
            
