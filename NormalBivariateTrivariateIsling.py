import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import norm

#Normal Simulation- Metropolis-Hastings
np.random.seed(20261111)
nsim = 1000000
mcsample = np.zeros(nsim)

#Proposal moves uniformly from x - 1 to x + 1 
def proposal(x):
    return np.random.uniform(x - 1, x + 1)

#Initial value of propsal chain
pmc = 0
for i in range(1, nsim):
    #proposing new value 
    nv = proposal(pmc)

    #acceptance probability
    ast = np.exp(-(nv**2 - pmc**2) / 2)

    #accept or reject
    if np.random.uniform(0, 1) < ast:
        pmc = nv
    
    mcsample[i] = pmc

#plot histogram
plt.hist(mcsample, bins = 100, density = True)

#plot standard normal density
x = np.linspace(-4, 4, 500)
plt.plot(x, norm.pdf(x))
#Current value- propose a new value- calculate acceptance prob - accept or reject

plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.show()

#Bivariate Normal- Gibbs Sampling Example
rho = 0.6
nsim = 20000
mcsample = np.zeros((nsim, 2))
for i in range(1, nsim):
    #Sample X | Y
    mcsample[i, 0] = np.random.normal(
        loc = rho * mcsample[i-1, 1],
        scale = np.sqrt(1 - rho**2)
    )
    #Sample Y | X
    mcsample[i, 1] = np.random.normal(
        loc = rho * mcsample[i, 0],
        scale = np.sqrt(1 - rho**2)
    )

#Plot the last 500 observations
plt.scatter(
    mcsample[nsim - 500: nsim, 0],
     mcsample[nsim - 500:nsim, 1],
)

plt.xlabel("X")
plt.ylabel("Y")
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.show()

#Trivariate (X, P, N)- Gibbs Sampling
nsim = 10000
mcsample = np.zeros((nsim, 3))

#initial values
mcsample[0, :] = [0, 0.5, 1]

for i in range(1, nsim):

    # X | P, N
    # Binomial(N, P)
    X = np.random.binomial(
        n=int(mcsample[i - 1, 2]),
        p=mcsample[i - 1, 1]
    )

    mcsample[i, 0] = X

    # P | X, N
    # Beta(X+1, N-X+1)
    P = np.random.beta(
        X + 1,
        int(mcsample[i - 1, 2]) - X + 1
    )

    mcsample[i, 1] = P

    # N | X, P
    # X + Poisson(4(1-P))
    N = X + np.random.poisson(
        4 * (1 - P)
    )

    mcsample[i, 2] = N

#Plotting last 500 observations
last500 = mcsample[nsim - 500:nsim, :]
fig, axes = plt.subplots(3, 3, figsize=(10, 10))

#Scatterplot matrix
variables = ["X", "P", "N"]


for row in range(3):
    for col in range(3):

        ax = axes[row, col]

        if row == col:
            ax.hist(last500[:, col], bins=20)

        else:
            ax.scatter(
                last500[:, col],
                last500[:, row],
                s=5
            )

        if row == 2:
            ax.set_xlabel(variables[col])

        if col == 0:
            ax.set_ylabel(variables[row])

plt.tight_layout()
plt.show()

#Ising Model 
"""
Imagine a giant grid, each location has a "spin" of either +1 or -1
pick a random location, look at its neighbors, determine the conditional distribution
calculate P(spin = +1), sample spin, pick another spin, repeat
Gradually updating the code 1 piece at a time 
"""
def IsingM(betav):

    grds = 60
    nsim = 100_000

    # Create a (grds+2) x (grds+2) matrix
    # containing either -1 or +1
    sigmav = np.random.choice(
        [-1, 1],
        size=(grds + 2, grds + 2)
    )

    # Fixed boundaries = 0
    sigmav[0, :] = 0
    sigmav[-1, :] = 0
    sigmav[:, 0] = 0
    sigmav[:, -1] = 0

    for n in range(nsim):

        # Choose a random interior location
        x = np.random.randint(1, grds + 1)
        y = np.random.randint(1, grds + 1)

        # Sum of four neighbors
        nv = (
            sigmav[x, y + 1]
            + sigmav[x, y - 1]
            + sigmav[x - 1, y]
            + sigmav[x + 1, y]
        )

        # Acceptance probability
        acptf = 1 / (1 + np.exp(-2 * betav * nv))

        # Accept +1, otherwise -1
        if np.random.uniform(0, 1) < acptf:
            sigmav[x, y] = 1
        else:
            sigmav[x, y] = -1

    # Display the grid
    plt.imshow(
        sigmav,
        cmap="gray",
        interpolation="nearest"
    )

    plt.axis("off")
    plt.show()

#Running the model for different Beta values
np.random.seed(20261111)

IsingM(0)
IsingM(0.441)
IsingM(0.75)
IsingM(-1.5)