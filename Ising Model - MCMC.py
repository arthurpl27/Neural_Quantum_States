import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import bernoulli

# Parameters
m = int(50)
n = int(5 * 10**6)
betas = [0.3, 0.5]
J = 1
B = 0.1

# Initial S0 with random spins {-1;1}
S0 = np.zeros((m,m))
for i in range(m):
    for j in range(m):
        S0[i,j] = 2 * bernoulli.rvs(0.5) - 1

# Ising Model
def ising(n, m, beta, J, B, S0):
    E = np.zeros(n+1)
    M = np.zeros(n+1)
    # Initialize system
    S = S0
    # Magnetic moment for the initial condition
    M[0] = S.sum()
    # Energy for the initial condition
    E[0] = - J * ( (S[:,:m-1] * S[:,1:]).sum() + (S[:m-1,:]*S[1:,:]).sum() ) - B * M[0]
    for k in range(n):
        # PROPOSAL: generate candidate state
        Sc = S.copy()
        # select randomly an atom on the lattice
        i = int( np.floor(m * np.random.random()) )
        j = int( np.floor(m * np.random.random()) )
        Sc[i,j] = - S[i,j] # flip the spin
        # Change in magnetic moment due to this flip
        dM = - 2 * S[i,j]
        # Change in the energy due to this flip
        dE = 0
        if i>0:
            dE = dE + S[i-1,j]
        if i < m-1:
            dE = dE + S[i+1,j]
        if j>0:
            dE = dE + S[i,j-1]
        if j<m-1:
            dE = dE + S[i,j+1]
        dE = 2*S[i,j]*(J*dE + B)
        
        # ACCEPT-REJECT STEP
        alpha = np.min([np.exp(-dE*beta),1])
        U = np.random.random()
        if U < alpha:
            S = Sc
        else:
            dM = 0
            dE = 0
        # Update energy and magnetic moment
        E[k+1] = E[k] + dE
        M[k+1] = M[k] + dM
    return E, M, S


for beta in betas:
    E, M, S = ising(n, m, beta, J, B, S0)

    # Plots
    fig = plt.figure(figsize=(12, 10))
    fig.suptitle(rf"Ising Model ($\beta={beta}$)", fontsize=16)

    gs = fig.add_gridspec(2, 2, hspace=0.3)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(E)
    ax1.set_xlabel("Monte Carlo samples")
    ax1.set_ylabel("Energy")

    ax2 = fig.add_subplot(gs[1, 0])
    im = ax2.imshow(S0, cmap='bwr', vmin=-1, vmax=1)
    ax2.set_title("Initial State")

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.imshow(S, cmap='bwr', vmin=-1, vmax=1)
    ax3.set_title("Final State")

    plt.tight_layout()
    plt.show()
