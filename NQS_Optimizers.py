import platform
import netket as nk
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from scipy.sparse.linalg import eigsh
from functools import partial

from the_model import build_model

N_chains = 20
seed = 0

model = build_model(
    h=1,
    J=1,
    L=4,
    seed=0,
    N_chains=N_chains,
    compute_ground_state=True,
    verbose=True)

g = model["g"]
hi = model["hi"]
hamiltonian = model["hamiltonian"]
hamiltonian_jax = model["hamiltonian_jax"]
hamiltonian_sparse = model["hamiltonian_sparse"]
e_gs = model["e_gs"]
psi_gs = model["psi_gs"]
# Draw the graph (uncomment line below)
# g.draw()

# Flax model
class FeedForward(nn.Module):          # 2-layer neural network with ReLU activation function and 32 neurons.
    hidden_size: int = 32
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)


# ============================================================
#                       VMC - NQS
# ============================================================

print("================ VMC - NQS ================")

start = time.time()

Length = 10000
iterations = 300
eta = 0.01


sampler = nk.sampler.MetropolisSampler(hi, nk.sampler.rules.LocalRule(), n_chains=N_chains)
initial_vstate = nk.vqs.MCState(sampler, FeedForward(), n_samples=Length, seed=seed, sampler_seed=seed)

# Create a single variational state to obtain common initial parameters
initial_vstate.init_parameters(seed=seed)

# Copy parameters for every experiment
initial_parameters = jax.tree.map(lambda x: x.copy(), initial_vstate.parameters)

print("Number of variational parameters:", initial_vstate.n_parameters)
print("------------------------------------------------------------------------------------")


# ============================================================
#                       VMC FUNCTION
# ============================================================

def run_vmc(optimizer, preconditioner=None):
    """
    Run one VMC optimization starting from the same neural-network parameters and sampler seed.

    Parameters
    ----------
    optimizer :
        NetKet optimizer, e.g. SGD, Adam, RMSProp.

    preconditioner :
        None or nk.optimizer.SR(...).

    Returns
    -------
    log.data :
        NetKet RuntimeLog data.
    """

    vstate = nk.vqs.MCState(sampler, FeedForward(), n_samples=Length, seed=seed,
                            sampler_seed=seed)                                         # Construct variational state

    vstate.parameters = jax.tree.map(lambda x: x.copy(), initial_parameters)        # Set same initial parameters

    gs = nk.driver.VMC(hamiltonian, optimizer, variational_state=vstate, preconditioner=preconditioner)
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=iterations, out=log)

    return log.data


# =========================================================================
#      Effect of the optimizer WITHOUT Stochastic Reconfiguration
# =========================================================================
#
# SGD
# Adam
# RMSProp
# SGD with Momentum
#
# ============================================================

print("================ Optimizers WITHOUT Stochastic Reconfiguration ================")

case1 = {}

print("Running SGD...")
case1["SGD"] = run_vmc(optimizer=nk.optimizer.Sgd(learning_rate=eta), preconditioner=None)

print("Running Adam...")
case1["Adam"] = run_vmc(optimizer=nk.optimizer.Adam(learning_rate=eta), preconditioner=None)

print("Running RMSProp...")
case1["RMSProp"] = run_vmc(optimizer=nk.optimizer.RmsProp(learning_rate=eta), preconditioner=None)

print("Running SGD with Momentum...")
case1["Momentum"] = run_vmc(optimizer=nk.optimizer.Momentum(learning_rate=eta), preconditioner=None)

print("------------------------------------------------------------------------------------")


# ============================================================
#              Effect of Stochastic Reconfiguration
# ============================================================
#
# SGD
# SGD + SR
#
# ============================================================

print("================ Effect of Stochastic Reconfiguration ================")

case2 = {}

print("Running SGD...")
case2["SGD"] = run_vmc(optimizer=nk.optimizer.Sgd(learning_rate=eta), preconditioner=None)

print("Running SGD + SR...")
case2["SGD + SR"] = run_vmc(optimizer=nk.optimizer.Sgd(learning_rate=eta),
                             preconditioner=nk.optimizer.SR(diag_shift=0.1))

print("------------------------------------------------------------------------------------")


# ============================================================
#           Optimizers WITH Stochastic Reconfiguration
# ============================================================
#
# SGD + SR
# Adam + SR
# RMSProp + SR
# SGD with Momentum + SR
#
# ============================================================

print("================ Optimizers WITH Stochastic Reconfiguration ================")

case3 = {}

print("Running SGD + SR...")
case3["SGD + SR"] = run_vmc(optimizer=nk.optimizer.Sgd(learning_rate=eta),
                             preconditioner=nk.optimizer.SR(diag_shift=0.1))

print("Running Adam + SR...")
case3["Adam + SR"] = run_vmc(optimizer=nk.optimizer.Adam(learning_rate=eta),
                              preconditioner=nk.optimizer.SR(diag_shift=0.1))

print("Running RMSProp + SR...")
case3["RMSProp + SR"] = run_vmc(optimizer=nk.optimizer.RmsProp(learning_rate=eta),
                                 preconditioner=nk.optimizer.SR(diag_shift=0.1))

print("Running SGD with Momentum + SR...")
case3["Momentum + SR"] = run_vmc(optimizer=nk.optimizer.Momentum(learning_rate=eta),
                                 preconditioner=nk.optimizer.SR(diag_shift=0.1))

print("------------------------------------------------------------------------------------")


# ============================================================
#                         Plots
# ============================================================

def plot_case(results, title):

    fig = plt.figure(figsize=(12, 5))
    gs = fig.add_gridspec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])


    # ------------------- Energy -------------------

    for name, data in results.items():

        ax1.plot(data["Energy"].iters, data["Energy"].Mean, label=name)

    ax1.axhline(y=e_gs[0], color="green", linestyle="--", linewidth=1, label="Ground State Energy")
    ax1.set_title("Energy")
    ax1.set_xlabel("Iterations")
    ax1.set_ylabel("Energy")
    ax1.grid(alpha=0.3)
    ax1.legend()


    # ------------------- Relative error -------------------

    for name, data in results.items():

        error = np.abs(data["Energy"].Mean - e_gs[0])
        relative_error = (error / np.abs(e_gs[0]))
        ax2.semilogy(data["Energy"].iters, relative_error, label=name)

    ax2.set_title("Relative Energy Error")
    ax2.set_xlabel("Iterations")
    ax2.set_ylabel(r"$|E-E_{\rm GS}|/E_{\rm GS}$")
    ax2.grid(alpha=0.3)
    ax2.legend()


    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.show()


# ------------------- Case 1 -------------------

plot_case(case1,"Optimizers without Stochastic Reconfiguration")

# ------------------- Case 2 -------------------

plot_case(case2,"Effect of Stochastic Reconfiguration")

# ------------------- Case 3 -------------------

plot_case(case3,"Optimizers with Stochastic Reconfiguration")

end = time.time()

print("------------------")
print("Total Simulation Time:", end - start, "seconds.")
print("------------------------------------------------------------------------------------")