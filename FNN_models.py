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

# ============================================================
#                       Flax models
# ============================================================

class FNN1(nn.Module):        # 1-layer FNN with 32 neurons
    hidden_size: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        return jnp.sum(x, axis=-1)


class FNN2(nn.Module):        # 2-layer FNN with 32 neurons
    hidden_size: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)


class FNN3(nn.Module):        # 2-layer FNN with 64 neurons
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)


class FNN4(nn.Module):        # 3-layer FNN with 64 neurons
    hidden_size: int = 64

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)


class FNN5(nn.Module):        # 3-layer FNN with 128 neurons
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        return jnp.sum(x, axis=-1)


# ============================================================
#                       VMC - NQS
# ============================================================

print("================ VMC - NQS ================")

Length = 10000
iterations = 300
eta = 0.01

sampler = nk.sampler.MetropolisSampler(hi, nk.sampler.rules.LocalRule(), n_chains=N_chains)

initial_vstates = {
    "FNN1": nk.vqs.MCState(sampler, FNN1(), n_samples=Length, seed=seed, sampler_seed=seed),
    "FNN2": nk.vqs.MCState(sampler, FNN2(), n_samples=Length, seed=seed, sampler_seed=seed),
    "FNN3": nk.vqs.MCState(sampler, FNN3(), n_samples=Length, seed=seed, sampler_seed=seed),
    "FNN4": nk.vqs.MCState(sampler, FNN4(), n_samples=Length, seed=seed, sampler_seed=seed),
    "FNN5": nk.vqs.MCState(sampler, FNN5(), n_samples=Length, seed=seed, sampler_seed=seed) }


# Store initial parameters for each architecture
initial_parameters = {name: jax.tree.map(lambda x: x.copy(), vstate.parameters)
                                                for name, vstate in initial_vstates.items()}

# Number of parameters for each model
for name, vstate in initial_vstates.items():
    print(f"{name}: {vstate.n_parameters} variational parameters")

print("------------------------------------------------------------------------------------")


def run_vmc(model_name, optimizer):
    """
    Run one VMC optimization for a given neural-network
    architecture using Stochastic Reconfiguration.
    """

    models = {"FNN1": FNN1(), "FNN2": FNN2(), "FNN3": FNN3(), "FNN4": FNN4(), "FNN5": FNN5()}
    vstate = nk.vqs.MCState(sampler, models[model_name], n_samples=Length, seed=seed, sampler_seed=seed)
    vstate.parameters = jax.tree.map(lambda x: x.copy(), initial_parameters[model_name])
    gs = nk.driver.VMC_SR(hamiltonian, optimizer, diag_shift=0.01, variational_state=vstate)
    log = nk.logging.RuntimeLog()
    gs.run(n_iter=iterations, out=log)

    return log.data

# ============================================================
#                         SGD + SR
# ============================================================

def SGD_SR(model_name, learning_rate):

    run = run_vmc(model_name=model_name, optimizer=nk.optimizer.Sgd(learning_rate=learning_rate))
    return run

# ============================================================
#              Run neural-network models
# ============================================================

results = {}
model_names = ["FNN1", "FNN2", "FNN3", "FNN4", "FNN5"]

for model_name in model_names:

    start = time.time()
    print(f"Running SGD + SR for {model_name}")

    results[model_name] = SGD_SR(model_name, eta)
    end = time.time()

    print("------------------")
    print(f"Total Simulation Time for {model_name}:", end - start, "seconds.")
    print("------------------------------------------------------------------------------------")

# ============================================================
#                         Plots
# ============================================================

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

fig.suptitle("Effect of the FNN Architecture on SGD + SR", fontsize=16)

# ------------------------ Energy ------------------------------

for model_name in model_names:

    data = results[model_name]
    iterations_plot = data["Energy"]["iters"]
    energy = data["Energy"]["Mean"]
    ax1.plot(iterations_plot, energy, label=model_name)

ax1.axhline(e_gs, linestyle="--", label=r"$E_{\mathrm{gs}}$")
ax1.set_xlabel("Iteration")
ax1.set_ylabel("Energy")
ax1.set_title("Energy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# ----------------- Relative Energy Error ----------------------

for model_name in model_names:

    data = results[model_name]
    iterations_plot = data["Energy"]["iters"]
    energy = data["Energy"]["Mean"]
    relative_error = np.abs((energy - e_gs) / e_gs)
    ax2.semilogy(iterations_plot, relative_error, label=model_name)

ax2.set_xlabel("Iteration")
ax2.set_ylabel(r"Relative energy error $|E-E_{\mathrm{gs}}|/E_{\mathrm{gs}}$")
ax2.set_title("Relative Energy Error")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
