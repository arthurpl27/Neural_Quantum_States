# ===========================================================================================================
#           This code has been written with the help of an (empty) notebook that can be found
#                                    via the following citation:
#         Filippo Vicentini, VMC-from-scratch: Finding Ground-States, (2024), GitHub repository,
#      https://github.com/PhilipVinc/Lectures/blob/main/2404_ICTP/1-vmc-from-scratch-nosolution.ipynb
# ===========================================================================================================

import platform
import netket as nk
import json
import numpy as np
import matplotlib.pyplot as plt
import time
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn    # Flax: framework to define models using Jax
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

# Flax model

class Jastrow(nn.Module):
    @nn.compact
    def __call__(self, input_x):
        n_sites = input_x.shape[-1]
        J = self.param("J", nn.initializers.normal(), (n_sites,n_sites), float)
        dtype = jax.numpy.promote_types(J.dtype, input_x.dtype)
        J = J.astype(dtype)
        input_x = input_x.astype(dtype)
        J_symm = J.T + J     # make J symmetric
        res = jnp.einsum('...i, ij, ...j -> ...', input_x, J_symm, input_x)
        return res


# ============================================================
#                     METROPOLIS-HASTINGS
# ============================================================

sampler = nk.sampler.MetropolisSampler(hi,                            # Hilbert space to be sampled
                                       nk.sampler.rules.LocalRule(),  # Transition rule: flip a random site in a configuration
                                       n_chains = N_chains)

# ============================================================
#                        LOCAL ENERGIES
# ============================================================


# ------------------------------------------------ COMMENTS --------------------------------------------------------------
# • get_conn_padded : given a configuration sigma, this function returns all the configurations eta such that
# <sigma|H|eta> is not zero. This function also returns the non-zero entries of the Hamiltonian <sigma|H|eta>.

# • For a given RNG, we can provide a specific size for the configurations using a random_state function. For instance,
# with a 2D Ising Model on a (4x4) lattice such that N=16, if sigma = hi.random_state(RNG) then sigma.shape = (N,),
# eta.shape = (17,N) since we have 17 connected samples eta each composed of N spins, and H_sigmaeta.shape = (17,)
# since we have 17 matrix elements. However, if we specify a specific size for the configurations, (size_0,...,size_k),
# then if sigma = hi.random_state(RNG, (size_0,...,size_k)) then sigma.shape = ((size_0,...,size_k), N) since a
# configuration has N spins, eta.shape = ((size_0,...,size_k), 17, N) since we have 17 connected samples eta each
# composed of N spins in a tensor of shape eta.shape, and H_sigmaeta.shape = ((size_O,...,size_k), 17) for the 17 elements.
# -------------------------------------------------------------------------------------------------------------------------

def compute_local_energies(model, parameters, hamiltonian_jax, sigma):

    eta, H_sigmaeta = hamiltonian_jax.get_conn_padded(sigma)     # H_sigmaeta = get non-zero entries of the Hamiltonian
                                                                 # eta = configurations such that <sigma|H|eta> is non-zero

    logpsi_sigma = model.apply(parameters, sigma)
    logpsi_eta = model.apply(parameters, eta)
    logpsi_sigma = jnp.expand_dims(logpsi_sigma, -1)     # Add a dimension to the last axis since eta has an extra axis

    res = jnp.sum(H_sigmaeta * jnp.exp(logpsi_eta - logpsi_sigma), axis = -1)   # axis=-1 since we have to sum over the
                                                                                # last dimension of H_sigmaeta
    return res

# ============================================================
#                        ENERGY SAMPLING
# ============================================================

@partial(jax.jit, static_argnames='model')    ### jit compile ###
def estimate_energy(model, parameters, hamiltonian_jax, sigma):
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance/E_loc.size)
    return nk.stats.Stats(mean = E_average, error_of_mean = E_error, variance = E_variance)

# ============================================================
#                   ENERGY GRADIENT SAMPLING
# ============================================================

@partial(jax.jit, static_argnames='model')
def estimate_energy_and_gradient(model, parameters, hamiltonian_jax, sigma):
    sigma = sigma.reshape(-1, sigma.shape[-1])
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)
    E_average = jnp.mean(E_loc)
    E_variance = jnp.var(E_loc)
    E_error = jnp.sqrt(E_variance / E_loc.size)
    E = nk.stats.Stats(mean=E_average, error_of_mean=E_error, variance=E_variance)
    # compute the gradient using VJP
    logpsi_sigma_fun = lambda pars: model.apply(pars, sigma)
    _, vjpfun = jax.vjp(logpsi_sigma_fun, parameters)
    E_grad = vjpfun((E_loc - E_average) / E_loc.size)[0]
    return E, E_grad


# ============================================================
#                       ITERATION LOOP
# ============================================================

print("================ ITERATION LOOP ================")
lengths = [5,10,50,100]
lengths = [L * 10**3 for L in lengths]
results = {}

for Length in lengths:
    start = time.time()
    iterations = 300
    chain_length = int(Length / sampler.n_chains)
    eta = 0.0005   # learning parameter
    model = Jastrow()   # Model
    key = jax.random.key(seed)           # RNG to initialise parameters randomly
    parameters = model.init(key, np.ones((hi.size,)))

    sampler_state = sampler.init_state(model, parameters, seed=1)

    logger = nk.logging.RuntimeLog()  # stores historical data of the energy

    for i in range(iterations):
        sampler_state = sampler.reset(model, parameters, state=sampler_state)
        samples, sampler_state = sampler.sample(model, parameters, state=sampler_state, chain_length=chain_length)
        E, E_grad = estimate_energy_and_gradient(model, parameters, hamiltonian_jax, samples)
        update = jax.tree.map(lambda x : eta * x, E_grad)
        parameters = jax.tree.map(lambda x,y : x-y, parameters, update)   # Gradient descent
        logger(step=i, item={'Energy': E})

    results[Length] = logger
    end = time.time()
    print("------------------")
    print(Length, "samples : ", end-start, "seconds")
    print("Ground state energy estimate = ", logger.data['Energy']['Mean'][-1])
    print("Exact ground state = ", e_gs[0])
    print("Energy Difference |E_min-E(GS)| = ", jnp.abs(logger.data['Energy']['Mean'][-1] - e_gs[0]))
    print("Relative Energy Difference |E_min-E(GS)|/E(GS) = ",
          jnp.abs((logger.data['Energy']['Mean'][-1] - e_gs[0])/e_gs[0]))
    print("------------------")
print("------------------------------------------------------------------------------------")

# ============================================================
#                          PLOTS
# ============================================================

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 4)

ax1 = fig.add_subplot(gs[0, 0:2])   # Absolute error
ax2 = fig.add_subplot(gs[0, 2:4])   # Absolute error (semi-log)
ax3 = fig.add_subplot(gs[1, 1:3])   # Relative error

for Length, logger in results.items():

    iters = logger.data['Energy']['iters']
    energies = logger.data['Energy']['Mean']

    abs_errors = np.abs(energies - e_gs[0])
    rel_errors = abs_errors / np.abs(e_gs[0])

    ax1.plot(iters, abs_errors, label=f"{Length:,}")
    ax2.semilogy(iters, abs_errors, label=f"{Length:,}")
    ax3.semilogy(iters, rel_errors, label=f"{Length:,}")

# ------------------- Absolute error ------------------------

ax1.set_title("Energy Error")
ax1.set_xlabel("Iterations")
ax1.set_ylabel(r"$|E - E_{\rm GS}|$")
ax1.grid(alpha=0.3)
ax1.legend(title="Samples")

# ----------------- Absolute error (semi-log) ----------------

ax2.set_title("Energy Error (Semi-log)")
ax2.set_xlabel("Iterations")
ax2.set_ylabel(r"$|E - E_{\rm GS}|$")
ax2.grid(alpha=0.3)
ax2.legend(title="Samples")

# ---------------- Relative error ---------------------------

ax3.set_title("Relative Energy Error")
ax3.set_xlabel("Iterations")
ax3.set_ylabel(r"$|E - E_{\rm GS}| / E_{\rm GS}$")
ax3.grid(alpha=0.3)
ax3.legend(title="Samples")

plt.tight_layout()
plt.show()




