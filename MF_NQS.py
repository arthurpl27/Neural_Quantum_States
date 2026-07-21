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

# Parameters
h = 1
J = 1
L = 4
seed = 0
N_chains = 20
g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

print("------------------------------------------------------------------------------------")
print("================ PARAMETERS ================")
print("Number of sites:", g.n_nodes)
print("List of sites:", [node for node in g.nodes()])
print("Number of edges:", g.n_edges)
print("List of edges:", g.edges())    # start and end node
print("------------------------------------------------------------------------------------")

#g.draw()
# ============================================================
#               CONSTRUCTION OF THE HAMILTONIAN
# ============================================================

# Define Hilbert space based on this graph + fixed total magnetization of zero
hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

# Create empty operator
hamiltonian = nk.operator.LocalOperator(hi)

sigma_x = []
sigma_zz = []
for site in g.nodes():
    sigma_x.append(nk.operator.spin.sigmax(hi,site))

for (i,j) in g.edges():
    sigma_zz.append(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j))

hamiltonian = -h * np.sum(sigma_x) + J * np.sum(sigma_zz)
hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()
print("================ HAMILTONIAN ================")
print(hamiltonian)
print("------------------------------------------------------------------------------------")
print("================ JAX HAMILTONIAN ================")
print(hamiltonian_jax)
print("------------------------------------------------------------------------------------")

# ============================================================
#       COMPUTE EXACT ENERGY/STATE OF THE GROUND STATE
# ============================================================

print("================ GROUND STATE ENERGY & EIGENVECTOR ================")

e_gs, psi_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors = True)

print("Ground state energy:", e_gs)
print("Ground state:", psi_gs)
print("------------------------------------------------------------------------------------")

# Flax model
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (hi.size,), float)
        # compute probabilities
        p = nn.log_sigmoid(lam*x)
        return 0.5 * jnp.sum(p, axis=-1)

### To input N random parameters using RNG:   inputs = hi.random_state(jax.random.key(1), (N,)) ###

"""
# =====================================================================================
#       INPUT MODEL & PARAMETERS - RETURNS: NORMALIZED EXPONENTIATED WAVEFUNCTION
# =====================================================================================
print("================ EXPONENTIATED WAVEFUNCTION ================")

model = MF()   # Model
key = jax.random.key(seed)           # RNG to initialise parameters randomly
parameters = model.init(key, np.ones((hi.size,)))
print("Initial Parameters:", parameters)

def to_array(model, parameters):
    all_configurations = hi.all_states()         # generate all configurations of the Hilbert space
    log_psi = model.apply(parameters, all_configurations)
    psis = jnp.exp(log_psi)
    Z = jnp.linalg.norm(psis)
    return psis / Z

start1 = time.time()
amplitudes = to_array(model, parameters)
end1 = time.time()
print("Amplitudes:", amplitudes)
print("Compiling time:", end1-start1, "seconds")

start2 = time.time()
to_array_jit = jax.jit(to_array, static_argnames="model")     ### jit for faster computations ###
amplitudes_jit = to_array_jit(model, parameters)
end2 = time.time()
print("Amplitudes jit:", amplitudes_jit)
print("Compilation time jit:", end2-start2, "seconds")
print("------------------------------------------------------------------------------------")"""

# ============================================================
#                     METROPOLIS-HASTINGS
# ============================================================

sampler = nk.sampler.MetropolisSampler(hi,                            # Hilbert space to be sampled
                                       nk.sampler.rules.LocalRule(),  # Transition rule: flip a random site in a configuration
                                       n_chains = N_chains)

# ============================================================
#                        LOCAL ENERGY
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

def compute_local_energies_jit(model, parameters, hamiltonian_jax, sigma):          ### jit for faster computations ###
    local_energies_jit = jax.jit(compute_local_energies, static_argnames='model')
    local_energies = local_energies_jit(model, parameters, hamiltonian_jax, sigma)

    return local_energies

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

"""isinstance(estimate_energy(model, parameters, hamiltonian_jax, samples), nk.stats.Stats)
print(estimate_energy(model, parameters, hamiltonian_jax, samples))""" #check code runs well
# ============================================================
#                        COMPUTE ENERGY
# ============================================================
"""
print("================ COMPUTATION OF ENERGY ================")

hamiltonian_sparse = hamiltonian.to_sparse()
hamiltonian_jax_sparse = hamiltonian_jax.to_sparse(jax_ = True)

def compute_energy(model, parameters, hamiltonian_sparse):
    psi = to_array(model, parameters)
    Z = psi.conj().T @ psi
    return psi.conj().T @ (hamiltonian_sparse @ psi) / Z    # compute <psi|H|psi> / <psi|psi>

start3 = time.time()
energy = compute_energy(model, parameters, hamiltonian_sparse)
end3 = time.time()
print("Energy:", energy)
print("Energy compilation time:", end3-start3, "seconds")


compute_energy_jit = jax.jit(compute_energy, static_argnames="model")   ### jit again for faster computation ###
start4 = time.time()
energy_jit = compute_energy_jit(model, parameters, hamiltonian_jax_sparse)
end4 = time.time()
print("Energy jit:", energy_jit)
print("Energy compilation time jit:", end4-start4, "seconds")
print("------------------------------------------------------------------------------------")"""
"""
# ============================================================
#                       COMPUTE GRADIENT
# ============================================================

print("================ COMPUTATION OF GRADIENT ================")

def compute_energy_and_gradient(model, parameters, hamiltonian_jax_sparse):
    grad_fun = jax.value_and_grad(compute_energy, argnums = 1)
    return grad_fun(model, parameters, hamiltonian_jax_sparse)

gradient_jax = jax.jit(compute_energy_and_gradient, static_argnames="model")     ### Always jit to compute gradient ###
jax_energy, jax_gradient = gradient_jax(model, parameters, hamiltonian_jax_sparse)  # Energy is the same as before
print("Gradient:", jax_gradient)
print("------------------------------------------------------------------------------------")
"""
# ============================================================
#                   ENERGY GRADIENT SAMPLING
# ============================================================

@partial(jax.jit, static_argnames='model')
def estimate_energy_and_gradient(model, parameters, hamiltonian_jax, sigma):
    sigma = sigma.reshape(-1, sigma.shape[-1])
    E_loc = compute_local_energies(model, parameters, hamiltonian_jax, sigma)
    E = estimate_energy(model, parameters, hamiltonian_jax, sigma)
    logpsi_sigma_function = lambda params : model.apply(params, sigma)
    Jacobian = jax.jacrev(logpsi_sigma_function)(parameters)
    weights = (E_loc - E.mean) / E_loc.size
    E_grad = jax.tree.map(lambda J : jnp.tensordot(weights, J, axes = ([0], [0])), Jacobian)
    # compute the gradient using VJP
    #logpsi_sigma_fun = lambda pars: model.apply(pars, sigma)
    #_, vjpfun = jax.vjp(logpsi_sigma_fun, parameters)
   # E_grad = vjpfun((E_loc - E.mean) / E_loc.size)[0]
    return E, E_grad


# ============================================================
#                       ITERATION LOOP
# ============================================================

print("================ ITERATION LOOP ================")

iterations = 300
chain_length = int(1000 / sampler.n_chains)
eta = 0.01   # learning parameter
# model = MF()
# parameters = model.init(...)
model = MF()   # Model
key = jax.random.key(seed)           # RNG to initialise parameters randomly
parameters = model.init(key, np.ones((hi.size,)))
print("Initial Parameters:", parameters)

sampler_state = sampler.init_state(model, parameters, seed=1)

logger = nk.logging.RuntimeLog()  # stores historical data of the energy

for i in range(iterations):
    sampler_state = sampler.reset(model, parameters, state=sampler_state)
    samples, sampler_state = sampler.sample(model, parameters, state=sampler_state, chain_length=chain_length)
    E, E_grad = estimate_energy_and_gradient(model, parameters, hamiltonian_jax, samples)
    update = jax.tree.map(lambda x : eta * x, E_grad)
    parameters = jax.tree.map(lambda x,y : x-y, parameters, update)   # Gradient descent
    logger(step=i, item={'Energy': E})
    print(E)
    print(E.mean)


"""
print("Energy Differences: |E-E(GS)| :", np.abs(logger.data['Energy']['value']- e_gs))
print("------------------------------------------------------------------------------------")
# Plots
plt.figure(figsize=(12,6))
plt.semilogy(logger.data['Energy']['iters'], np.abs(logger.data['Energy']['value']- e_gs))
plt.ylim(18.0105, 18.0109)
plt.xlabel("Iterations")
plt.ylabel("Energy Errors: |E-E(GS)|")
plt.show()"""

print(logger.data["Energy"].keys())
plt.plot(logger.data['Energy']['iters'], logger.data['Energy']['Mean'])
plt.show()

