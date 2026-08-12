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

# Flax models
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

# =====================================================================================
#       INPUT MODEL & PARAMETERS - RETURNS: NORMALIZED EXPONENTIATED WAVEFUNCTION
# =====================================================================================
print("================ EXPONENTIATED WAVEFUNCTION ================")

model_jastrow = Jastrow()   # Model
key = jax.random.key(seed)           # RNG to initialise parameters randomly
parameters = model_jastrow.init(key, hi.random_state(jax.random.key(seed), (hi.size,)))
print("Initial Parameters:", parameters)

def to_array(model, parameters):
    all_configurations = hi.all_states()         # generate all configurations of the Hilbert space
    log_psi = model.apply(parameters, all_configurations)
    psis = jnp.exp(log_psi)
    Z = jnp.linalg.norm(psis)
    return psis / Z

start1 = time.time()
amplitudes = to_array(model_jastrow, parameters)
end1 = time.time()
print("Amplitudes:", amplitudes)
print("Compiling time:", end1-start1, "seconds")

start2 = time.time()
to_array_jit = jax.jit(to_array, static_argnames="model")     ### jit for faster computations ###
amplitudes_jit = to_array_jit(model_jastrow, parameters)
end2 = time.time()
print("Amplitudes jit:", amplitudes_jit)
print("Compilation time jit:", end2-start2, "seconds")
print("------------------------------------------------------------------------------------")

# ============================================================
#                        COMPUTE ENERGY
# ============================================================

print("================ COMPUTATION OF ENERGY ================")

hamiltonian_sparse = hamiltonian.to_sparse()
hamiltonian_jax_sparse = hamiltonian_jax.to_sparse(jax_ = True)

def compute_energy(model, parameters, hamiltonian_sparse):
    psi = to_array(model, parameters)
    Z = psi.conj().T @ psi
    return psi.conj().T @ (hamiltonian_sparse @ psi) / Z    # compute <psi|H|psi> / <psi|psi>

start3 = time.time()
energy = compute_energy(model_jastrow, parameters, hamiltonian_sparse)
end3 = time.time()
print("Energy:", energy)
print("Energy compilation time:", end3-start3, "seconds")


compute_energy_jit = jax.jit(compute_energy, static_argnames="model")   ### jit again for faster computation ###
start4 = time.time()
energy_jit = compute_energy_jit(model_jastrow, parameters, hamiltonian_jax_sparse)
end4 = time.time()
print("Energy jit:", energy_jit)
print("Energy compilation time jit:", end4-start4, "seconds")
print("------------------------------------------------------------------------------------")

# ============================================================
#                       COMPUTE GRADIENT
# ============================================================

print("================ COMPUTATION OF GRADIENT ================")

def compute_energy_and_gradient(model, parameters, hamiltonian_jax_sparse):
    grad_fun = jax.value_and_grad(compute_energy, argnums = 1)
    return grad_fun(model, parameters, hamiltonian_jax_sparse)

gradient_jax = jax.jit(compute_energy_and_gradient, static_argnames="model")     ### Always jit to compute gradient ###
jax_energy, jax_gradient = gradient_jax(model_jastrow, parameters, hamiltonian_jax_sparse)  # Energy is the same as before
print("Gradient:", jax_gradient)
print("------------------------------------------------------------------------------------")

# ============================================================
#                       ITERATION LOOP
# ============================================================

print("================ ITERATION LOOP ================")

iterations = 100
eta = 0.0005  # learning parameter
# model_jastrow = Jastrow()
# parameters = model_jastrow.init(...)

def multiply_by_learning_rate(x):
    return eta * x

def sub(x,y):
    return x - y

logger = nk.logging.RuntimeLog()  # stores historical data of the energy

for i in range(iterations):
    energy_iter, gradient_iter = gradient_jax(model_jastrow, parameters, hamiltonian_jax_sparse)
    update = jax.tree.map(multiply_by_learning_rate, gradient_iter)
    parameters = jax.tree.map(sub, parameters, update)   # Gradient descent
    logger(step=i, item={'Energy': energy_iter})


print("Energy differences: |E-E(GS)| :", np.abs(logger.data['Energy']['value']- e_gs))
print("Minimum energy difference:", np.min(np.abs(logger.data['Energy']['value']- e_gs)))
print("------------------------------------------------------------------------------------")
# Plots
plt.figure(figsize=(12,6))
plt.semilogy(logger.data['Energy']['iters'], np.abs(logger.data['Energy']['value']- e_gs))
plt.xlabel("Iterations")
plt.ylabel("Energy Errors: |E-E(GS)|")
plt.show()

