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
# Draw the graph (uncomment line below)
# g.draw()

# Flax models
class MF(nn.Module):
    @nn.compact
    def __call__(self, x):
        lam = self.param("lambda", nn.initializers.normal(), (hi.size,), float)
        # compute probabilities
        p = nn.log_sigmoid(lam*x)
        return 0.5 * jnp.sum(p, axis=-1)

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

class FeedForward(nn.Module):      # 2-layer neural network with ReLU activation function and 32 neurons.
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
eta = 0.01   # learning parameter

# ---------- Models ----------
model_MF = MF()
model_jastrow = Jastrow()
model_FFNN = FeedForward()

# ---------- Sampler ----------
sampler = nk.sampler.MetropolisSampler(hi,                            # Hilbert space to be sampled
                                       nk.sampler.rules.LocalRule(),  # Transition rule: flip a random site
                                       n_chains = N_chains)

# ---------- Construct variational states ----------
vstate_MF = nk.vqs.MCState(sampler, model_MF, n_samples=Length)
vstate_jastrow = nk.vqs.MCState(sampler, model_jastrow, n_samples=Length)
vstate_FFNN = nk.vqs.MCState(sampler, model_FFNN, n_samples=Length)

# ---------- Initialise parameters ----------
vstate_MF.init_parameters()
vstate_jastrow.init_parameters()
vstate_FFNN.init_parameters()

# ---------- Construct optimizers ----------
optimizer_MF = nk.optimizer.Sgd(learning_rate = eta)      # Stochastic Gradient Descent
optimizer_jastrow = nk.optimizer.Sgd(learning_rate = eta) # Stochastic Gradient Descent
optimizer_FFNN = nk.optimizer.Sgd(learning_rate = eta)    # Stochastic Gradient Descent

# ---------- Construct optimizers ----------
gs_MF = nk.driver.VMC(hamiltonian,
                   optimizer_MF,
                   variational_state=vstate_MF,
                   preconditioner=nk.optimizer.SR(diag_shift=0.1))   # Stochastic Reconfiguration
gs_jastrow = nk.driver.VMC(hamiltonian,
                   optimizer_jastrow,
                   variational_state=vstate_jastrow,
                   preconditioner=nk.optimizer.SR(diag_shift=0.1))   # Stochastic Reconfiguration
gs_FFNN = nk.driver.VMC(hamiltonian,
                   optimizer_FFNN,
                   variational_state=vstate_FFNN,
                   preconditioner=nk.optimizer.SR(diag_shift=0.1))   # Stochastic Reconfiguration

# ---------- Initialize loggers ----------
log_MF = nk.logging.RuntimeLog()
log_jastrow = nk.logging.RuntimeLog()
log_FFNN = nk.logging.RuntimeLog()

# ---------- Run optimizers ----------
gs_MF.run(n_iter = iterations, out = log_MF)
gs_jastrow.run(n_iter = iterations, out = log_jastrow)
gs_FFNN.run(n_iter = iterations, out = log_FFNN)

# ---------- Estimate energies ----------
MF_energy = vstate_MF.expect(hamiltonian)
jastrow_energy = vstate_jastrow.expect(hamiltonian)
FFNN_energy = vstate_FFNN.expect(hamiltonian)

# ---------- Relative Errors ----------
error_MF = jnp.abs((MF_energy.mean - e_gs[0])/e_gs[0])
error_jastrow = jnp.abs((jastrow_energy.mean - e_gs[0])/e_gs[0])
error_FFNN = jnp.abs((FFNN_energy.mean - e_gs[0])/e_gs[0])
print("-------- MF ----------")
print(f"Optimized Energy:  {MF_energy}")
print(f"Relative Error: {error_MF}")
print("------- Jastrow ---------")
print(f"Optimized Energy:  {jastrow_energy}")
print(f"Relative Error: {error_jastrow}")
print("------- Feed Forward NN ---------")
print(f"Optimized Energy:  {FFNN_energy}")
print(f"Relative Error: {error_FFNN}")
print("------------------")

# ---------- Extract data from loggers ----------
data_MF = log_MF.data
data_jastrow = log_jastrow.data
data_FFNN = log_FFNN.data

# --------------- Energy Errors --------------------
error_MF = np.abs(data_MF["Energy"].Mean - e_gs[0])
error_jastrow = np.abs(data_jastrow["Energy"].Mean - e_gs[0])
error_FFNN = np.abs(data_FFNN["Energy"].Mean - e_gs[0])
RelErrors_MF = error_MF/np.abs(e_gs[0])
RelErrors_jastrow = error_jastrow/np.abs(e_gs[0])
RelErrors_FFNN = error_FFNN/np.abs(e_gs[0])

# ============================================================
#                          PLOTS
# ============================================================

fig = plt.figure(figsize=(12, 8))
gs = fig.add_gridspec(2, 4)


ax1 = fig.add_subplot(gs[0, 1:3])
ax2 = fig.add_subplot(gs[1, 0:2])
ax3 = fig.add_subplot(gs[1, 2:4])


# ------------------- Energy -------------------

ax1.errorbar(data_MF["Energy"].iters, data_MF["Energy"].Mean, yerr=data_MF["Energy"].Sigma, label="MF")
ax1.errorbar(data_jastrow["Energy"].iters, data_jastrow["Energy"].Mean, yerr=data_jastrow["Energy"].Sigma,
             label="Jastrow")
ax1.errorbar(data_FFNN["Energy"].iters, data_FFNN["Energy"].Mean, yerr=data_FFNN["Energy"].Sigma, label="FFNN")
ax1.axhline(y=e_gs[0], color="green", linestyle="--", linewidth=1, label="Ground State Energy")
ax1.set_title("Energy")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Energy")
ax1.grid(alpha=0.3)
ax1.legend()

# ------------------- Absolute error -------------------

ax2.semilogy(data_MF["Energy"].iters, error_MF, label="MF")
ax2.semilogy(data_jastrow["Energy"].iters, error_jastrow, label="Jastrow")
ax2.semilogy(data_FFNN["Energy"].iters, error_FFNN, label="FFNN")
ax2.set_title("Energy Error")
ax2.set_xlabel("Iterations")
ax2.set_ylabel(r"$|E-E_{\rm GS}|$")
ax2.grid(alpha=0.3)
ax2.legend()

# ------------------- Relative error -------------------

ax3.semilogy(data_MF["Energy"].iters, RelErrors_MF, label="MF")
ax3.semilogy(data_jastrow["Energy"].iters, RelErrors_jastrow, label="Jastrow")
ax3.semilogy(data_FFNN["Energy"].iters, RelErrors_FFNN,label="FFNN")
ax3.set_title("Relative Energy Error")
ax3.set_xlabel("Iterations")
ax3.set_ylabel(r"$|E-E_{\rm GS}|/E_{\rm GS}$")
ax3.grid(alpha=0.3)
ax3.legend()


plt.tight_layout()
plt.show()

end = time.time()
print("------------------")
print("Total Simulation Time : ", end-start, "seconds")
print("------------------------------------------------------------------------------------")
