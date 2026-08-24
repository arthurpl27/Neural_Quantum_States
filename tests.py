import time

import netket as nk
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import flax.linen as nn

from the_model import build_model


# ============================================================
#                       MODEL SETUP
# ============================================================

N_chains = 20
seed = 0

model = build_model(
    h=1,
    J=1,
    L=4,
    seed=seed,
    N_chains=N_chains,
    compute_ground_state=True,
    verbose=True
)

g = model["g"]
hi = model["hi"]
hamiltonian = model["hamiltonian"]
hamiltonian_jax = model["hamiltonian_jax"]
hamiltonian_sparse = model["hamiltonian_sparse"]
e_gs = model["e_gs"]
psi_gs = model["psi_gs"]

# Draw the graph, if desired
# g.draw()


# ============================================================
#             FEED-FORWARD NEURAL NETWORK MODEL
# ============================================================

class FeedForward(nn.Module):
    """
    Two-layer feed-forward neural network with ReLU activation.
    """

    hidden_size: int = 32

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)

        x = nn.Dense(self.hidden_size)(x)

        return jnp.sum(x, axis=-1)


# ============================================================
#                 VMC LEARNING-RATE STUDY
# ============================================================

print("================ VMC: Neural Network Study ================")

start = time.time()

Length = 10000
iterations = 2500

learning_rates = [
    0.1,
    0.05,
    0.01,
    0.005,
    0.001
]

# Dictionary for storing the results
results = {}


# ============================================================
#                  RUN VMC FOR EACH LEARNING RATE
# ============================================================

for eta in learning_rates:

    print(f"\nRunning Feed-Forward NN with eta = {eta}")

    # Create a separate sampler for each simulation
    sampler = nk.sampler.MetropolisSampler(
        hi,
        nk.sampler.rules.LocalRule(),
        n_chains=N_chains
    )

    # Construct a new neural network for this learning rate
    model_FFNN = FeedForward()

    # Construct the variational state
    vstate_FFNN = nk.vqs.MCState(
        sampler,
        model_FFNN,
        n_samples=Length
    )

    # Initialize the neural-network parameters
    vstate_FFNN.init_parameters()

    # Construct the optimizer using the current learning rate
    optimizer_FFNN = nk.optimizer.Sgd(
        learning_rate=eta
    )

    # Construct the VMC driver
    gs_FFNN = nk.driver.VMC(
        hamiltonian,
        optimizer_FFNN,
        variational_state=vstate_FFNN,
        preconditioner=nk.optimizer.SR(
            diag_shift=0.1
        )
    )

    # Initialize the logger
    log_FFNN = nk.logging.RuntimeLog()

    # Run the VMC optimization
    gs_FFNN.run(
        n_iter=iterations,
        out=log_FFNN
    )

    # Extract data from the logger
    data_FFNN = log_FFNN.data

    iterations_data = np.asarray(
        data_FFNN["Energy"].iters
    )

    energy_mean = np.asarray(
        data_FFNN["Energy"].Mean
    )

    energy_sigma = np.asarray(
        data_FFNN["Energy"].Sigma
    )

    # Calculate relative error
    relative_error = np.abs(
        energy_mean - e_gs[0]
    ) / np.abs(e_gs[0])

    # Store the results for this learning rate
    results[eta] = {
        "iterations": iterations_data,
        "energy": energy_mean,
        "sigma": energy_sigma,
        "relative_error": relative_error
    }

    # Print final results
    print(f"Final energy:      {energy_mean[-1]:.8f}")
    print(f"Relative error:    {relative_error[-1]:.8e}")


# ============================================================
#                          PLOTS
# ============================================================

# Avoid problems on a logarithmic scale if an error is exactly zero
epsilon = 1e-16

fig = plt.figure(figsize=(12, 5))

ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)


# ------------------- Energy -------------------

for eta in learning_rates:

    data = results[eta]

    ax1.errorbar(
        data["iterations"],
        data["energy"],
        yerr=data["sigma"],
        label=rf"$\eta={eta}$",
        alpha=0.8
    )

ax1.axhline(
    y=e_gs[0],
    color="green",
    linestyle="--",
    linewidth=1.5,
    label="Ground-state energy"
)

ax1.set_title("Feed-Forward NN: Energy")
ax1.set_xlabel("Iterations")
ax1.set_ylabel("Energy")
ax1.grid(alpha=0.3)
ax1.legend()


# ------------------- Relative Error -------------------

for eta in learning_rates:

    data = results[eta]

    ax2.semilogy(
        data["iterations"],
        np.maximum(data["relative_error"], epsilon),
        label=rf"$\eta={eta}$"
    )

ax2.set_title("Feed-Forward NN: Relative Energy Error")
ax2.set_xlabel("Iterations")
ax2.set_ylabel(
    r"$|E-E_{\mathrm{GS}}|/|E_{\mathrm{GS}}|$"
)
ax2.grid(alpha=0.3)
ax2.legend()


fig.suptitle(
    "Learning-Rate Study for the Feed-Forward Neural Network",
    fontsize=14
)

plt.tight_layout()
plt.show()


# ============================================================
#                    FINAL RESULTS SUMMARY
# ============================================================

print("\n================ Final Results ================")

for eta in learning_rates:

    data = results[eta]

    final_energy = data["energy"][-1]
    final_relative_error = data["relative_error"][-1]

    print(
        f"eta = {eta:<6} | "
        f"Energy = {final_energy:.8f} | "
        f"Relative error = {final_relative_error:.8e}"
    )


# ============================================================
#                    TOTAL SIMULATION TIME
# ============================================================

end = time.time()

print("\n----------------------------------------------")
print(
    "Total Simulation Time:",
    end - start,
    "seconds"
)
print("----------------------------------------------")

