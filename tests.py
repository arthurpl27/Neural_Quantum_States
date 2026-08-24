import platform
import json
import time

import netket as nk
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp

import flax
import flax.linen as nn

from scipy.sparse.linalg import eigsh
from functools import partial

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
#                     FLAX VARIATIONAL MODELS
# ============================================================

class MF(nn.Module):
    """
    Mean-field variational model.
    """

    @nn.compact
    def __call__(self, x):
        lam = self.param(
            "lambda",
            nn.initializers.normal(),
            (hi.size,),
            float
        )

        # Compute probabilities
        p = nn.log_sigmoid(lam * x)

        return 0.5 * jnp.sum(p, axis=-1)


class Jastrow(nn.Module):
    """
    Jastrow variational model.
    """

    @nn.compact
    def __call__(self, input_x):
        n_sites = input_x.shape[-1]

        J = self.param(
            "J",
            nn.initializers.normal(),
            (n_sites, n_sites),
            float
        )

        dtype = jax.numpy.promote_types(
            J.dtype,
            input_x.dtype
        )

        J = J.astype(dtype)
        input_x = input_x.astype(dtype)

        # Make J symmetric
        J_symm = J.T + J

        result = jnp.einsum(
            "...i,ij,...j->...",
            input_x,
            J_symm,
            input_x
        )

        return result


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
#                  VMC LEARNING-RATE STUDY
# ============================================================

print("================ VMC: Learning-Rate Study ================")

start = time.time()

Length = 10000
iterations = 300

learning_rates = [
    0.1,
    0.05,
    0.01,
    0.005,
    0.001
]

# Model constructors
model_factories = {
    "MF": MF,
    "Jastrow": Jastrow,
    "FNN": FeedForward
}

# Store all simulation results
results = {
    model_name: {}
    for model_name in model_factories
}


# ============================================================
#                  RUN ALL VMC SIMULATIONS
# ============================================================

for model_name, model_factory in model_factories.items():

    print(f"\n================ {model_name} ================")

    for eta in learning_rates:

        print(
            f"\nRunning model: {model_name} "
            f"with learning rate eta = {eta}"
        )

        # Create an independent sampler for this simulation
        sampler = nk.sampler.MetropolisSampler(
            hi,
            nk.sampler.rules.LocalRule(),
            n_chains=N_chains
        )

        # Construct the variational model
        variational_model = model_factory()

        # Construct the variational state
        vstate = nk.vqs.MCState(
            sampler,
            variational_model,
            n_samples=Length
        )

        # Initialize model parameters
        vstate.init_parameters()

        # Construct the optimizer with the current learning rate
        optimizer = nk.optimizer.Sgd(
            learning_rate=eta
        )

        # Construct the VMC driver
        vmc = nk.driver.VMC(
            hamiltonian,
            optimizer,
            variational_state=vstate,
            preconditioner=nk.optimizer.SR(
                diag_shift=0.1
            )
        )

        # Create a logger
        logger = nk.logging.RuntimeLog()

        # Run the VMC optimization
        vmc.run(
            n_iter=iterations,
            out=logger
        )

        # Extract data from the logger
        data = logger.data

        iterations_data = np.asarray(
            data["Energy"].iters
        )

        energy_mean = np.asarray(
            data["Energy"].Mean
        )

        energy_sigma = np.asarray(
            data["Energy"].Sigma
        )

        # Calculate errors
        absolute_error = np.abs(
            energy_mean - e_gs[0]
        )

        relative_error = absolute_error / np.abs(e_gs[0])

        # Store the results
        results[model_name][eta] = {
            "iterations": iterations_data,
            "energy": energy_mean,
            "sigma": energy_sigma,
            "absolute_error": absolute_error,
            "relative_error": relative_error
        }

        # Print final results for this simulation
        print(
            f"Final energy:       {energy_mean[-1]:.8f}"
        )

        print(
            f"Absolute error:     {absolute_error[-1]:.8e}"
        )

        print(
            f"Relative error:     {relative_error[-1]:.8e}"
        )


# ============================================================
#                         PLOTS
# ============================================================

# A small positive value avoids problems if an error becomes exactly zero
epsilon = 1e-16

for model_name in model_factories:

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 5)
    )

    ax_energy = axes[0]
    ax_absolute_error = axes[1]
    ax_relative_error = axes[2]

    for eta in learning_rates:

        data = results[model_name][eta]

        iterations_data = data["iterations"]
        energy = data["energy"]
        sigma = data["sigma"]
        absolute_error = data["absolute_error"]
        relative_error = data["relative_error"]

        label = rf"$\eta={eta}$"

        # ------------------- Energy -------------------

        ax_energy.errorbar(
            iterations_data,
            energy,
            yerr=sigma,
            label=label,
            alpha=0.8
        )

        # ------------------- Absolute Error -------------------

        ax_absolute_error.semilogy(
            iterations_data,
            np.maximum(absolute_error, epsilon),
            label=label
        )

        # ------------------- Relative Error -------------------

        ax_relative_error.semilogy(
            iterations_data,
            np.maximum(relative_error, epsilon),
            label=label
        )

    # Ground-state energy
    ax_energy.axhline(
        y=e_gs[0],
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="Ground-state energy"
    )

    # ------------------- Energy Plot -------------------

    ax_energy.set_title(f"{model_name}: Energy")
    ax_energy.set_xlabel("Iterations")
    ax_energy.set_ylabel("Energy")
    ax_energy.grid(alpha=0.3)
    ax_energy.legend()

    # ------------------- Absolute Error Plot -------------------

    ax_absolute_error.set_title(
        f"{model_name}: Absolute Energy Error"
    )

    ax_absolute_error.set_xlabel("Iterations")
    ax_absolute_error.set_ylabel(
        r"$|E-E_{\mathrm{GS}}|$"
    )

    ax_absolute_error.grid(alpha=0.3)
    ax_absolute_error.legend()

    # ------------------- Relative Error Plot -------------------

    ax_relative_error.set_title(
        f"{model_name}: Relative Energy Error"
    )

    ax_relative_error.set_xlabel("Iterations")
    ax_relative_error.set_ylabel(
        r"$|E-E_{\mathrm{GS}}|/|E_{\mathrm{GS}}|$"
    )

    ax_relative_error.grid(alpha=0.3)
    ax_relative_error.legend()

    fig.suptitle(
        f"Learning-rate study for the {model_name} model",
        fontsize=14
    )

    plt.tight_layout()
    plt.show()


# ============================================================
#                    FINAL RESULTS SUMMARY
# ============================================================

print("\n================ Final Results ================")

for model_name in model_factories:

    print(f"\n{model_name}:")

    for eta in learning_rates:

        data = results[model_name][eta]

        final_energy = data["energy"][-1]
        final_absolute_error = data["absolute_error"][-1]
        final_relative_error = data["relative_error"][-1]

        print(
            f"eta = {eta:<6} | "
            f"Energy = {final_energy:.8f} | "
            f"Absolute error = {final_absolute_error:.8e} | "
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
