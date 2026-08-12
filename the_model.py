import numpy as np
import netket as nk


def build_model(
    h=1,
    J=1,
    L=4,
    seed=0,
    N_chains=20,
    compute_ground_state=True,
    verbose=False):

    # ============================================================
    #                       PARAMETERS
    # ============================================================

    g = nk.graph.Hypercube(length=L, n_dim=2, pbc=True)

    if verbose:
        print("------------------------------------------------------------------------------------")
        print("================ PARAMETERS ================")
        print("Number of sites:", g.n_nodes)
        print("List of sites:", [node for node in g.nodes()])
        print("Number of edges:", g.n_edges)
        print("List of edges:", g.edges())
        print("------------------------------------------------------------------------------------")

    # ============================================================
    #               CONSTRUCTION OF THE HAMILTONIAN
    # ============================================================

    # Define Hilbert space based on this graph
    hi = nk.hilbert.Spin(s=1/2, N=g.n_nodes)

    sigma_x = []
    sigma_zz = []

    for site in g.nodes():
        sigma_x.append(nk.operator.spin.sigmax(hi, site))

    for (i,j) in g.edges():
        sigma_zz.append(nk.operator.spin.sigmaz(hi, i) @ nk.operator.spin.sigmaz(hi, j))

    hamiltonian = -h * np.sum(sigma_x) + J * np.sum(sigma_zz)
    hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()
    hamiltonian_sparse = hamiltonian.to_sparse()

    if verbose:
        print("================ HAMILTONIAN ================")
        print(hamiltonian)
        print("------------------------------------------------------------------------------------")

        print("================ JAX HAMILTONIAN ================")
        print(hamiltonian_jax)
        print("------------------------------------------------------------------------------------")

    # ============================================================
    #       COMPUTE EXACT ENERGY/STATE OF THE GROUND STATE
    # ============================================================

    e_gs = None
    psi_gs = None

    if compute_ground_state:
        if verbose:
            print("================ GROUND STATE ENERGY & EIGENVECTOR ================")

        e_gs, psi_gs = nk.exact.lanczos_ed(hamiltonian, compute_eigenvectors=True)

        if verbose:
            print("Ground state energy:", e_gs)
            print("Ground state:", psi_gs)
            print("------------------------------------------------------------------------------------")

    return {
        "g": g,
        "hi": hi,
        "hamiltonian": hamiltonian,
        "hamiltonian_jax": hamiltonian_jax,
        "hamiltonian_sparse": hamiltonian_sparse,
        "e_gs": e_gs,
        "psi_gs": psi_gs,
        "h": h,
        "J": J,
        "L": L,
        "seed": seed,
        "N_chains": N_chains}
