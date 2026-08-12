import numpy as np
import netket as nk


def build_ising_model(
    h=1.0,
    J=1.0,
    L=4,
    pbc=True,
    compute_ground_state=True,
    verbose=False,
):
    """
    Build a transverse-field Ising model on a 2D L x L periodic lattice.

    Parameters
    ----------
    h : float
        Transverse-field strength.
    J : float
        Interaction strength.
    L : int
        Linear size of the lattice.
    pbc : bool
        Whether to use periodic boundary conditions.
    compute_ground_state : bool
        Whether to compute the exact ground-state energy and vector.
    verbose : bool
        Whether to print model information.

    Returns
    -------
    dict
        Dictionary containing the graph, Hilbert space, Hamiltonian,
        JAX Hamiltonian, sparse Hamiltonian, and optionally the ground state.
    """

    # ------------------------------------------------------------------
    # Graph
    # ------------------------------------------------------------------
    graph = nk.graph.Hypercube(
        length=L,
        n_dim=2,
        pbc=pbc,
    )

    if verbose:
        print("-" * 80)
        print("================ PARAMETERS ================")
        print("Number of sites:", graph.n_nodes)
        print("List of sites:", list(graph.nodes()))
        print("Number of edges:", graph.n_edges)
        print("List of edges:", list(graph.edges()))
        print("-" * 80)

    # ------------------------------------------------------------------
    # Hilbert space
    # ------------------------------------------------------------------
    hilbert_space = nk.hilbert.Spin(
        s=1 / 2,
        N=graph.n_nodes,
    )

    # ------------------------------------------------------------------
    # Hamiltonian
    #
    # H = -h sum_i sigma_x(i)
    #     + J sum_<i,j> sigma_z(i) sigma_z(j)
    # ------------------------------------------------------------------
    sigma_x = [
        nk.operator.spin.sigmax(hilbert_space, site)
        for site in graph.nodes()
    ]

    sigma_zz = [
        nk.operator.spin.sigmaz(hilbert_space, i)
        @ nk.operator.spin.sigmaz(hilbert_space, j)
        for i, j in graph.edges()
    ]

    hamiltonian = -h * np.sum(sigma_x) + J * np.sum(sigma_zz)

    # Other representations
    hamiltonian_jax = hamiltonian.to_pauli_strings().to_jax_operator()
    hamiltonian_sparse = hamiltonian.to_sparse()

    # ------------------------------------------------------------------
    # Exact ground state
    # ------------------------------------------------------------------
    ground_state_energy = None
    ground_state = None

    if compute_ground_state:
        ground_state_energy, ground_state = nk.exact.lanczos_ed(
            hamiltonian,
            compute_eigenvectors=True,
        )

    if verbose:
        print("================ HAMILTONIAN ================")
        print(hamiltonian)
        print("-" * 80)

        if compute_ground_state:
            print("================ GROUND STATE ================")
            print("Ground state energy:", ground_state_energy)
            print("Ground state:", ground_state)
            print("-" * 80)

    return {
        "graph": graph,
        "hilbert_space": hilbert_space,
        "hamiltonian": hamiltonian,
        "hamiltonian_jax": hamiltonian_jax,
        "hamiltonian_sparse": hamiltonian_sparse,
        "ground_state_energy": ground_state_energy,
        "ground_state": ground_state,
    }
