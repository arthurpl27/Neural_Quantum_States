# Neural-Quantum-States
This repository contains the following files:

(0) the_model : An implementation of a (4 x 4) Transverse Field Ising Model that is used in the subsequent Python files for the simulations. It contains the Hamiltonian along with the computation of the exact ground-state energy / eigenvector for benchmarking purposes. 

(1) Ising_Model_MH : A short implementation of a 2D Ising Model on a (50 x 50) lattice with a Metropolis-Hastings algorithm to estimate the energy of the system.

(2) MF_Ansatz : A (useless) neural quantum state with an implementation of a Mean-Field Ansatz on a (4 x 4) Transverse Field Ising Model based on 16 parameters (trainable weights). Update via plain gradient descent and no use of stochastic sampling (we perform an extraction of all the states in the Hilbert space). This implementation is just to get a gist of how NetKet works.  

(3) Jastrow_Ansatz : A (useless) neural quantum state with an implementation of a Jastrow Ansatz on a (4 x 4) Transverse Field Ising Model based on 16 parameters (trainable weights). Update via plain gradient descent and no use of stochastic sampling (we perform an extraction of all the states in the Hilbert space). 

(4) Jastrow_VMC : An implementation to estimate the ground-state energy of a (4 x 4) Transverse Field Ising Model based on 16 parameters (trainable weights). We sample configurations with variational Monte Carlo by sampling from a Jastrow ansatz. We update the parameters with a plain gradient descent. In this code, VMC has been implemented from scratch.

(5) FNN_NQS : An implementation of a neural quantum state with a feedforward neural network on a (4 x 4) Transverse Field Ising Model based on 16 parameters (trainable weights). Implementation in pure NetKet of the full algorithm for ground-state energy estimation via NQS with a stochastic reconfiguration update. The present code outputs a comparison of the energy, energy errors and relative energy errors for a Mean-Field ansatz, Jastrow ansatz and the feedforward network. This code can easily be modified to study the performance of the algorithm by tuning the parameters of the system or by changing some of the tools currently implemented (sampler, update rule, ...).    

(6) NQS_Optimizers : An implementation of VMC-NQS to see the effect of SR on different optimizers (SGD, Adam, RMSProp, Momentum). 

(7) FNN_Models : A comparison of the effect of different feedforward neural networks on the energy composed of 1L-32N, 2L-32N, 2L-64N, 3L-64N and 3L-128N (where L = layers and N = neurons) with a ReLU activation function and SGD + SR update. We performed a smoothing of the errors true dynamics using a Gaussian filter. WARNING ! The last two models are expensive to run, adapt the models for a specific hardware.
