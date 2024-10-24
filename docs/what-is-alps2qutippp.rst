What is this project about
~~~~~~~~~~~~~~~~~~~~~~~~~~

Alps2Qutip++ is a library that tries to connect two very popular and useful tools used in the area of quantum modeling of condense matter systems and quantum information:

* The ALPS (Algorithms and Libraries for Physics Simulations) project: an open-source initiative aimed at providing high-performance libraries and tools for simulating complex quantum systems, particularly in the field of condensed matter physics. Developed by a collaboration of researchers from various institutions, ALPS focuses on the implementation of state-of-the-art algorithms for many-body quantum systems. Among its key features are

  - **Algorithms**: Implementations of advanced algorithms like Quantum Monte Carlo (QMC), exact diagonalization, and density matrix renormalization group (DMRG).
  - **Libraries**: Modular libraries providing essential building blocks for developing new algorithms and simulation tools.
  - **Frameworks**: Tools for setting up and managing large-scale simulations with support for parallel computing.
  - **Interfaces**: User-friendly interfaces for configuring simulations and analyzing results.

For more information, visit the ALPSCore website: https://alpscore.org


* Qutip(Quantum Toolbox in Python):  is an open-source library designed for simulating the dynamics of open quantum systems. It is widely used in quantum mechanics and quantum computing research due to its comprehensive functionalities and ease of use. Among its features it can be mentioned
  - **Quantum Objects Creation**: Creation and manipulation of quantum states and operators, including kets, density matrices, and common quantum operators.
  - **Basic Operations**: Operations such as addition, multiplication, tensor products, and computation of Hermitian conjugates, traces, and eigenstates.
  - **Time Evolution**: Simulation of time evolution under both unitary and non-unitary dynamics, supporting Schrödinger and master equations.
  - **Visualization**: Tools for visualizing quantum states, such as Bloch sphere plots and quantum circuit diagrams.
  - **Advanced Topics**: Includes quantum optimal control and integration with machine learning libraries like TensorFlow and Jax.
  - **Community and Contributions**: Actively maintained with contributions from the research community, participating in initiatives like Google Summer of Code.

For more information, visit the QuTiP documentation: http://qutip.org and the GitHub repository: https://github.com/qutip/qutip



Alps2Qutip++ allows to load models defined using the ALPS library XML syntax and produce objects that can be used in Qutip functions, and vice-versa. The '++' is because it goes beyond, allowing to represent states and operators on lattice quantum systems in a very efficient way, in terms of the algebra of tensor products. For example, in Qutip, a Hamiltonian operator is stored as an sparse matrix, without taking into account the fact that in lattice models, Hamiltonians are linear combinations of local operators and few body terms. For example, in a transverse Ising model in a chain,

.. math::

   {\mathbf H}=\sum_{i=1}^{N} b \sigma_{x,i} - \sum_{i=1}^{N-1} J \sigma_{x,i}\sigma_{x,i+1}

even using sparse arrays, the number of non vanishing elements scales exponentially with the size of the chain. On the other hand, in terms of the elements of the algebra, the operator can be expressed just by a number of parameters that scales linearly with the size of the system. This kind of decompositions are also very useful to implement semi-analytical methods, like mean field approaches or spin-wave-like expansions. 


[1] [ALPS Montreal 20](https://www.thp.uni-koeln.de/trebst/Talks/ALPS_Montreal.pdf)

Bauer, B., Carr, L. D., Evertz, H. G., Feiguin, A., Freire, J., Fuchs, S., ... & Wessel, S. (2011). The ALPS project release 2.0: open source software for strongly correlated systems. Journal of Statistical Mechanics: Theory and Experiment, 2011(05), P05001.
