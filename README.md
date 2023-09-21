# SAGE is an Analytical Green's function Evaluator
This project is an attempt to build an engine to aumatically expand a linear system of equations of motion (EOMs) of Green's functions for models of a single electron coupled to restricted hard-core bosons. Due to geometrical constraints arising from different configurations of these bosons, such systems are difficult to treat exactly and deriving a closed form recurrence relations for the EOMs is usually not possible. This engine simplifies the problem of deriving the system of EOMs explicitly and solving them numerically. This package relies heavily on SymPy for handling the equations and their symplifications and thus can be rather slow and tends to have a big memory footprint. The following modules are included:

### classes
This defines the two main classes representing the states (class `State`), and their linear combinations representing the EOMs (class `Equation`).

### expansion
This defines the functions that perform the expansion of the EOMs based on the propagator and interaction functions defined by the user for the Hamiltonian representing their problem. Also included are functions for converting the EOMs into matrix objects that can be populated according to their parameter values, which can then be solved using standard numerical procedures.

### operators
This defines a number of standard operators useful for building Hamiltonians. This list is not exhaustive and includes only processes needed for the problems I was solving.

### kcuf3
This defines the expansion operators for some of the Hamiltonians relevant for the physics of the KCuF<sub>3</sub> perovskite, with active e<sub>g</sub> orbitals and spins. These Hamiltonians can be complicated and thus the functions defined here look rather formidable. For more details see: 
[Phys. Rev. B **94**, 085117 (2016)](https://arxiv.org/abs/1612.08009);
[Phys. Rev. B **95**, 235153 (2017)](https://arxiv.org/abs/1706.06071);
[Phys. Rev. B **100**, 125109 (2019)](https://arxiv.org/abs/1908.02232).

### tj
This defines the expansion operators for several versions of the *t-J* Hamiltonian, including the 3-site exchange hopping, full quantum spin fluctuations, and the option of turning off certain constraints relevant for comparison with other numerical methods such as the self-consistent Born approximation.
