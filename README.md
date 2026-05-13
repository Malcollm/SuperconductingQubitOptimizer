# SuperconductingQubitOptimizer
The project assigns resonance frequencies to superconducting qubits in a quantum computer, given a specific qubit layout. The algorithm I developed assigns frequencies in a way that minimizes frequency collisions, cross-talk, and qubit errors. To do this, the algorithm is given a qubit layout in the form of an undirected graph and a list of possible resonance frequencies to assign. There exist n*f variables being used where \\n\\ is the number of qubits, and f is the number of frequency states. Each qubit has \\f\\ variables where each one represents a frequency that the qubit could have. The variables are binary, such that they are compatible with the quantum optimization QAOA. 1 represents that the frequency is taken, and 0 represents that the frequency is not taken. This is the classical cost function.

$$
C(x) = A\sum_i \left(1 - \sum_a x_{i,a} \right)^2 + B\sum_{(i,j)\in E} \sum_a \sum_b P_{a,b}x_{i,a}x_{j,b}
$$

The first part is a constraint that fixes each qubit to only have one frequency selected. The second section sums over all coupled qubits with the tensor P. P is high for frequencies that are close together and 0 when they are far apart.

This cost function then needs to be converted to a cost Hamiltonian, and in order to do this, we need to convert all the variables into Pauli-Z operators.

$$
x_{i,a} = \frac{1 - Z_{i,a}}{2}
$$

Once this is done, the terms are simplified algebraically and can be converted into a cost Hamiltonian. This cost Hamiltonian is fed into the QAOA algorithm, which will output the frequency assignments. Finally, the program will display a visual of the optimal frequency assignments.
