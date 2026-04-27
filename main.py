from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import SamplerV2 as Sampler
from qiskit_algorithms import QAOA
from qiskit_algorithms.optimizers import COBYLA
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_aer import AerSimulator
import networkx as nx
import matplotlib.pyplot as plt
import math


def hc_compiler(graph,freq_states):
    """
    Converts a classical cost function into a cost hamiltonian that can be used in the QAOA algorithm.

    Each qubit has f binary variables where f is the number of frequency available. 1 represents that qubit is selected
    for that given frequency and 0 represents that it was not selected.

    Ex: 2 qubits 3 possible frequency states
    010_001
    qubit 0 selected frequency in the index 1 of the frequency list
    qubit 1 selected frequency in the index 2 of the frequency list

    :param graph: (dict[int, list[int]]) Graph of qubit layout within the quantum computer
    :param freq_states: (list[float]) List of available frequencies that can be used
    :return: (list[tuple[str, float]]) List of all pauli string generated
    """

    qubits = len(graph)
    freqs = len(freq_states)
    hc_data = {}

    # Constraints each qubit to one frequency
    c1 = 1 # Constraint cost coff
    for i in range(qubits):
        for a in range(freqs):
            for b in range(a+1, freqs):
                set_pauli_term(pauli_string_config([a + freqs * i, b + freqs * i], freqs * qubits),
                               c1 * 0.5, hc_data) # cross terms
            set_pauli_term(pauli_string_config([a + freqs * i], freqs * qubits),
                           c1 * (2 - freqs) / 2, hc_data) # single terms
    set_pauli_term(pauli_string_config([], freqs * qubits),
                   c1 * qubits * ((2 - freqs) ** 2) / 4, hc_data) # constant term

    # Neighboring qubits should have different frequency's
    c2 = 1 # Neighboring qubit cost coff
    for qubit in graph: # Loops though each qubit connecting path
        for path in graph[qubit]:
            if path > qubit:
                for freq_a in range(freqs): # loops though each frequency pair
                    for freq_b in range(freqs):
                        spacing = math.fabs(freq_states[freq_a] - freq_states[freq_b]) # Frequency difference of 2 qubits
                        if spacing < 0.1:
                            c3 = 1 # Large penalty if spacing is less than 100 MHz
                        elif 0.1 <= spacing < 0.3:
                            c3 = 0.5 # Smaller penalty if spacing is between 100 MHz and 300 MHz
                        else:
                            c3 = 0 # No penalty if spacing is larger than 300 MHz
                        set_pauli_term(
                            pauli_string_config([freq_a + freqs * qubit, freq_b + freqs * path], freqs * qubits),
                            c2 * c3 * 0.25, hc_data)  # cross term
                        set_pauli_term(pauli_string_config([freq_a + freqs * qubit], freqs * qubits), c2 * c3 * -0.25,
                                       hc_data)  # single terms
                        set_pauli_term(pauli_string_config([freq_b + freqs * path], freqs * qubits), c2 * c3 * -0.25, hc_data)
                        set_pauli_term(pauli_string_config([], freqs * qubits), c2 * c3 * 0.25, hc_data)  # constant term



    return list(hc_data.items())


def set_pauli_term(pauli_string, coeff, hc_data):
    """
    Adds a coefficient to a Pauli term in the Hamiltonian dictionary.

    If the Pauli string already exists, the value is added to the existing
    coefficient. Otherwise, a new term is created.

    :param pauli_string: (String) Pauli string
    :param coeff: (float) Pauli term coefficient
    :param hc_data: (dict[str, float]) Dictionary of Hamiltonian terms
    """
    if pauli_string in hc_data:
        hc_data[pauli_string] += coeff
    else:
        hc_data[pauli_string] = coeff


def pauli_string_config(z_places, length):
    """
        Creates a Pauli string with Z operators at selected positions.

        :param z_places: (list[int]) Indices where the Pauli string should contain "Z"
        :param length: (int) Length of the Pauli string

        :return:
            str: Pauli string made of "Z" and "I" characters

        Example: string_config([0, 2], 5) returns "ZIZII".
    """
    string = ""
    for i in range(length):
        if i in z_places:
            string += str("Z")
        else:
            string += str("I")
    return string


qubit_graph = { # Graph of all qubits/connections indexed starting from 0 in sequential order
    0 : [1, 2],
    1 : [0, 2],
    2 : [0, 1],
    3 : [1],
}

freq_list = [4.9, 5.1, 5.15, 6, 6.2] # List of available frequency's in GHz

h_c = SparsePauliOp.from_list(hc_compiler(qubit_graph, freq_list)) # Creates cost hamiltonian

sampler = Sampler()
optimizer = COBYLA()
backend = AerSimulator()

pm = generate_preset_pass_manager(
    optimization_level=1,
    backend=backend
)

qaoa = QAOA(sampler=sampler, optimizer=optimizer, reps=2, transpiler=pm)
result = qaoa.compute_minimum_eigenvalue(h_c)
qubit_result = result.best_measurement["bitstring"] # Get optimized results using the QAOA algorithm

qubit_freqs = {} # Frequency assigned to each qubit
for q in range(len(qubit_graph)):
    for freq in range(len(freq_list)):
        if qubit_result[q*len(freq_list) + freq] == "1":
            qubit_freqs[q] = freq_list[freq]

print("Qubit frequency's: " + str(qubit_freqs))

# Creates and plots results into a graph
G = nx.Graph()
for node, neighbors in qubit_graph.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

labels = {}
for node in G.nodes:
    labels[node] = f"{node}\n{qubit_freqs[node]} GHz"

pos = nx.spring_layout(G)

fig = plt.figure()
fig.canvas.manager.set_window_title("Qubit Frequency Graph")

nx.draw(
    G,
    pos,
    labels=labels,
    with_labels=True,
    node_size=1200,
    font_size=6
)
plt.show()
