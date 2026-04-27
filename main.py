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
    qubits = len(graph)
    freqs = len(freq_states)
    hc_data = {}

    # Constraints each qubit to one frequency
    c1 = 1 # Constraint cost coff

    for i in range(qubits):
        for a in range(freqs):
            for b in range(a+1, freqs):
                set_pauli_term(string_config([a + freqs * i, b + freqs * i], freqs * qubits), c1 * 0.5, hc_data) # cross terms
            set_pauli_term(string_config([a + freqs * i], freqs * qubits), c1 * (2 - freqs) / 2, hc_data) # single terms
    set_pauli_term(string_config([],freqs * qubits), c1 * qubits * ((2 - freqs) ** 2) / 4, hc_data) # constant term

    # Neighboring qubits should have different frequency's
    c2 = 1 # Neighboring qubit cost coff (<100MHz)
    c3 = 0.3 # Neighboring qubit cost coff (100MHz< and <300MHz)

    for qubit in graph: # Loops though each qubit connecting path
        for path in graph[qubit]:
            if path > qubit:
                for freq_a in range(freqs): # loops though each frequency pair
                    for freq_b in range(freq_a, freqs):
                        spacing = math.fabs(freq_states[freq_a] - freq_states[freq_b]) # Frequency difference of 2 qubits
                        if spacing < 0.1: # Large penalty if spacing is less than 100 MHz
                            set_pauli_term(
                                string_config([freq_a + freqs * qubit, freq_b + freqs * path], freqs * qubits),
                                c2 * 0.25, hc_data)  # cross term
                            set_pauli_term(string_config([freq_a + freqs * qubit], freqs * qubits), c2 * -0.25,
                                           hc_data)  # single terms
                            set_pauli_term(string_config([freq_b + freqs * path], freqs * qubits), c2 * -0.25, hc_data)
                            set_pauli_term(string_config([], freqs * qubits), c2 * 0.25, hc_data)  # constant term
                        if 0.1 <= spacing < 0.3: # Moderate penalty if spacing is between 100 MHz and 300 MHz
                            set_pauli_term(
                                string_config([freq_a + freqs * qubit, freq_b + freqs * path], freqs * qubits),
                                c3 * 0.25, hc_data)  # cross term
                            set_pauli_term(string_config([freq_a + freqs * qubit], freqs * qubits), c3 * -0.25,
                                           hc_data)  # single terms
                            set_pauli_term(string_config([freq_b + freqs * path], freqs * qubits), c3 * -0.25, hc_data)
                            set_pauli_term(string_config([], freqs * qubits), c3 * 0.25, hc_data)  # constant term


    return list(hc_data.items())


def set_pauli_term(term, value, hc_data):
    if term in hc_data:
        hc_data[term] += value
    else:
        hc_data[term] = value


def string_config(z_places, length):
    string = ""
    for i in range(length):
        if i in z_places:
            string += str("Z")
        else:
            string += str("I")
    return string


qubit_graph = { # Graph of all qubits/connections indexed starting from 0 in sequential order
    0 : [1, 2, 3],
    1 : [0, 2, 3],
    2 : [0, 1, 3],
    3 : [0, 1, 2]
}

freq_list = [5, 4.7, 6, 5.1, 6.2] # List of available frequency's in GHz

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
qubit_result = result.best_measurement["bitstring"] # Get optimized results
print(qubit_result)

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