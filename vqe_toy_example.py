import numpy as np
import matplotlib.pyplot as plt
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.primitives import StatevectorEstimator
from scipy.optimize import minimize

hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 1.0), ("IX", 1.0)])

exact_min = min(np.linalg.eigvalsh(hamiltonian.to_matrix()))
print(f"Classical Exact Result (Expected): {exact_min}")

ansatz = RealAmplitudes(num_qubits=2, reps=1)
ansatz.decompose().draw("mpl", style="iqp", filename="The_Ansatz.png") 

estimator = StatevectorEstimator()
history = [] 

def cost_function(params):
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    
    history.append(energy)
    return energy

initial_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi

result = minimize(
    cost_function, 
    initial_params, 
    method="COBYLA", 
    options={'maxiter': 60}
)

print(f"The Result of VQE: {result.fun}")

plt.figure(figsize=(8, 5))
plt.plot(history, label="VQE Energy", color="blue", linewidth=2)
plt.axhline(y=exact_min, color="red", linestyle="--", label="Exact Ground State (-2.0)")
plt.xlabel("Optimizer Iterations")
plt.ylabel("Energy Expectation Value")
plt.title("VQE Convergence for Custom 2-Qubit Hamiltonian")
plt.legend()
plt.grid(True)
plt.savefig("VQE_Graph.png")
plt.show()
