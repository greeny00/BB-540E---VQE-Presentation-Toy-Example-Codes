import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator


MY_API_TOKEN = "***************************"

service = QiskitRuntimeService(channel="ibm_quantum_platform", token=MY_API_TOKEN)

# Choose least busy computer
backend = service.least_busy(operational=True, simulator=False)
print(f"Seçilen Kuantum Bilgisayarı: {backend.name}")


hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 1.0), ("IX", 1.0)])

ansatz = RealAmplitudes(num_qubits=2, reps=1)

target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
ansatz_isa = pm.run(ansatz)
hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

history = []

def cost_function(params, estimator):
    pub = (ansatz_isa, [hamiltonian_isa], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    
    history.append(energy)
    print(f"Iteration {len(history)} | Energy value: {energy:.4f}")
    return energy

initial_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi

estimator = Estimator(mode=backend)
estimator.options.default_shots = 4000 # To lower noise we use 4000

result = minimize(
    cost_function, 
    initial_params, 
    args=(estimator,),
    method="COBYLA", 
    options={'maxiter': 10} 
)

plt.figure(figsize=(8, 5))
plt.plot(history, label="VQE Energy (IBM Hardware)", color="purple", linewidth=2)
plt.axhline(y=-2.0, color="red", linestyle="--", label="Exact Ground State (-2.0)")
plt.xlabel("Optimizer Iterations")
plt.ylabel("Energy Expectation Value")
plt.title(f"VQE Convergence on {backend.name}")
plt.legend()
plt.grid(True)
plt.savefig("ibm_hardware_convergence.png")
plt.show()
