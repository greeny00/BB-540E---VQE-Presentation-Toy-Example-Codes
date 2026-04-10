import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from qiskit.quantum_info import SparsePauliOp
from qiskit.circuit.library import RealAmplitudes
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, Session
from qiskit_ibm_runtime import EstimatorV2 as Estimator

# --- API TOKEN BURAYA GELECEK ---
# Sadece bu kod çalıştığı sürece bellekte kalır, bilgisayara kaydedilmez.
MY_API_TOKEN = "t9VjIKEcYNKk6e5WoFbZqd9UHdSNY4WaYigsmN1TpX1f"

print("IBM sunucularına bağlanılıyor...")
service = QiskitRuntimeService(channel="ibm_quantum_platform", token=MY_API_TOKEN)

# En az meşgul olan gerçek donanımı seç
backend = service.least_busy(operational=True, simulator=False)
print(f"Seçilen Kuantum Bilgisayarı: {backend.name}")

# 1. Özgün Hamiltonian
hamiltonian = SparsePauliOp.from_list([("ZZ", 1.0), ("XI", 1.0), ("IX", 1.0)])

# 2. Ansatz Devresi
ansatz = RealAmplitudes(num_qubits=2, reps=1)

# --- GERÇEK DONANIM İÇİN DERLEME (TRANSPILATION) ---
target = backend.target
pm = generate_preset_pass_manager(target=target, optimization_level=3)
ansatz_isa = pm.run(ansatz)
hamiltonian_isa = hamiltonian.apply_layout(layout=ansatz_isa.layout)

# 3. Estimator ve VQE Maliyet Fonksiyonu
history = []

def cost_function(params, estimator):
    pub = (ansatz_isa, [hamiltonian_isa], [params])
    # Gerçek cihazda ölçüm yapılıyor
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]
    
    history.append(energy)
    print(f"İterasyon {len(history)} | Bulunan Enerji: {energy:.4f}")
    return energy

# 4. Bulut Üzerinde Klasik Optimizer'ı Çalıştırma
initial_params = np.random.rand(ansatz.num_parameters) * 2 * np.pi

print(f"VQE {backend.name} üzerinde standart modda başlatılıyor. IBM kuyruk durumuna göre bu işlem zaman alabilir...")

# Session kullanmadan doğrudan backend'e gönderiyoruz
estimator = Estimator(mode=backend)
estimator.options.default_shots = 4000 # Gürültüyü azaltmak için 4000 ölçüm

result = minimize(
    cost_function, 
    initial_params, 
    args=(estimator,),
    method="COBYLA", 
    # DİKKAT: Ücretsiz planda sırada çok beklememek için iterasyonu 10'a düşürdük. 
    # 60 iterasyon günlerce sürebilir!
    options={'maxiter': 10} 
)

# 5. Gerçek Donanım Grafiğini Çizdirme
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