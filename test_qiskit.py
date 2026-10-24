"""Pruebas de la parte Qiskit. Ejecutar: python3 test_qiskit.py"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from qiskit.circuit.library import QAOAAnsatz
from qiskit.quantum_info import Statevector
from qap_core import *
from qap_qaoa import build_ising_operator, qaoa_probs_numpy, sub_qaoa_qiskit

F, D = synthetic_instance(6, seed=3)
sub = build_subproblem(F, D, np.random.default_rng(0).permutation(6), [1, 3, 4])

# 1. El SparsePauliOp reproduce la energía QUBO de las 512 bitstrings (orden de qubits correcto)
H, off = build_ising_operator(sub)
diag = np.real(H.to_matrix(sparse=True).diagonal()) + off
err = np.abs(diag - energies_all(sub["Q"], sub["const"])).max()
print("1. diag(SparsePauliOp) + offset vs energía QUBO: error máx =", err)
assert err < 1e-9

# 2. Réplica numpy == circuito de Qiskit con los mismos parámetros
reps = 2
Hn = H / ising_scale(sub["Q"], sub["const"])
ansatz = QAOAAnsatz(cost_operator=Hn, reps=reps)
print("   parámetros de QAOAAnsatz:", [p.name for p in ansatz.parameters])
theta = np.random.default_rng(5).uniform(0, np.pi, 2 * reps)
pq = Statevector(ansatz.assign_parameters(theta)).probabilities()
pn, _ = qaoa_probs_numpy(sub, theta, reps)
err = np.abs(pq - pn).max()
print("2. probabilidades Qiskit vs numpy (mismos parámetros): error máx =", err)
assert err < 1e-9

# 3. El subsolver de Qiskit corre de punta a punta
r = sub_qaoa_qiskit(sub, np.random.default_rng(1), shots=64, reps=2, restarts=1, maxiter=60)
print(f"3. sub_qaoa_qiskit: frac. factible={r['feasible_frac']:.3f}, profundidad={r['depth']}")
print("OK: pruebas de Qiskit pasan")
