"""Pruebas del núcleo sin Qiskit. Ejecutar: python3 -I test_core.py"""
import itertools
import numpy as np
from qap_core import *

F, D = synthetic_instance(6, seed=3)
rng = np.random.default_rng(0)

# --- 1. QUBO del problema completo (S = todos): energía == costo en TODA permutación
p0 = np.arange(6)
sub = build_subproblem(F, D, p0, list(range(6)), lam=50.0)
k = sub["k"]
maxerr = 0.0
for perm in itertools.permutations(range(6)):
    x = np.zeros((6, 6), int)
    x[np.arange(6), list(perm)] = 1
    e = qubo_energy(sub["Q"], sub["const"], x.ravel())
    # perm[a] = índice de ubicación L[b]; con p0 = identidad, L = [0..5]
    maxerr = max(maxerr, abs(e - cost_perm(F, D, np.array(perm))))
print("1. QUBO completo vs costo en 720 permutaciones: error máx =", maxerr)
assert maxerr < 1e-9

# --- 2. Subproblema 3x3 con elementos fijos: costo_total = fixed_cost + energía
p = rng.permutation(6)
S = [1, 3, 4]
sub = build_subproblem(F, D, p, S, lam=50.0)
maxerr = 0.0
for perm in itertools.permutations(range(3)):
    x = np.zeros((3, 3), int)
    x[np.arange(3), list(perm)] = 1
    e = qubo_energy(sub["Q"], sub["const"], x.ravel())
    q = apply_assignment(p, sub, np.array(perm))
    maxerr = max(maxerr, abs(sub["fixed_cost"] + e - cost_perm(F, D, q)))
print("2. Subproblema 3x3 (con elementos fijos): error máx =", maxerr)
assert maxerr < 1e-9

# --- 3. Ising == QUBO en TODAS las 2^9 bitstrings
h, J, off = qubo_to_ising(sub["Q"], sub["const"])
E = energies_all(sub["Q"], sub["const"])
maxerr = 0.0
for i in range(2 ** 9):
    bits = [(i >> v) & 1 for v in range(9)]
    maxerr = max(maxerr, abs(ising_energy(h, J, off, bits) - E[i]))
print("3. Ising vs QUBO en 512 bitstrings: error máx =", maxerr)
assert maxerr < 1e-9

# --- 4. Con lam heurístico, toda bitstring infactible cuesta más que el mejor factible
sub = build_subproblem(F, D, p, S)  # lam por defecto
E = energies_all(sub["Q"], sub["const"])
best_feas = min(E[i] for i in range(512)
                if decode([(i >> v) & 1 for v in range(9)], 3)[0])
min_infeas = min(E[i] for i in range(512)
                 if not decode([(i >> v) & 1 for v in range(9)], 3)[0])
print(f"4. lam={sub['lam']:.1f}: mejor factible={best_feas:.1f}, mejor infactible={min_infeas:.1f}")
assert min_infeas > best_feas

# --- 5. Recuperar F y D desde el tensor C[i,j,k,l] (escala verificada)
n = 5
F5, D5 = synthetic_instance(n, seed=11)
C = np.einsum("ik,jl->ijkl", F5, D5)
Fr, Dr, err = recover_FD_from_cost_tensor(C)
print("5. Reconstrucción F,D desde tensor: error máx de C =", err)
assert err < 1e-9
pp = rng.permutation(n)
# el costo con (Fr, Dr) puede diferir de la escala original: se verifica reconstruyendo C
c_rec = cost_perm(Fr, Dr, pp)
scale = cost_perm(F5, D5, pp) / c_rec if c_rec else 1.0
print("   (F,D recuperadas son iguales salvo un factor global; razón =", round(scale, 6), ")")

# --- 6. LNS con subsolver exacto vs fuerza bruta (N=8)
F8, D8 = synthetic_instance(8, seed=5)
opt, popt, t = brute_force(F8, D8)
print(f"6. N=8 fuerza bruta: óptimo={opt:.0f} ({t:.1f}s)")
costs = [lns(F8, D8, sub_exact, k=3, iters=60, seed=s)["cost"] for s in range(10)]
print("   LNS(k=3, exacto) 10 semillas -> costos:", costs, "| brecha mín =",
      round(100 * (min(costs) - opt) / opt, 2), "%")
print("OK: todas las pruebas pasan")
