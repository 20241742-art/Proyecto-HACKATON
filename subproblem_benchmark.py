"""
subproblem_benchmark.py -- Evalúa SOLO la parte cuántica, sin ruido de la búsqueda.
Para M subproblemas kxk (k^2 qubits) idénticos para todos los métodos mide:
  - fracción de muestras factibles
  - P(la mejor muestra factible tiene la energía óptima del subproblema)  (empates cuentan como acierto)
  - barrido de profundidad p y de lambda
Uso: python3 subproblem_benchmark.py [numpy|qiskit] [shots=64] [k=3]
Ojo: con k=3 (512 estados) y 1024 shots el azar acierta ~86 %: usar pocas shots o k=4.
"""
import sys
import numpy as np
from qap_core import *
from qap_qaoa import sub_qaoa_numpy, sub_qaoa_qiskit

backend = sys.argv[1] if len(sys.argv) > 1 else "numpy"
qaoa = sub_qaoa_numpy if backend == "numpy" else sub_qaoa_qiskit
F, D = synthetic_instance(10, seed=7)
M = 30
shots = int(sys.argv[2]) if len(sys.argv) > 2 else 64
K = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def make_subs(lam_scale):
    rng = np.random.default_rng(123)
    subs = []
    for _ in range(M):
        p = rng.permutation(10)
        S = sorted(rng.choice(10, K, replace=False).tolist())
        sub0 = build_subproblem(F, D, p, S, lam=0.0)
        lam = default_lambda(sub0["Q"], lam_scale)
        subs.append(build_subproblem(F, D, p, S, lam=lam))
    return subs


def evaluate(solver, subs, **kw):
    rng = np.random.default_rng(99)
    hit = feas = 0.0
    for sub in subs:
        ex = sub_exact(sub, rng)
        r = solver(sub, rng, shots=shots, **kw)
        feas += r["feasible_frac"]
        if r["best_assign"] is not None and \
                abs(assign_energy(sub, r["best_assign"]) - assign_energy(sub, ex["best_assign"])) < 1e-9:
            hit += 1
    return feas / len(subs), hit / len(subs)


print(f"backend={backend}, M={M} subproblemas {K}x{K} ({K*K} qubits), shots={shots}")
print(f"{'config':32s} {'frac.factible':>14s} {'P(óptimo)':>10s}")
subs = make_subs(1.0)
f, h = evaluate(sub_random_baseline, subs)
print(f"{'azar (línea base)':32s} {f:14.3f} {h:10.2f}")
for reps in (1, 2, 3):
    f, h = evaluate(qaoa, subs, reps=reps)
    print(f"{'QAOA p=%d, lambda x1.0' % reps:32s} {f:14.3f} {h:10.2f}")
for ls in (0.25, 0.5, 2.0):
    f, h = evaluate(qaoa, make_subs(ls), reps=2)
    print(f"{'QAOA p=2, lambda x%.2f' % ls:32s} {f:14.3f} {h:10.2f}")
