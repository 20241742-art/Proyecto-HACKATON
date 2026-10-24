"""
run_experiment.py -- Comparación reproducible: exacto vs QAOA vs azar (mismo presupuesto).

Uso:
  python3 run_experiment.py --backend numpy            # sin Qiskit
  python3 run_experiment.py --backend qiskit           # con Qiskit
  python3 run_experiment.py --F F.csv --D D.csv --known_opt 340          # qap10 completo
  python3 run_experiment.py --F F.csv --D D.csv --n 4 --k 3               # primeras 4 filas/columnas
  python3 run_experiment.py --F F.csv --D D.csv --brute                   # además fuerza bruta N=10 (~3 min)

Por defecto shots=64: con k=3 hay 512 estados y con 1024 shots el azar ya encuentra
el óptimo del subproblema el 86 % de las veces, así que la comparación no discrimina.

Si no se pasan --F/--D se usa una instancia SINTÉTICA (NO es qap10; no hay óptimo 340).
"""
import argparse, json, time, platform
import scipy
import numpy as np
from qap_core import *
from qap_qaoa import sub_qaoa_numpy, sub_qaoa_qiskit

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=None, help="usar solo las primeras n filas/columnas")
ap.add_argument("--k", type=int, default=3)
ap.add_argument("--iters", type=int, default=30)
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--shots", type=int, default=64)
ap.add_argument("--backend", choices=["numpy", "qiskit"], default="numpy")
ap.add_argument("--F"); ap.add_argument("--D")
ap.add_argument("--known_opt", type=float, default=None, help="óptimo de referencia (340 para qap10); solo para evaluar")
ap.add_argument("--brute", action="store_true", help="forzar fuerza bruta aunque N>9")
a = ap.parse_args()

if a.F and a.D:
    F, D = load_instance(a.F, a.D, n=a.n); label = "qap10" + (f" (primeras {a.n} filas/columnas)" if a.n else "")
else:
    F, D = synthetic_instance(a.n or 10); label = "SINTÉTICA (no es qap10)"
n = F.shape[0]

qaoa = sub_qaoa_numpy if a.backend == "numpy" else sub_qaoa_qiskit
solvers = {"exacto (k!)": sub_exact, f"QAOA-{a.backend}": qaoa, "azar (mismas shots)": sub_random_baseline}

print(f"Instancia: {label}, N={n}, vecindario k={a.k} ({a.k**2} qubits), {a.iters} iteraciones, {a.seeds} semillas")
if n <= 9 or a.brute:
    opt, bp, t = brute_force(F, D)
    print(f"Baseline clásico (fuerza bruta, {n}! perms): costo={opt:g}  asignación={bp.tolist()}  ({t:.1f}s)")
    if a.known_opt is not None and abs(opt - a.known_opt) > 1e-9:
        print(f"  AVISO: no coincide con el óptimo de referencia {a.known_opt:g}")
else:
    opt = a.known_opt
    print("Óptimo de referencia:", opt if opt is not None else "no disponible")

rows = {}
for name, s in solvers.items():
    res = [lns(F, D, s, k=a.k, iters=a.iters, seed=sd, shots=a.shots) for sd in range(a.seeds)]
    costs = [r["cost"] for r in res]
    best = res[int(np.argmin(costs))]
    fr = [r["mean_feasible_frac"] for r in res if r["mean_feasible_frac"] is not None]
    rows[name] = dict(best=min(costs), mean=float(np.mean(costs)), perm=best["perm"].tolist(),
                      feas=float(np.mean(fr)) if fr else None,
                      sec=float(np.mean([r["seconds"] for r in res])))
# Todas las permutaciones que devuelve lns son factibles por construcción (no hay reparación).
print(f"\n{'método':22s} {'mejor':>7s} {'media':>7s} {'frac.factible':>14s} {'seg/corrida':>12s}" + ("   gap(mejor)" if opt else ""))
for name, r in rows.items():
    gap = f"{100*(r['best']-opt)/opt:10.2f}%" if opt else ""
    feas = "n/a" if r["feas"] is None else f"{r['feas']:.3f}"
    print(f"{name:22s} {r['best']:7.0f} {r['mean']:7.1f} {feas:>14s} {r['sec']:12.2f}   {gap}")
print("\nMejor asignación por método (elemento i -> ubicación perm[i]):")
for name, r in rows.items():
    print(f"  {name:22s} {r['perm']}")

rep = dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
           seeds=list(range(a.seeds)), k=a.k, iters=a.iters, shots=a.shots, backend=a.backend,
           qaoa=dict(reps=2, restarts=3, maxiter=150, optimizer="COBYLA"))
if a.backend == "qiskit":
    import qiskit; rep["qiskit"] = qiskit.__version__
print("\nReproducibilidad:", json.dumps(rep))
