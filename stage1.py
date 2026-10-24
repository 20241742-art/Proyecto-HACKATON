"""
stage1.py -- Etapa 1 del reto: subinstancias N=3 y N=4 (primeras N filas/columnas de qap10).

Clásico: fuerza bruta (N! permutaciones).
Qiskit:  QAOA sobre el QUBO COMPLETO de N^2 qubits (9 o 16), statevector.
Azar:    mismas shots, bitstrings uniformes (línea base honesta).

Se reporta por separado:
  - costo de la mejor muestra FACTIBLE sin tocar (resultado crudo de QAOA)
  - costo tras reparación clásica (asignación factible más cercana en Hamming)
  - fracción factible, P(óptimo) entre las shots, tiempo
La energía QUBO penalizada se imprime aparte y rotulada; NO es el costo.

Uso:
  python3 stage1.py --F F.csv --D D.csv              # datos reales (de qap_mps.py)
  python3 stage1.py                                  # sintético, solo para probar
  python3 stage1.py --F F.csv --D D.csv --n 3 --shots 256 --reps 3
"""
import argparse, json, time, platform, math
import numpy as np
from qap_core import *
from qap_qaoa import sub_qaoa_qiskit

CONTROL_OPT = {3: 24.0, 4: 32.0}

ap = argparse.ArgumentParser()
ap.add_argument("--F"); ap.add_argument("--D")
ap.add_argument("--n", type=int, nargs="+", default=[3, 4])
ap.add_argument("--shots", type=int, default=256)
ap.add_argument("--reps", type=int, default=2)
ap.add_argument("--restarts", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()


def summarize(sub, idx_counts, E, opt):
    """idx_counts: dict índice->cuentas. Devuelve métricas crudas y reparadas por separado."""
    k = sub["k"]
    shots = sum(idx_counts.values())
    feas = hits = 0
    best_raw, perm_raw = np.inf, None
    best_rep, perm_rep = np.inf, None
    for i, c in idx_counts.items():
        bits = [(i >> v) & 1 for v in range(k * k)]
        ok, assign = decode(bits, k)
        if ok:
            feas += c
            cost = E[i]                                   # con S=todos y const: energía = costo
            hits += c * (abs(cost - opt) < 1e-9)
            if cost < best_raw:
                best_raw, perm_raw = cost, assign
        else:
            assign = repair_to_feasible(bits, k)
        cost = assign_energy(sub, assign)
        if cost < best_rep:
            best_rep, perm_rep = cost, assign
    return dict(feas_frac=feas / shots, p_opt=hits / shots,
                best_raw=best_raw, perm_raw=None if perm_raw is None else perm_raw.tolist(),
                best_repaired=best_rep, perm_repaired=perm_rep.tolist())


if a.F and a.D:
    label = "qap10 (primeras N filas/columnas)"
else:
    label = "SINTÉTICA (no es qap10; los óptimos de control 24/32 no aplican)"
print(f"Instancia: {label} | shots={a.shots}, p={a.reps}, reinicios={a.restarts}, semilla={a.seed}")

for n in a.n:
    if a.F and a.D:
        F, D = load_instance(a.F, a.D, n=n)
    else:
        F, D = synthetic_instance(n, seed=7)
    print(f"\n================ N = {n}  ({n*n} qubits, {2**(n*n)} estados, {math.factorial(n)} factibles)")

    # --- clásico
    opt, perm, t_bf = brute_force(F, D)
    print(f"Clásico (fuerza bruta): costo={opt:g}  asignación={perm.tolist()}  tiempo={t_bf*1e3:.2f} ms")
    if a.F and a.D and n in CONTROL_OPT:
        print(f"  verificación de escala: control={CONTROL_OPT[n]:g} -> {'OK' if abs(opt-CONTROL_OPT[n])<1e-9 else 'NO COINCIDE'}")

    sub = full_qubo(F, D)
    E = energies_all(sub["Q"], sub["const"])
    n_feas_states = math.factorial(n)
    print(f"QUBO: {n*n} variables, lambda={sub['lam']:g} (heurística: máx. suma |coef| por variable + 1); "
          f"estados factibles={n_feas_states}")

    # --- Qiskit QAOA
    rng = np.random.default_rng(a.seed)
    t0 = time.perf_counter()
    r = sub_qaoa_qiskit(sub, rng, shots=a.shots, reps=a.reps, restarts=a.restarts, maxiter=a.maxiter, seed=a.seed)
    t_q = time.perf_counter() - t0
    mq = summarize(sub, r["counts"], E, opt)

    # --- azar con las mismas shots
    idx = np.random.default_rng(a.seed).integers(0, 2 ** (n * n), size=a.shots)
    uniq, cnt = np.unique(idx, return_counts=True)
    mr = summarize(sub, dict(zip(uniq.tolist(), cnt.tolist())), E, opt)

    print(f"\n{'método':10s} {'frac.fact':>9s} {'P(ópt)':>7s} {'mejor crudo':>12s} {'mejor reparado':>15s} {'tiempo':>9s}")
    fmt = lambda v: "ninguno" if not np.isfinite(v) else f"{v:g}"
    print(f"{'QAOA':10s} {mq['feas_frac']:9.3f} {mq['p_opt']:7.3f} {fmt(mq['best_raw']):>12s} {fmt(mq['best_repaired']):>15s} {t_q:8.2f}s")
    print(f"{'azar':10s} {mr['feas_frac']:9.3f} {mr['p_opt']:7.3f} {fmt(mr['best_raw']):>12s} {fmt(mr['best_repaired']):>15s} {'-':>9s}")
    print(f"QAOA: asignación cruda={mq['perm_raw']}  reparada={mq['perm_repaired']}  "
          f"profundidad={r['depth']}  energía QUBO normalizada óptima (NO es costo)={r['opt_energy_norm']:.4f}")
    print(f"Factibles al azar esperados: {n_feas_states / 2**(n*n):.4f} de las shots")

import qiskit, scipy
print("\nReproducibilidad:", json.dumps(dict(python=platform.python_version(), numpy=np.__version__,
      scipy=scipy.__version__, qiskit=qiskit.__version__, seed=a.seed, shots=a.shots, reps=a.reps,
      restarts=a.restarts, maxiter=a.maxiter, optimizer="COBYLA", simulator="StatevectorEstimator/Sampler")))
