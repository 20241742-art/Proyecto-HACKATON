"""
qap_core.py -- Núcleo del QAP (solo numpy, sin Qiskit).

Convención del reto (QAP10):  C(x) = sum_{i<k} sum_{j!=l} F_ik D_jl x_ij x_kl
Para una permutación p (elemento i -> ubicación p[i]):
    C(p) = sum_{i<k} F[i,k] * D[p[i], p[k]]      (cada par se cuenta UNA vez)

Todo lo que se reporta como "costo" pasa SIEMPRE por cost_perm() sobre la
permutación completa. La energía QUBO penalizada es otra cosa y se rotula aparte.
"""
from __future__ import annotations
import itertools
import time
import numpy as np


# ----------------------------------------------------------------------------
# 1. Datos
# ----------------------------------------------------------------------------
def synthetic_instance(n: int, seed: int = 7):
    """Instancia SINTÉTICA (no es qap10). Sirve para probar el pipeline.
    F: flujos enteros dispersos (matriz triangular superior, i<k).
    D: distancias enteras simétricas (distancia Manhattan en una cuadrícula)."""
    rng = np.random.default_rng(seed)
    F = np.triu(rng.integers(0, 6, size=(n, n)) * (rng.random((n, n)) < 0.6), 1)
    cols = int(np.ceil(np.sqrt(n)))
    pts = np.array([(r // cols, r % cols) for r in range(n)])
    D = np.abs(pts[:, None, :] - pts[None, :, :]).sum(-1)
    return F.astype(float), D.astype(float)


def load_instance(F_path: str, D_path: str, n: int | None = None):
    """Carga F y D desde .csv/.npy (generados por qap_mps.py a partir de qap10).
    F se usa solo por encima de la diagonal (convención i<k).
    Si se da n, se usan las primeras n filas y columnas (Etapa 1: N=3, N=4)."""
    load = lambda p: np.load(p) if p.endswith(".npy") else np.loadtxt(p, delimiter=",")
    F, D = load(F_path), load(D_path)
    if n is not None:
        F, D = F[:n, :n], D[:n, :n]
    return np.triu(F, 1).astype(float), D.astype(float)


def recover_FD_from_cost_tensor(C: np.ndarray):
    """Factoriza el tensor de costos C[i,j,k,l] = F_ik * D_jl (i<k) que arma qap_mps.py
    a partir de los coeficientes objetivo de las variables y_ijkl del MPS.
    F y D individualmente solo se determinan salvo un factor (F*c, D/c), pero el
    PRODUCTO F_ik*D_jl se reconstruye exacto, así que el costo queda en la escala del
    archivo. Se fija c para que el menor |D| no nulo valga 1 (enteros legibles).
    Devuelve F (i<k), D y el error máximo de reconstrucción de C (debe ser ~0)."""
    n = C.shape[0]
    mats = [C[i, :, k, :] for i in range(n) for k in range(i + 1, n) if np.abs(C[i, :, k, :]).sum() > 0]
    ref = max(mats, key=lambda m: np.abs(m).sum())
    D = ref / np.abs(ref[ref != 0]).min()
    F = np.zeros((n, n))
    for i in range(n):
        for k in range(i + 1, n):
            F[i, k] = (C[i, :, k, :] * D).sum() / (D * D).sum()
    for M in (F, D):                           # redondea si ya son enteros salvo ruido
        if np.allclose(M, np.round(M), atol=1e-9):
            M[:] = np.round(M)
    err = max(np.abs(C[i, :, k, :] - F[i, k] * D).max()
              for i in range(n) for k in range(i + 1, n))
    return F, D, float(err)


# ----------------------------------------------------------------------------
# 2. Costo y referencia clásica
# ----------------------------------------------------------------------------
def cost_perm(F, D, p) -> float:
    p = np.asarray(p)
    iu, ku = np.triu_indices(len(p), 1)
    return float((F[iu, ku] * D[p[iu], p[ku]]).sum())


def brute_force(F, D):
    """Enumera permutaciones una a una (sin guardarlas). Útil hasta N~9 en Python;
    N=10 (3.6M) tarda minutos. Devuelve (mejor_costo, mejor_perm, segundos)."""
    n = F.shape[0]
    t0 = time.perf_counter()
    best, bp = np.inf, None
    for p in itertools.permutations(range(n)):
        c = cost_perm(F, D, p)
        if c < best:
            best, bp = c, p
    return best, np.array(bp), time.perf_counter() - t0


# ----------------------------------------------------------------------------
# 3. QUBO completo (x_ij, n^2 variables) y Hamiltoniano de Ising
#    El mismo constructor sirve para un subproblema: se fijan elementos y se
#    liberan solo los de S (vecindario). Con S = todos, es el QUBO del reto.
# ----------------------------------------------------------------------------
def build_subproblem(F, D, p, S, lam: float | None = None):
    """QUBO del subproblema que reasigna los elementos S entre sus ubicaciones
    actuales, con el resto de la permutación p fija.
    Devuelve dict con Q (triangular superior; diag = términos lineales),
    const (constante de la penalización), fixed_cost (pares fijo-fijo),
    S, L (ubicaciones libres) y lam usado.
    Variable v = a*k + b  <->  x[S[a], L[b]]."""
    p = np.asarray(p)
    n = len(p)
    S = sorted(S)
    k = len(S)
    L = [int(p[a]) for a in S]
    fixed = [m for m in range(n) if m not in S]

    nv = k * k
    Q = np.zeros((nv, nv))
    idx = lambda a, b: a * k + b

    # términos lineales: interacción de cada libre con los fijos
    for a in range(k):
        for b in range(k):
            h = 0.0
            for m in fixed:
                q = int(p[m])
                h += F[S[a], m] * D[L[b], q] if S[a] < m else F[m, S[a]] * D[q, L[b]]
            Q[idx(a, b), idx(a, b)] += h
    # términos cuadráticos: pares libre-libre (a<c  ==  S[a]<S[c])
    for a in range(k):
        for c in range(a + 1, k):
            f = F[S[a], S[c]]
            if f == 0:
                continue
            for b in range(k):
                for d in range(k):
                    if b != d:
                        Q[idx(a, b), idx(c, d)] += f * D[L[b], L[d]]
    fixed_cost = sum(F[i, j] * D[p[i], p[j]] for i, j in itertools.combinations(fixed, 2))

    if lam is None:
        lam = default_lambda(Q)
    # Penalización  lam*(sum_b x_ab - 1)^2 + lam*(sum_a x_ab - 1)^2.
    # Con x^2 = x, cada restricción aporta: diagonal -lam, pares 2*lam, constante lam.
    # Cada variable está en 1 fila y 1 columna -> diagonal total -2*lam.
    for a in range(k):
        for b in range(k):
            Q[idx(a, b), idx(a, b)] += -2 * lam
    for a in range(k):                       # pares en la misma fila a
        for b in range(k):
            for d in range(b + 1, k):
                Q[idx(a, b), idx(a, d)] += 2 * lam
    for b in range(k):                       # pares en la misma columna b
        for a in range(k):
            for c in range(a + 1, k):
                Q[idx(a, b), idx(c, b)] += 2 * lam
    const = 2 * k * lam                      # lam por cada una de las 2k restricciones
    return dict(Q=Q, const=const, fixed_cost=float(fixed_cost), S=S, L=L, lam=lam, k=k)


def default_lambda(Q, scale: float = 1.0) -> float:
    """Heurística: lam >= máximo cambio de costo que puede 'comprar' violar una
    restricción = mayor suma de |coeficientes| que toca a una variable.
    scale<1 suaviza el paisaje (más muestras infactibles); se barre en los experimentos."""
    Qs = Q + Q.T - np.diag(np.diag(Q))
    return float(scale * np.abs(Qs).sum(axis=1).max()) + 1.0


def qubo_energy(Q, const, x) -> float:
    x = np.asarray(x, dtype=float)
    return float(x @ np.triu(Q) @ x + const)


def qubo_to_ising(Q, const):
    """x = (1 - Z)/2  ->  H = offset + sum_v h_v Z_v + sum_{v<w} J_vw Z_v Z_w."""
    nv = Q.shape[0]
    Qu = np.triu(Q)
    h = np.zeros(nv)
    J = {}
    offset = const
    for v in range(nv):
        h[v] += -Qu[v, v] / 2
        offset += Qu[v, v] / 2
    for v in range(nv):
        for w in range(v + 1, nv):
            q = Qu[v, w]
            if q != 0:
                J[(v, w)] = q / 4
                h[v] += -q / 4
                h[w] += -q / 4
                offset += q / 4
    return h, J, offset


def ising_scale(Q, const) -> float:
    """Suma de |coeficientes| del Hamiltoniano de Ising (sin la constante).
    Es la normalización que usan AMBOS subsolvers QAOA, para que sean el mismo circuito."""
    h, J, _ = qubo_to_ising(Q, const)
    return max(float(np.abs(h).sum() + sum(abs(c) for c in J.values())), 1.0)


def ising_energy(h, J, offset, x) -> float:
    z = 1 - 2 * np.asarray(x, dtype=float)
    return float(offset + h @ z + sum(c * z[v] * z[w] for (v, w), c in J.items()))


def energies_all(Q, const):
    """Energía QUBO de las 2^nv bitstrings (x_v = (idx >> v) & 1, igual que Qiskit)."""
    nv = Q.shape[0]
    idx = np.arange(2 ** nv)
    X = ((idx[:, None] >> np.arange(nv)) & 1).astype(float)
    Qu = np.triu(Q)
    return np.einsum("ni,ij,nj->n", X, Qu, X) + const


# ----------------------------------------------------------------------------
# 4. Decodificación y factibilidad
# ----------------------------------------------------------------------------
def decode(bits, k):
    """bits (len k*k) -> (es_factible, asignación a->b o None)."""
    M = np.asarray(bits, dtype=int).reshape(k, k)
    if (M.sum(0) == 1).all() and (M.sum(1) == 1).all():
        return True, M.argmax(1)
    return False, None


def apply_assignment(p, sub, assign):
    """Aplica la reasignación factible del subproblema a la permutación p."""
    q = np.array(p).copy()
    for a, b in enumerate(assign):
        q[sub["S"][a]] = sub["L"][b]
    return q


def repair_to_feasible(bits, k):
    """Reparación clásica: devuelve la asignación factible más cercana en distancia
    de Hamming a la bitstring (enumera k!, solo para k pequeño). Su resultado SE
    REPORTA APARTE del resultado crudo de las muestras, como pide el reto."""
    M = np.asarray(bits, dtype=int).reshape(k, k)
    best, ba = np.inf, None
    for perm in itertools.permutations(range(k)):
        d = np.abs(M - np.eye(k, dtype=int)[list(perm)]).sum()
        if d < best:
            best, ba = d, perm
    return np.array(ba)


# ----------------------------------------------------------------------------
# 5. Búsqueda por vecindarios grandes (LNS) con subsolver intercambiable
# ----------------------------------------------------------------------------
def sub_exact(sub, rng, shots=None):
    """Referencia clásica del subproblema: enumera k! reasignaciones."""
    k = sub["k"]
    E = energies_all(sub["Q"], sub["const"])
    best, ba = np.inf, None
    for perm in itertools.permutations(range(k)):
        x = np.zeros((k, k), dtype=int)
        x[np.arange(k), list(perm)] = 1
        e = E[bits_to_index(x.ravel())]
        if e < best:
            best, ba = e, np.array(perm)
    return dict(best_assign=ba, feasible_frac=None, n_samples=0, info="exact")


def sub_random_baseline(sub, rng, shots=64):
    """Línea base honesta: mismas 'shots' pero bitstrings uniformes al azar."""
    k, nv = sub["k"], sub["k"] ** 2
    E = energies_all(sub["Q"], sub["const"])
    idx = rng.integers(0, 2 ** nv, size=shots)
    return _from_indices(sub, idx, E, "random")


def full_qubo(F, D, lam=None):
    """QUBO del problema COMPLETO (n^2 variables): subproblema con S = todos y p = identidad,
    así la variable a*n+b es exactamente x[a, b]."""
    n = F.shape[0]
    return build_subproblem(F, D, np.arange(n), list(range(n)), lam=lam)


def assign_energy(sub, assign) -> float:
    """Energía QUBO de una asignación factible del subproblema (para comparar empates)."""
    k = sub["k"]
    x = np.zeros((k, k), dtype=int)
    x[np.arange(k), np.asarray(assign)] = 1
    return qubo_energy(sub["Q"], sub["const"], x.ravel())


def bits_to_index(bits):
    return int(sum(int(b) << v for v, b in enumerate(bits)))


def _from_indices(sub, idx, E, info):
    k, nv = sub["k"], sub["k"] ** 2
    best, ba, nfeas = np.inf, None, 0
    for i in idx:
        bits = [(int(i) >> v) & 1 for v in range(nv)]
        ok, a = decode(bits, k)
        if ok:
            nfeas += 1
            if E[int(i)] < best:
                best, ba = E[int(i)], a
    return dict(best_assign=ba, feasible_frac=nfeas / len(idx), n_samples=len(idx), info=info)


def lns(F, D, solver, k=3, iters=40, seed=0, p0=None, shots=64, verbose=False):
    """Mejora una permutación reoptimizando vecindarios de k elementos.
    Solo se acepta un cambio si el costo REAL de la permutación completa no empeora."""
    rng = np.random.default_rng(seed)
    n = F.shape[0]
    p = rng.permutation(n) if p0 is None else np.array(p0)
    cur = cost_perm(F, D, p)
    hist = [cur]
    fracs, t0 = [], time.perf_counter()
    for it in range(iters):
        S = sorted(rng.choice(n, size=k, replace=False).tolist())
        sub = build_subproblem(F, D, p, S)
        res = solver(sub, rng, shots=shots)
        if res["feasible_frac"] is not None:
            fracs.append(res["feasible_frac"])
        if res["best_assign"] is not None:
            q = apply_assignment(p, sub, res["best_assign"])
            c = cost_perm(F, D, q)
            if c <= cur:
                p, cur = q, c
        hist.append(cur)
        if verbose:
            print(f"it {it:3d} costo {cur:.0f}")
    return dict(perm=p, cost=cur, history=hist, mean_feasible_frac=float(np.mean(fracs)) if fracs else None,
                seconds=time.perf_counter() - t0)
