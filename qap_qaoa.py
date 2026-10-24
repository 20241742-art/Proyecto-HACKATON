"""
qap_qaoa.py -- Subsolvers QAOA para vecindarios del QAP.

  * sub_qaoa_qiskit : QAOAAnsatz + StatevectorEstimator (energía) + StatevectorSampler (muestras)
  * sub_qaoa_numpy  : réplica del MISMO circuito en numpy (validación de la lógica sin Qiskit).
                      Ambos normalizan el Hamiltoniano con qap_core.ising_scale(), así que con
                      los mismos parámetros dan las mismas probabilidades (test_qiskit.py lo verifica).
                      Orden de parámetros = el de QAOAAnsatz: [beta_1..beta_p, gamma_1..gamma_p].

Ambos reciben el subproblema de qap_core.build_subproblem(), optimizan los 2p
parámetros con COBYLA (clásico) y devuelven la mejor asignación FACTIBLE entre las
muestras, junto con la fracción factible.

Qué hace Qiskit: construye el Hamiltoniano de Ising (SparsePauliOp), el circuito
variacional (capas de costo e^{-i*gamma*H} y mezclador e^{-i*beta*X}), evalúa <H> y muestra.
Qué es clásico: armar el QUBO, el optimizador COBYLA, decodificar/validar factibilidad,
aceptar o rechazar el movimiento en la búsqueda por vecindarios.

Convención de bits: x_v = (idx >> v) & 1, qubit v <-> variable v (little-endian de Qiskit).
"""
import numpy as np
from scipy.optimize import minimize
from qap_core import (qubo_to_ising, ising_scale, energies_all, _from_indices)


# ----------------------------------------------------------------------------
# Réplica numpy (misma matemática que el circuito de Qiskit)
# ----------------------------------------------------------------------------
def _apply_rx_all(psi, nq, beta):
    """Aplica e^{-i beta X} a cada qubit."""
    c, s = np.cos(beta), -1j * np.sin(beta)
    psi = psi.reshape([2] * nq)
    for ax in range(nq):
        a0 = np.take(psi, 0, axis=ax)
        a1 = np.take(psi, 1, axis=ax)
        psi = np.stack([c * a0 + s * a1, s * a0 + c * a1], axis=ax)
    return psi.reshape(-1)


def _qaoa_state_numpy(E, nq, gammas, betas):
    # E[idx] con idx little-endian; reshape([2]*nq) usa el primer eje como el bit
    # MÁS significativo, y _apply_rx_all aplica lo mismo a todos los ejes, así que
    # la simetría del mezclador hace irrelevante el orden de ejes.
    psi = np.full(2 ** nq, 1 / np.sqrt(2 ** nq), dtype=complex)
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * g * E)
        psi = _apply_rx_all(psi, nq, b)
    return psi


def qaoa_probs_numpy(sub, theta, reps):
    """Probabilidades del estado QAOA (theta = [betas..., gammas...], como QAOAAnsatz).
    La fase de costo usa E/escala; E y el Ising difieren solo en una constante
    (fase global), por eso coincide con el circuito de Qiskit."""
    nq = sub["k"] ** 2
    En = energies_all(sub["Q"], sub["const"]) / ising_scale(sub["Q"], sub["const"])
    psi = _qaoa_state_numpy(En, nq, theta[reps:], theta[:reps])
    p = np.abs(psi) ** 2
    return p / p.sum(), En


def sub_qaoa_numpy(sub, rng, shots=64, reps=2, restarts=3, maxiter=150):
    nq = sub["k"] ** 2

    def energy(theta):
        p, En = qaoa_probs_numpy(sub, theta, reps)
        return float(p @ En)

    best = None
    for _ in range(restarts):
        th0 = rng.uniform(0, np.pi, 2 * reps)
        r = minimize(energy, th0, method="COBYLA", options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r
    prob, _ = qaoa_probs_numpy(sub, best.x, reps)
    idx = rng.choice(2 ** nq, size=shots, p=prob)
    out = _from_indices(sub, idx, energies_all(sub["Q"], sub["const"]), "qaoa_numpy")
    out["opt_energy_norm"] = best.fun
    out["theta"] = best.x
    return out


# ----------------------------------------------------------------------------
# Qiskit (>= 1.2 / 2.x): primitivas V2 sobre simulador de statevector
# ----------------------------------------------------------------------------
def build_ising_operator(sub):
    """SparsePauliOp con el Hamiltoniano de Ising del subproblema (sin constante)."""
    from qiskit.quantum_info import SparsePauliOp
    h, J, offset = qubo_to_ising(sub["Q"], sub["const"])
    nq = sub["k"] ** 2
    terms = [("Z", [v], float(c)) for v, c in enumerate(h) if c != 0]
    terms += [("ZZ", [v, w], float(c)) for (v, w), c in J.items()]
    # from_sparse_list evita errores de orden de etiquetas (qubit 0 = más a la derecha)
    return SparsePauliOp.from_sparse_list(terms, num_qubits=nq), offset


def sub_qaoa_qiskit(sub, rng, shots=64, reps=2, restarts=3, maxiter=150, seed=None):
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.primitives import StatevectorEstimator, StatevectorSampler

    H, offset = build_ising_operator(sub)
    nq = sub["k"] ** 2
    # Normalizamos H para que gamma tenga escala comparable (no cambia el argmin).
    # Misma escala que la réplica numpy.
    Hn = H / ising_scale(sub["Q"], sub["const"])

    ansatz = QAOAAnsatz(cost_operator=Hn, reps=reps)    # H^n + capas costo/mezclador
    # Se transpila UNA vez a puertas básicas (RZ, RZZ, RX, H): sin esto el simulador
    # exponencia matrices en cada evaluación de COBYLA. No cambia el circuito.
    ansatz = transpile(ansatz, basis_gates=["h", "rx", "rz", "rzz"], optimization_level=0)
    estimator = StatevectorEstimator()

    def energy(theta):
        res = estimator.run([(ansatz, Hn, theta)]).result()[0]
        return float(res.data.evs)

    best = None
    for _ in range(restarts):
        th0 = rng.uniform(0, np.pi, ansatz.num_parameters)
        r = minimize(energy, th0, method="COBYLA", options=dict(maxiter=maxiter))
        if best is None or r.fun < best.fun:
            best = r

    qc = ansatz.assign_parameters(best.x)
    qc.measure_all()
    sampler = StatevectorSampler(default_shots=shots, seed=int(rng.integers(1 << 31)) if seed is None else seed)
    counts = sampler.run([qc]).result()[0].data.meas.get_counts()

    # bitstring de Qiskit: el carácter más a la derecha es el qubit 0
    E = energies_all(sub["Q"], sub["const"])
    idx = []
    for bs, c in counts.items():
        i = int(bs, 2)                       # int(bs,2) ya es little-endian respecto al qubit
        idx.extend([i] * c)
    out = _from_indices(sub, np.array(idx), E, "qaoa_qiskit")
    out["opt_energy_norm"] = best.fun
    out["theta"] = best.x
    out["counts"] = {int(bs, 2): c for bs, c in counts.items()}
    out["depth"] = qc.depth()
    out["n_rzz"] = qc.count_ops().get("rzz", 0)
    return out
