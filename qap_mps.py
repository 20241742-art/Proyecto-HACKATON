"""
qap_mps.py -- Recupera F y D de qap10 a partir del archivo MPS linealizado de MIPLIB.

Uso:
  python3 qap_mps.py qap10.mps.gz            # escribe F.csv y D.csv y verifica la escala

Cómo se recuperan los coeficientes (para explicarlo en la presentación):
  1. El MPS tiene 4150 variables = 100 x_ij (asignación) + 4050 y_ijkl (= x_ij*x_kl,
     con i<k y j!=l: 45 pares * 90). El objetivo solo pesa las y: coef(y_ijkl) = F_ik*D_jl.
  2. Se leen los coeficientes objetivo y la pertenencia a filas (sección COLUMNS).
  3. Los nombres de las variables NO se interpretan (en el archivo real son X001 / Y001A001,
     que no codifican (i,j,k,l)). La estructura se deduce de las filas: x = columnas con
     coeficiente negativo en las filas de enlace; las 2N filas de asignación (solo x, positivas)
     dan (i,j) de cada x; cada y toca las filas de exactamente dos x distintas, x_ij y x_kl,
     que es el producto que linealiza.
  4. Se arma el tensor C[i,j,k,l] (i<k) y se factoriza como F_ik*D_jl.
  5. Verificación de escala: (a) error de factorización ~0, (b) el costo calculado con
     F, D coincide con la suma directa de coeficientes del MPS en permutaciones al azar,
     (c) fuerza bruta en las subinstancias N=3 y N=4 debe dar 24 y 32 (óptimos de control).
"""
from __future__ import annotations
import gzip, sys
from collections import defaultdict
import numpy as np
from qap_core import recover_FD_from_cost_tensor, cost_perm, brute_force

CONTROL_OPT = {3: 24.0, 4: 32.0}


def read_mps(path: str):
    """Lee un MPS (.mps o .mps.gz). Devuelve:
       obj      : columna -> coeficiente en la fila objetivo
       rows     : fila -> [(columna, coeficiente)]   (solo filas no objetivo)
       row_order: filas en el orden de la sección ROWS
       cols     : columnas en orden de aparición"""
    opener = gzip.open if path.endswith(".gz") else open
    obj_row, section = None, None
    obj = defaultdict(float)
    rows = defaultdict(list)
    row_order = []
    cols = []
    with opener(path, "rt") as fh:
        for line in fh:
            if not line.strip() or line.startswith("*"):
                continue
            if not line[0].isspace():
                section = line.split()[0].upper()
                continue
            t = line.split()
            if section == "ROWS":
                if t[0].upper() == "N" and obj_row is None:
                    obj_row = t[1]
                else:
                    row_order.append(t[1])
            elif section == "COLUMNS":
                if "'MARKER'" in t:
                    continue
                c = t[0]
                if not cols or cols[-1] != c:
                    cols.append(c)
                for r, v in zip(t[1::2], t[2::2]):
                    if r == obj_row:
                        obj[c] += float(v)
                    else:
                        rows[r].append((c, float(v)))
    return obj, rows, row_order, cols


def parse_structure(rows, row_order, cols):
    """Recupera la estructura del QAP SIN depender de los nombres de las variables.

    * x_ij  = columnas con algún coeficiente NEGATIVO (en las filas de enlace).
    * y     = el resto de columnas.
    * Filas de asignación = filas sin ninguna y (todas con coeficiente > 0). Son 2N:
      las primeras N (en el orden de la sección ROWS) son las de elemento (sum_j x_ij = 1)
      y las siguientes N las de ubicación (sum_i x_ij = 1).
    * Cada fila de enlace contiene exactamente una x; cada y toca filas de exactamente
      DOS x distintas, (i,j) y (k,l): ese es el par al que corresponde el producto x_ij*x_kl.
    Devuelve (n, pos_x: col -> (i,j), pares_y: col -> ((i,j),(k,l)))."""
    xcols = {c for r in rows.values() for c, v in r if v < 0}
    ycols = [c for c in cols if c not in xcols]
    assign = [r for r in row_order if rows[r] and all(c in xcols and v > 0 for c, v in rows[r])]
    if len(assign) % 2 or not assign:
        raise ValueError(f"No encuentro las filas de asignacion (halladas {len(assign)}, se esperaban 2N)")
    n = len(assign) // 2
    elem_rows, loc_rows = assign[:n], assign[n:]
    elem_idx = {r: i for i, r in enumerate(elem_rows)}
    loc_idx = {r: j for j, r in enumerate(loc_rows)}
    pos_x = {}
    for r in elem_rows:
        for c, _ in rows[r]:
            pos_x[c] = [elem_idx[r], None]
    for r in loc_rows:
        for c, _ in rows[r]:
            pos_x[c][1] = loc_idx[r]
    pos_x = {c: tuple(p) for c, p in pos_x.items()}
    if len(pos_x) != n * n or any(p[1] is None for p in pos_x.values()):
        raise ValueError("Las filas de asignacion no definen una cuadricula N x N de variables x")
    # y -> conjunto de x que aparecen en las filas de enlace donde esta
    x_of_row = {}
    for r, members in rows.items():
        xs = [c for c, v in members if c in xcols]
        if r not in elem_idx and r not in loc_idx and len(xs) == 1:
            x_of_row[r] = xs[0]
    touched = defaultdict(set)
    for r, members in rows.items():
        if r in x_of_row:
            for c, v in members:
                if c in ycols:
                    touched[c].add(x_of_row[r])
    pares = {}
    for c in ycols:
        xs = touched[c]
        if len(xs) != 2:
            raise ValueError(f"La variable {c} toca {len(xs)} variables x (se esperaban 2)")
        a, b = sorted(xs)
        pares[c] = (pos_x[a], pos_x[b])
    return n, pos_x, pares


def build_tensor(n, pos_x, pares, obj):
    """C[i,j,k,l] (i<k) = coeficiente objetivo de la y que representa x_ij * x_kl."""
    C = np.zeros((n, n, n, n))
    for c, ((i, j), (k, l)) in pares.items():
        if i == k or j == l:
            raise ValueError(f"{c}: par de x incompatible con una permutacion")
        if i > k:
            i, j, k, l = k, l, i, j
        C[i, j, k, l] += obj.get(c, 0.0)
    lin = np.zeros((n, n))
    for c, (i, j) in pos_x.items():
        lin[i, j] += obj.get(c, 0.0)
    return C, lin


def tensor_cost(C, p):
    n = len(p)
    return sum(C[i, p[i], k, p[k]] for i in range(n) for k in range(i + 1, n))


def main(path, out_prefix="", control=CONTROL_OPT):
    obj, rows, row_order, cols = read_mps(path)
    n, pos_x, pares = parse_structure(rows, row_order, cols)
    print(f"Columnas: {len(cols)} | x: {len(pos_x)} | y: {len(pares)} | N = {n}")
    print("Ejemplos x:", list(pos_x.items())[:2], "| ejemplos y:", list(pares.items())[:1])
    C, lin = build_tensor(n, pos_x, pares, obj)
    if np.abs(lin).max() > 0:
        print("AVISO: hay coeficientes objetivo en x (términos lineales); revisar el modelo.")
    F, D, err = recover_FD_from_cost_tensor(C)
    print(f"(a) error de factorización C = F*D: {err:.2e}")

    rng = np.random.default_rng(0)
    n = F.shape[0]
    diff = max(abs(cost_perm(F, D, p) - tensor_cost(C, p)) for p in (rng.permutation(n) for _ in range(1000)))
    print(f"(b) |costo(F,D) - suma de coeficientes del MPS| en 1000 permutaciones: {diff:.2e}")

    ok = err < 1e-6 and diff < 1e-6
    for m, target in control.items():
        opt, perm, t = brute_force(np.triu(F[:m, :m], 1), D[:m, :m])
        flag = "OK" if abs(opt - target) < 1e-6 else "NO COINCIDE"
        ok &= flag == "OK"
        print(f"(c) N={m}: óptimo por fuerza bruta = {opt:g} (control {target:g}) {flag}  perm={perm.tolist()}")

    np.savetxt(out_prefix + "F.csv", F, delimiter=",", fmt="%.10g")
    np.savetxt(out_prefix + "D.csv", D, delimiter=",", fmt="%.10g")
    print(f"Escrito {out_prefix}F.csv y {out_prefix}D.csv" + ("" if ok else "  <-- NO usar hasta resolver los avisos"))
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1]) else 1)
