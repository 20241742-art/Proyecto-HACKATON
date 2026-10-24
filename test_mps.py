"""Prueba del parser con un MPS SINTÉTICO que imita la estructura de qap10
(100 x + 4050 y, restricciones de enlace). Ejecutar: python3 test_mps.py"""
import gzip, os, tempfile, itertools
import numpy as np
from qap_core import synthetic_instance, cost_perm, brute_force
import qap_mps

def write_mps(path, F, D, name_y, base=1):
    n = F.shape[0]
    xs = [(i, j) for i in range(n) for j in range(n)]
    ys = [(i, j, k, l) for i in range(n) for k in range(i + 1, n) for j in range(n) for l in range(n) if j != l]
    xn = lambda i, j: f"x_{i+base}_{j+base}"
    yn = lambda i, j, k, l: name_y(i + base, j + base, k + base, l + base)
    rows = {f"A{i}": [xn(i, j) for j in range(n)] for i in range(n)}
    rows.update({f"B{j}": [xn(i, j) for i in range(n)] for j in range(n)})
    for i, k, j in itertools.product(range(n), range(n), range(n)):   # sum_l y_ijkl = x_ij
        if i < k:
            rows[f"L{i}_{j}_{k}"] = [xn(i, j)] + [yn(i, j, k, l) for l in range(n) if l != j]
            rows[f"R{i}_{k}_{j}"] = [xn(k, j)] + [yn(i, l, k, j) for l in range(n) if l != j]
    member = {}
    for r, cs in rows.items():
        for c in cs:
            member.setdefault(c, []).append(r)
    with gzip.open(path, "wt") as fh:
        fh.write("NAME test\nROWS\n N obj\n" + "".join(f" E {r}\n" for r in rows) + "COLUMNS\n")
        fh.write("    MARKER  'MARKER'  'INTORG'\n")
        for (i, j) in xs:
            for r in member[xn(i, j)]:
                fh.write(f"    {xn(i,j)}  {r}  {'-1' if r[0] in 'LR' else '1'}\n")
        for (i, j, k, l) in ys:
            c = yn(i, j, k, l)
            fh.write(f"    {c}  obj  {F[i,k]*D[j,l]}\n")
            for r in member.get(c, []):
                fh.write(f"    {c}  {r}  1\n")
        fh.write("    MARKER  'MARKER'  'INTEND'\nRHS\nENDATA\n")

F, D = synthetic_instance(10, seed=1)
F = F * 3                                        # escala distinta a la de D
ctrl = {m: brute_force(F[:m, :m], D[:m, :m])[0] for m in (3, 4)}
tmp = tempfile.mkdtemp()
for label, name_y in [("y_i_j_k_l", lambda i, j, k, l: f"y_{i}_{j}_{k}_{l}"),
                      ("y_i_k_j_l (otro orden)", lambda i, j, k, l: f"y_{i}_{k}_{j}_{l}")]:
    print(f"\n=== MPS sintético con nombres {label} ===")
    path = os.path.join(tmp, "t.mps.gz")
    write_mps(path, F, D, name_y)
    assert qap_mps.main(path, out_prefix=os.path.join(tmp, ""), control=ctrl)
    Fr = np.loadtxt(os.path.join(tmp, "F.csv"), delimiter=",")
    Dr = np.loadtxt(os.path.join(tmp, "D.csv"), delimiter=",")
    rng = np.random.default_rng(1)
    assert all(abs(cost_perm(Fr, Dr, p) - cost_perm(F, D, p)) < 1e-9 for p in (rng.permutation(10) for _ in range(200)))
print("\nOK: el parser recupera el costo en la escala correcta con ambos órdenes de nombre")
