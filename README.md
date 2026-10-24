# Proyecto_Hackaton_IBM_quantium
NTT DATA – IBM Quantum Hackathon 2026 · Reto QAP10 (asignación cuadrática) · 24/10/2026

Costo (convención del reto, cada par una vez): `C(p) = Σ_{i<k} F[i,k] · D[p[i], p[k]]`.
Óptimo de referencia de qap10: 340. Óptimos de control: N=3 → 24, N=4 → 32.

## Archivos
| Archivo | Qué hace |
|---|---|
| `qap_mps.py` | Lee `qap10.mps.gz`, recupera F y D, **verifica la escala** y escribe `F.csv`, `D.csv` |
| `qap_core.py` | Costo, fuerza bruta, QUBO con penalización λ, Ising, LNS (búsqueda por vecindarios) |
| `qap_qaoa.py` | QAOA con Qiskit (`QAOAAnsatz` + primitivas Statevector) y réplica idéntica en numpy |
| `stage1.py` | Etapa 1: N=3 y N=4, fuerza bruta vs QAOA sobre el QUBO completo (9 / 16 qubits) vs azar |
| `run_experiment.py` | N=10: LNS con subsolver exacto vs QAOA vs azar (mismas shots) |
| `subproblem_benchmark.py` | Solo la parte cuántica: fracción factible y P(óptimo) por p y λ |
| `test_core.py`, `test_mps.py`, `test_qiskit.py` | Pruebas |

## Cómo correrlo
```bash
pip install numpy scipy "qiskit>=1.2"
python3 test_core.py && python3 test_mps.py && python3 test_qiskit.py

# 1. Datos: descargar qap10.mps.gz de https://miplib.zib.de/instance_details_qap10.html
python3 qap_mps.py qap10.mps.gz          # debe imprimir (c) N=3: 24 OK y N=4: 32 OK

# 2. Etapa 1
python3 stage1.py --F F.csv --D D.csv --shots 256

# 3. Etapa 2 (N=10)
python3 run_experiment.py --F F.csv --D D.csv --known_opt 340 --backend qiskit
python3 run_experiment.py --F F.csv --D D.csv --known_opt 340 --brute   # + baseline exacto (~3 min)
```

## Cómo recuperamos F y D (y verificamos la escala)
El MPS real tiene 4150 variables = 100 `x_ij` (`X001`…`X100`) + 4050 `y` (`Y001A001`…), que linealizan
`x_ij·x_kl`. **Los nombres no codifican (i,j,k,l)**, así que no se interpretan: la estructura se deduce
de las filas. `x` = columnas con coeficiente negativo en las filas de enlace; las 2N filas de asignación
(`NX01`…`NX20`, solo x) dan (i,j) de cada x; cada `y` toca las filas de exactamente dos x distintas, que
son el producto que linealiza. El coeficiente objetivo de `y` es `F_ik·D_jl`. Se arma el tensor y se
factoriza. Verificación sobre el archivo real: (a) error de factorización = 0; (b) el costo con F, D es
igual a la suma directa de coeficientes del MPS en 1000 permutaciones; (c) fuerza bruta: N=3 → 24,
N=4 → 32 y N=10 → **340** (≈100 s).

## Qué hace Qiskit y qué es clásico
- **Qiskit:** Hamiltoniano de Ising (`SparsePauliOp`), circuito QAOA (`QAOAAnsatz`, transpilado a
  H/RZ/RZZ/RX), evaluación de ⟨H⟩ (`StatevectorEstimator`) y muestreo (`StatevectorSampler`).
- **Clásico:** construir el QUBO, el optimizador COBYLA, decodificar y validar factibilidad,
  la reparación (si se usa) y aceptar/rechazar movimientos en LNS.

## λ (penalización)
`λ = máx_v Σ_w |Q_vw| + 1` (sobre el QUBO sin penalizar): ningún cambio de costo puede pagar
una violación de restricción. `test_core.py` comprueba que toda bitstring infactible tiene más
energía que la mejor factible. `subproblem_benchmark.py` barre λ × {0.25, 0.5, 1, 2}.

## Reglas de reporte
- "Costo" es siempre `cost_perm` sobre una permutación factible. La energía QUBO se rotula aparte.
- Si se usa reparación (distancia de Hamming a la permutación más cercana), se reportan el
  resultado crudo y el reparado por separado (`stage1.py` lo hace).
- El gap se calcula solo sobre soluciones válidas: `100·(C − 340)/340`.
- 340 se usa solo para evaluar, nunca como punto de partida.
- Shots por defecto 64 en LNS: con k=3 hay solo 512 estados, y con 1024 shots el muestreo
  al azar ya encuentra el óptimo del subproblema el 86 % de las veces.

## Versiones probadas
Python 3.13, numpy 2.5, scipy 1.18, qiskit 2.6.1. Semillas y parámetros se imprimen al final de cada script.
