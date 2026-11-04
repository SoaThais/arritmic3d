"""
Varredura do BCL do S2 para achar a janela de vulnerabilidade (bloqueio -> reentrada -> conducao normal)
nos tres solvers: legacy, fim_eikonal (velocidade do alvo) e fim_eikonal com velocity_from_source.

Uso (na pasta do seu script do reentry, para achar plot_functions/arritmic3d):
    python scan_vulnerable_window.py                      # BCL 316..332
    python scan_vulnerable_window.py --bcl 318 330 1      # inicio fim passo (inteiros)
    python scan_vulnerable_window.py --solvers legacy fim_eikonal_src

Por rodada grava so 2 VTUs (em t_a e no fim) e le LAT/State deles.

Metricas (por BCL e solver):
  frac_S2   fracao dos nos com LAT >= t_ultimo_estimulo em t_a (= nos ativados pelo ultimo S2 ou depois).
            Perto de 1: o S2 varreu o tecido; menor: parte (a BZ) bloqueou.
  n_late    nos com LAT > t_ultimo_estimulo + LATE_MARGIN no ultimo frame: ha atividade sustentada
            (reentrada) depois de o ultimo estimulo ter acabado.
  classe    sustained (n_late > 0) | block (frac_S2 < BLOCK_FRAC) | conducted
Os limiares (BLOCK_FRAC, LATE_MARGIN) sao heuristicas: confira a tabela, nao so a classe.
"""
import argparse
import csv
import os
import shutil

import numpy as np
import pyvista as pv

import arritmic3d as a3d

# --- protocolo (igual ao seu script) ---------------------------------------------------------------
N_STIMS = [8, 2]
FIRST_TIME = 0.0
BCL_S1 = 420
DURATION = 4800
LATE_MARGIN = 500.0   # ms depois do ultimo estimulo
BLOCK_FRAC = 0.95
SLAB_ARGS = [
    "--nnodes", "60", "60", "2",
    "--spacing", "0.3", "0.3", "0.3",
    "--region-by-side", "south", "1",
    "--field", "restitution_model", "2",
    "--region", '{"shape" : "square", "cx" : 9.0, "cy" : 9.0, "r1" : 6.0, "r2" : 6.0, "restitution_model" : 5}',
]

SOLVERS = {
    # "legacy":          (a3d.PropagationSolverType.LEGACY_HEAP, {}),
    "fim_eikonal":     (a3d.PropagationSolverType.FIM_EIKONAL, {}),
}

def stim_times(bcl_s2):
    """Mesma regra de arr3D_activations.schedule_activation: o passo depois do ultimo estimulo de um
    bloco usa o BCL do bloco seguinte."""
    bcls = [BCL_S1, bcl_s2]
    t, times = FIRST_TIME, []
    for bi, (n, bcl) in enumerate(zip(N_STIMS, bcls)):
        for si in range(n):
            times.append(t)
            t += bcls[bi + 1] if (bi + 1 < len(bcls) and si + 1 >= n) else bcl
    return times


def run_one(slab_path, out_root, solver_name, bcl_s2):
    solver_type, extra = SOLVERS[solver_name]
    t_last = stim_times(bcl_s2)[-1]
    t_a = t_last + 300.0
    case_dir = os.path.join(out_root, f"{solver_name}_{bcl_s2}")
    shutil.rmtree(case_dir, ignore_errors=True)
    os.makedirs(case_dir, exist_ok=True)

    config = {
        "VTK_INPUT_FILE": slab_path,
        "APD_MODEL": "TenTuscher",
        "CV_MODEL": "TenTuscher",
        "ELECTROTONIC_EFFECT": 0.0,
        "CV_MEMORY_COEFF": 0.05,
        "CORRECTION_FACTOR_CV": 0.9,
        "APD_MEMORY_COEFF": 0.0,
        "SIMULATION_DURATION": DURATION,
        "VTK_OUTPUT_INITIAL_TIME": t_a,
        "VTK_OUTPUT_PERIOD": DURATION - t_a,        # frames em t_a e em DURATION
        "VTK_OUTPUT_FIELDS": ["State", "LAT"],
        "PROTOCOL": [{
            "ACTIVATION_REGION": 1,
            "FIRST_ACTIVATION_TIME": 0,
            "N_STIMS_PACING": N_STIMS,
            "BCL": [BCL_S1, bcl_s2],
        }],
    }
    config.update(extra)
    a3d.arritmic3d(case_dir, config=config, solver_type=solver_type)

    def lat_of(t):
        f = os.path.join(case_dir, f"slab_{int(t):05d}.vtu")
        return np.asarray(pv.read(f).point_data["LAT"], dtype=float)

    lat_a, lat_end = lat_of(t_a), lat_of(DURATION)
    frac_s2 = float(np.mean(lat_a >= t_last - 1e-3))
    n_late = int(np.sum(lat_end > t_last + LATE_MARGIN))
    cls = "sustained" if n_late > 0 else ("block" if frac_s2 < BLOCK_FRAC else "conducted")
    shutil.rmtree(case_dir, ignore_errors=True)
    return dict(solver=solver_name, bcl=bcl_s2, t_last_stim=t_last, frac_S2=round(frac_s2, 4),
                n_late=n_late, max_LAT_end=round(float(lat_end.max()), 1), classe=cls)


def main():
    ap = argparse.ArgumentParser()
    # ap.add_argument("--bcl", nargs=3, type=int, default=[316, 332, 1], metavar=("INI", "FIM", "PASSO"))
    ap.add_argument("--bcl", nargs=3, type=int, default=[333, 360, 1], metavar=("INI", "FIM", "PASSO"))
    ap.add_argument("--solvers", nargs="+", default=list(SOLVERS), choices=list(SOLVERS))
    ap.add_argument("--out", default="out_scan_window")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out, "slab"), exist_ok=True)
    slab_path = os.path.abspath(os.path.join(args.out, "slab", "slab.vtk"))
    a3d.build_slab(args_list=[slab_path] + SLAB_ARGS, save=True)

    rows = []
    for bcl in range(args.bcl[0], args.bcl[1] + 1, args.bcl[2]):
        for s in args.solvers:
            r = run_one(slab_path, args.out, s, bcl)
            rows.append(r)
            print(f"BCL={bcl:4d} {s:16s} frac_S2={r['frac_S2']:.3f} n_late={r['n_late']:5d} -> {r['classe']}", flush=True)

    csv_path = os.path.join(args.out, "scan.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("\nJanela de reentrada (classe == sustained) por solver:")
    for s in args.solvers:
        b = [r["bcl"] for r in rows if r["solver"] == s and r["classe"] == "sustained"]
        print(f"  {s:16s}: " + (f"{min(b)}..{max(b)} ms ({len(b)} BCLs)" if b else "nenhuma na faixa varrida"))
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()