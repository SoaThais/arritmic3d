import os
import arritmic3d as a3d
import sys
import numpy as np

sys.path.append('./test')
from plot_functions import plot_vtk, delete_case_dir

def run (case_dir, solver_type, f):

    delete_case_dir(case_dir)
    os.makedirs(os.path.join(case_dir, "input_data"), exist_ok=True)

    # --- PASSO 2: Construir a malha com estímulo central ---
    slab_path = os.path.join(case_dir, "input_data", "slab.vtk")
    slab_args = [
        slab_path,
        "--nnodes", "201", "51", "2",
        "--spacing", "0.1", "0.1", "0.1",
        "--field", "restitution_model", "8", # Tecido saudável
        # Define uma pequena região central e atribui-lhe o activation_region 1
        "--region", '{"shape" : "square", "cx" : -1.5, "cy" : 2.5, "r1" : 2.5, "r2" : 2.5, "activation_region" : 1}',
        "--field", "fibers_orientation", f"{f}"
    ]

    a3d.build_slab(args_list=slab_args, save=True)
    print("Malha gerada com sucesso!")

    # --- PASSO 3: Configurar a simulação ---
    config = {
        "VTK_INPUT_FILE": slab_path,
        "SAVE_ACM": True,
        "APD_MODEL": "config_TenTuscher",
        "CV_MODEL": "config_TenTuscher",
        "ELECTROTONIC_EFFECT": 0.0,
        "CV_MEMORY_COEFF": 0.0,
        "APD_MEMORY_COEFF": 0.0,
        "SIMULATION_DURATION": 8000,  # 100 ms é suficiente para ver a onda expandir
        "VTK_OUTPUT_PERIOD": 1,
        "VTK_OUTPUT_INITIAL_TIME": 8000,
        "COND_VELOC_TRANSVERSAL_REDUCTION": 0.268355,
        "PROTOCOL": [
            {
                "ACTIVATION_REGION": 1,  # Dispara a região central que criámos
                "FIRST_ACTIVATION_TIME": 1.1101069999999709,
                "N_STIMS_PACING": [8],   # Apenas 1 batimento
                "BCL": [1000]
            }
        ]
    }

    # --- PASSO 4: Executar a simulação ---
    print("\n--- A iniciar simulação ---")
    a3d.arritmic3d(case_dir, config=config, solver_type=solver_type)

angles = [0, 15, 30, 45, 60, 75, 90]

for a in angles:

    a_rad = np.deg2rad(a)

    f = f'[{np.cos(a_rad):.6f}, {np.sin(a_rad):.6f}, 0.0]'

    # --- HEAP ---
    case_dir    = f"Tests/HEAP/Plane/{a}/"
    solver_type = a3d.PropagationSolverType.LEGACY_HEAP

    run(case_dir, solver_type, f)

    # --- FIM ---
    case_dir    = f"Tests/FIM/Plane/{a}/"
    solver_type = a3d.PropagationSolverType.FIM_EIKONAL

    run(case_dir, solver_type, f)
