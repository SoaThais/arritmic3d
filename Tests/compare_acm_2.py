import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_acm_file(filepath, scale_factor=1.0):
    """
    Reads an .acm file and returns a dictionary indexed by coordinates.
    scale_factor: use 1.0 for MonoAlg3D (um -> mm) and 1000.0 for Arritmic3D.
    """
    data_dict = {}
    
    with open(filepath, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
                
            try:
                parts = line.split('[')
                header = parts[0].strip().split()
                coords_str = header[0]
                
                c = [float(val) * scale_factor for val in coords_str.split(',')]
                cx, cy, cz, dx, dy, dz = c
                
                ats_str = parts[1].split(']')[0].strip()
                ats = [float(val) for val in ats_str.split()]
                
                key = (round(cx, 3), round(cy, 3), round(cz, 3))
                
                data_dict[key] = {
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'ATs': np.array(ats)
                }
                
            except Exception as e:
                print(f"Warning: Line {line_num} ignored ({e}): {line}")
                
    return data_dict


def compare_results(mono_file, heap_file, fim_file):

    print("Loading MonoAlg3D data (reading in um) ...")
    mono_data = parse_acm_file(mono_file, scale_factor=1.0)
    
    print("Loading HEAP data (converting from mm to um) ...")
    heap_data = parse_acm_file(heap_file, scale_factor=1000.0)

    print("Loading FIM data (converting from mm to um) ...")
    fim_data = parse_acm_file(fim_file, scale_factor=1000.0)
    
    print(f"\nLoaded nodes - MonoAlg: {len(mono_data)}, HEAP: {len(heap_data)}, FIM: {len(fim_data)}")
    
    errors_by_beat = {}
    spatial_data = {}
    missing_nodes = 0
    
    vertex_offsets = [
        (-1, -1, -1), (+1, -1, -1), (-1, +1, -1), (+1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (-1, +1, +1), (+1, +1, +1)
    ]

    print("\nProcessing spatial interpolation and calculating errors...")
    
    for centroid, m_data in mono_data.items():
        cx, cy, cz = centroid
        dx, dy, dz = m_data['dx'], m_data['dy'], m_data['dz']
        mono_ats = m_data['ATs']
        
        heap_vertex_ats = []
        fim_vertex_ats = []
        
        # Encontra os 8 vértices no HEAP e FIM
        vertices_found = True
        for ox, oy, oz in vertex_offsets:
            vx = round(cx + ox * dx, 3)
            vy = round(cy + oy * dy, 3)
            vz = round(cz + oz * dz, 3)
            
            v_key = (vx, vy, vz)
            
            if v_key in heap_data and v_key in fim_data:
                heap_vertex_ats.append(heap_data[v_key]['ATs'])
                fim_vertex_ats.append(fim_data[v_key]['ATs'])
            else:
                vertices_found = False
                break
                
        if not vertices_found:
            missing_nodes += 1
            continue
            
        # Tira a média dos 8 vértices para HEAP e FIM
        heap_avg_ats = np.mean(heap_vertex_ats, axis=0)
        fim_avg_ats = np.mean(fim_vertex_ats, axis=0)
        
        n_beats = min(len(mono_ats), len(heap_avg_ats), len(fim_avg_ats))
        
        for b in range(n_beats):
            if b not in errors_by_beat:
                errors_by_beat[b] = {'heap': [], 'fim': []}
                spatial_data[b] = {
                    'x': [], 'y': [], 
                    'mono': [], 'heap': [], 'fim': [], 
                    'err_heap': [], 'err_fim': []
                }
            
            diff_heap = abs(mono_ats[b] - heap_avg_ats[b])
            diff_fim = abs(mono_ats[b] - fim_avg_ats[b])
            
            errors_by_beat[b]['heap'].append(diff_heap)
            errors_by_beat[b]['fim'].append(diff_fim)
            
            spatial_data[b]['x'].append(cx)
            spatial_data[b]['y'].append(cy)
            spatial_data[b]['mono'].append(mono_ats[b])
            spatial_data[b]['heap'].append(heap_avg_ats[b])
            spatial_data[b]['fim'].append(fim_avg_ats[b])
            spatial_data[b]['err_heap'].append(diff_heap)
            spatial_data[b]['err_fim'].append(diff_fim)
            
    if missing_nodes > 0:
        print(f"Warning: Could not interpolate {missing_nodes} nodes (missing vertices at the borders of the domain).")

    print("\n" + "="*85)
    print(f"{'ERROR REPORT (Activation Time - ms)':^85}")
    print("="*85)
    print(f"{'Beat':<5} | {'Method':<6} | {'Evaluated Nodes':<16} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'Max Error (ms)':<15}")
    print("-" * 85)

    for b in sorted(errors_by_beat.keys()):
        for method in ['heap', 'fim']:
            err_array = np.array(errors_by_beat[b][method])
            if len(err_array) > 0:
                mae = np.mean(err_array)
                rmse = np.sqrt(np.mean(err_array**2))
                max_err = np.max(err_array)
                print(f"{b:<5} | {method.upper():<6} | {len(err_array):<16} | {mae:<10.4f} | {rmse:<10.4f} | {max_err:<15.4f}")
        print("-" * 85)

    print("\nGenerating plots...")
    
    # for b in spatial_data.keys():
    for b in [7]:
        fig, axes = plt.subplots(1, 5, figsize=(32, 6))
        
        x = spatial_data[b]['x']
        y = spatial_data[b]['y']
        
        min_lat = min(min(spatial_data[b]['mono']), min(spatial_data[b]['heap']), min(spatial_data[b]['fim']))
        max_lat = max(max(spatial_data[b]['mono']), max(spatial_data[b]['heap']), max(spatial_data[b]['fim']))
        
        max_err = max(max(spatial_data[b]['err_heap']), max(spatial_data[b]['err_fim']))
        
        sc0 = axes[0].tripcolor(x, y, spatial_data[b]['mono'], shading='gouraud', cmap='jet', vmin=min_lat, vmax=max_lat)
        axes[0].set_title(f"MonoAlg3D LAT - Beat {b}")
        axes[0].set_xlabel("X (um)")
        axes[0].set_ylabel("Y (um)")
        axes[0].set_aspect('equal')
        plt.colorbar(sc0, ax=axes[0], label="Activation Time (ms)")
        
        sc1 = axes[1].tripcolor(x, y, spatial_data[b]['heap'], shading='gouraud', cmap='jet', vmin=min_lat, vmax=max_lat)
        axes[1].set_title(f"HEAP LAT - Beat {b}")
        axes[1].set_xlabel("X (um)")
        axes[1].set_ylabel("Y (um)")
        axes[1].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[1], label="Activation Time (ms)")

        sc2 = axes[2].tripcolor(x, y, spatial_data[b]['fim'], shading='gouraud', cmap='jet', vmin=min_lat, vmax=max_lat)
        axes[2].set_title(f"FIM LAT - Beat {b}")
        axes[2].set_xlabel("X (um)")
        axes[2].set_ylabel("Y (um)")
        axes[2].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[2], label="Activation Time (ms)")
        
        sc3 = axes[3].tripcolor(x, y, spatial_data[b]['err_heap'], shading='gouraud', cmap='jet', vmin=0, vmax=max_err)
        axes[3].set_title(f"HEAP Abs. Error - Beat {b}")
        axes[3].set_xlabel("X (um)")
        axes[3].set_ylabel("Y (um)")
        axes[3].set_aspect('equal')
        plt.colorbar(sc3, ax=axes[3], label="Absolute Error (ms)")

        sc4 = axes[4].tripcolor(x, y, spatial_data[b]['err_fim'], shading='gouraud', cmap='jet', vmin=0, vmax=max_err)
        axes[4].set_title(f"FIM Abs. Error - Beat {b}")
        axes[4].set_xlabel("X (um)")
        axes[4].set_ylabel("Y (um)")
        axes[4].set_aspect('equal')
        plt.colorbar(sc4, ax=axes[4], label="Absolute Error (ms)")
        
        plt.tight_layout()
        plot_filename = f"LAT_Comparison_Beat_{b}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_acm.py <monoalg_path.acm> <heap_path.acm> <fim_path.acm>")
        sys.exit(1)
        
    mono_file = sys.argv[1]
    heap_file = sys.argv[2]
    fim_file = sys.argv[3]
    
    compare_results(mono_file, heap_file, fim_file)