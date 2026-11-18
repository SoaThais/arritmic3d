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
                # Format: x,y,z,dx,dy,dz num_beats [ ATs... ] [ APDs... ]
                parts = line.split('[')
                
                # Part 0 contains coordinates and number of beats
                header = parts[0].strip().split()
                coords_str = header[0]
                
                c = [float(val) * scale_factor for val in coords_str.split(',')]
                cx, cy, cz, dx, dy, dz = c
                
                # Extract Activation Times (AT) from Part 1
                ats_str = parts[1].split(']')[0].strip()
                ats = [float(val) for val in ats_str.split()]
                
                # Use a tuple rounded to 3 decimal places as key to avoid 
                # floating point precision issues (resolution of 1 um)
                key = (round(cx, 3), round(cy, 3), round(cz, 3))
                
                data_dict[key] = {
                    'dx': dx, 'dy': dy, 'dz': dz,
                    'ATs': np.array(ats)
                }
                
            except Exception as e:
                print(f"Warning: Line {line_num} ignored ({e}): {line}")
                
    return data_dict

def compare_results(mono_file, arrit_file):

    print("Loading MonoAlg3D data (reading in um) ...")
    mono_data = parse_acm_file(mono_file, scale_factor=1.0)
    
    print("Loading Arritmic3D data (converting from mm to um) ...")
    arrit_data = parse_acm_file(arrit_file, scale_factor=1000.0)
    
    print(f"\nLoaded nodes - MonoAlg: {len(mono_data)}, Arritmic: {len(arrit_data)}")
    
    errors_by_beat = {}
    spatial_data = {}
    missing_nodes = 0
    
    # Offsets for the 8 vertices of a cube (centroid +/- dx, dy, dz)
    # Note that the dx in the file is the HALF-SPACING (real dx/2)
    vertex_offsets = [
        (-1, -1, -1), (+1, -1, -1), (-1, +1, -1), (+1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (-1, +1, +1), (+1, +1, +1)
    ]

    print("\nProcessing spatial interpolation and calculating errors...")
    
    for centroid, m_data in mono_data.items():

        cx, cy, cz = centroid
        dx, dy, dz = m_data['dx'], m_data['dy'], m_data['dz']
        mono_ats = m_data['ATs']
        
        arrit_vertex_ats = []
        
        # Find the 8 vertices in the Arritmic3D dictionary
        vertices_found = True
        for ox, oy, oz in vertex_offsets:
            vx = round(cx + ox * dx, 3)
            vy = round(cy + oy * dy, 3)
            vz = round(cz + oz * dz, 3)
            
            v_key = (vx, vy, vz)
            if v_key in arrit_data:
                arrit_vertex_ats.append(arrit_data[v_key]['ATs'])
            else:
                vertices_found = False
                break
                
        if not vertices_found:
            missing_nodes += 1
            continue
            
        # Average the ATs of the 8 vertices for each beat
        arrit_avg_ats = np.mean(arrit_vertex_ats, axis=0)
        
        # Number of beats can differ if one simulator blocked earlier
        n_beats = min(len(mono_ats), len(arrit_avg_ats))
        
        for b in range(n_beats):
            if b not in errors_by_beat:
                errors_by_beat[b] = []
                spatial_data[b] = {'x': [], 'y': [], 'mono': [], 'arrit': [], 'err': []}
            
            diff = abs(mono_ats[b] - arrit_avg_ats[b])
            errors_by_beat[b].append(diff)
            
            spatial_data[b]['x'].append(cx)
            spatial_data[b]['y'].append(cy)
            spatial_data[b]['mono'].append(mono_ats[b])
            spatial_data[b]['arrit'].append(arrit_avg_ats[b])
            spatial_data[b]['err'].append(diff)
            
    if missing_nodes > 0:
        print(f"Warning: Could not interpolate {missing_nodes} nodes (missing vertices at the borders of the Arritmic3D domain).")

    print("\n" + "="*75)
    print(f"{'ERROR REPORT (Activation Time - ms)':^75}")
    print("="*75)
    print(f"{'Beat':<6} | {'Evaluated Nodes':<16} | {'MAE (ms)':<10} | {'RMSE (ms)':<10} | {'Max Error (ms)':<15}")
    print("-" * 75)

    for b in sorted(errors_by_beat.keys()):
        err_array = np.array(errors_by_beat[b])
        
        mae = np.mean(err_array)
        rmse = np.sqrt(np.mean(err_array**2))
        max_err = np.max(err_array)
        
        print(f"{b:<6} | {len(err_array):<16} | {mae:<10.4f} | {rmse:<10.4f} | {max_err:<15.4f}")
    
    print("="*75)
    
    print("\nGenerating plots...")
    # Plot spatial data for each beat
    # for b in spatial_data.keys():
    for b in [7]:

        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        
        x = spatial_data[b]['x']
        y = spatial_data[b]['y']
        
        # Plot 1: MonoAlg3D LAT (Usando tripcolor com interpolação contínua)
        sc0 = axes[0].tripcolor(x, y, spatial_data[b]['mono'], shading='gouraud', cmap='jet')
        axes[0].set_title(f"MonoAlg3D LAT - Beat {b}")
        axes[0].set_xlabel("X (um)")
        axes[0].set_ylabel("Y (um)")
        axes[0].set_aspect('equal')
        plt.colorbar(sc0, ax=axes[0], label="Activation Time (ms)")
        
        # Plot 2: Arritmic3D LAT
        sc1 = axes[1].tripcolor(x, y, spatial_data[b]['arrit'], shading='gouraud', cmap='jet')
        axes[1].set_title(f"Arritmic3D LAT - Beat {b}")
        axes[1].set_xlabel("X (um)")
        axes[1].set_ylabel("Y (um)")
        axes[1].set_aspect('equal')
        plt.colorbar(sc1, ax=axes[1], label="Activation Time (ms)")
        
        # Plot 3: Absolute Error
        sc2 = axes[2].tripcolor(x, y, spatial_data[b]['err'], shading='gouraud', cmap='jet')
        axes[2].set_title(f"Absolute Error - Beat {b}")
        axes[2].set_xlabel("X (um)")
        axes[2].set_ylabel("Y (um)")
        axes[2].set_aspect('equal')
        plt.colorbar(sc2, ax=axes[2], label="Absolute Error (ms)")
        
        plt.tight_layout()
        plot_filename = f"LAT_Comparison_Beat_{b}.png"
        plt.savefig(plot_filename, dpi=150)
        plt.close()
        print(f"Saved plot: {plot_filename}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python compare_acm.py <monoalg_path.acm> <arritmic3d_path.acm>")
        sys.exit(1)
        
    mono_file = sys.argv[1]
    arrit_file = sys.argv[2]
    
    compare_results(mono_file, arrit_file)