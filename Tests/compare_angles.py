import os
import sys
import numpy as np
import matplotlib.pyplot as plt

def parse_acm_file(filepath, scale_factor=1.0):
    """
    Reads an .acm file and returns a dictionary indexed by coordinates.
    """
    data_dict = {}
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
        
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
                pass
    return data_dict

def calculate_errors_for_angle(mono_file, arrit_file, target_beat=7):
    """
    Calculates MAE and RMSE between MonoAlg3D and Arritmic3D for a specific beat.
    Returns (mae, rmse) or (None, None) if calculation fails.
    """
    mono_data = parse_acm_file(mono_file, scale_factor=1.0)
    arrit_data = parse_acm_file(arrit_file, scale_factor=1000.0)
    
    if not mono_data or not arrit_data:
        return None, None
        
    vertex_offsets = [
        (-1, -1, -1), (+1, -1, -1), (-1, +1, -1), (+1, +1, -1),
        (-1, -1, +1), (+1, -1, +1), (-1, +1, +1), (+1, +1, +1)
    ]
    
    errors = []
    
    for centroid, m_data in mono_data.items():
        cx, cy, cz = centroid
        dx, dy, dz = m_data['dx'], m_data['dy'], m_data['dz']
        mono_ats = m_data['ATs']
        
        # Check if the target beat exists in the MonoAlg data
        if len(mono_ats) <= target_beat:
            continue
            
        arrit_vertex_ats = []
        vertices_found = True
        
        for ox, oy, oz in vertex_offsets:
            vx = round(cx + ox * dx, 3)
            vy = round(cy + oy * dy, 3)
            vz = round(cz + oz * dz, 3)
            
            v_key = (vx, vy, vz)
            if v_key in arrit_data:
                # Ensure the Arritmic node also has the target beat recorded
                if len(arrit_data[v_key]['ATs']) > target_beat:
                    arrit_vertex_ats.append(arrit_data[v_key]['ATs'][target_beat])
                else:
                    vertices_found = False
                    break
            else:
                vertices_found = False
                break
                
        if not vertices_found:
            continue
            
        arrit_avg_at = np.mean(arrit_vertex_ats)
        
        # Calculate signed difference (Mono - Arritmic)
        diff = mono_ats[target_beat] - arrit_avg_at
        errors.append(diff)
        
    if not errors:
        return None, None
        
    err_array = np.array(errors)
    
    # Calculate MAE (using absolute error) and RMSE
    mae = np.mean(np.abs(err_array))
    rmse = np.sqrt(np.mean(err_array**2))
    
    return mae, rmse

def main():
    angles = [0, 15, 30, 45, 60, 75, 90]
    target_beat = 7  # Index 7 corresponds to the 8th beat
    
    # Dictionaries to store results: {angle: error_value}
    mae_fim = []
    rmse_fim = []
    mae_heap = []
    rmse_heap = []
    valid_angles = []

    print(f"Comparing errors for Beat Index: {target_beat}")
    print("-" * 50)
    print(f"{'Angle':<10} | {'Solver':<10} | {'MAE (ms)':<10} | {'RMSE (ms)':<10}")
    print("-" * 50)
    
    for angle in angles:

        mono_path = f"../../Projetos/Fisiocomp/MonoAlg3D_C/Cable_100um_Endo_1000/{angle}/activation_info_it_0.acm"
        fim_path  = f"Tests/FIM/Plane/{angle}/Activation_Times.acm"
        heap_path = f"Tests/HEAP/Plane/{angle}/Activation_Times.acm"
        
        # Calculate FIM errors
        mae_f, rmse_f = calculate_errors_for_angle(mono_path, fim_path, target_beat)
        
        # Calculate Heap errors
        mae_h, rmse_h = calculate_errors_for_angle(mono_path, heap_path, target_beat)
        
        # If files exist and calculation succeeds, store data
        if mae_f is not None and mae_h is not None:
            valid_angles.append(angle)
            mae_fim.append(mae_f)
            rmse_fim.append(rmse_f)
            mae_heap.append(mae_h)
            rmse_heap.append(rmse_h)
            
            print(f"{angle:<10} | {'FIM':<10} | {mae_f:<10.4f} | {rmse_f:<10.4f}")
            print(f"{'':<10} | {'Heap':<10} | {mae_h:<10.4f} | {rmse_h:<10.4f}")
            print("-" * 50)
        else:
            print(f"Data missing or incomplete for angle {angle}°")

    if not valid_angles:
        print("No valid data found to plot. Please check the file paths.")
        return

    # --- Plotting ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot MAE
    ax1.plot(valid_angles, mae_fim, marker='o', label='FIM', color='blue')
    ax1.plot(valid_angles, mae_heap, marker='s', label='Heap', color='red')
    ax1.set_title('Mean Absolute Error (MAE) vs Fiber Angle')
    ax1.set_xlabel('Angle (Degrees)')
    ax1.set_ylabel('MAE (ms)')
    ax1.set_xticks(angles)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend()
    
    # Plot RMSE
    ax2.plot(valid_angles, rmse_fim, marker='o', label='FIM', color='blue')
    ax2.plot(valid_angles, rmse_heap, marker='s', label='Heap', color='red')
    ax2.set_title('Root Mean Square Error (RMSE) vs Fiber Angle')
    ax2.set_xlabel('Angle (Degrees)')
    ax2.set_ylabel('RMSE (ms)')
    ax2.set_xticks(angles)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('Error_vs_Angle_Comparison.png', dpi=150)
    print("Plot saved as 'Error_vs_Angle_Comparison.png'")
    
if __name__ == '__main__':
    main()