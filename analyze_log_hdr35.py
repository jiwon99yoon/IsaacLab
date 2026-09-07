# analyze_log_hdr35.py
# Analysis script for HDR35_20 F/T sensor data (simulation_log_hdr35_ft.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- File Loading ---
file_path = 'simulation_log_hdr35_ft.csv'

try:
    data = pd.read_csv(file_path)
    print(f"Successfully loaded: {file_path}")
    print(f"   Total records: {len(data)}")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
    print("   Please check the filename and path.")
    exit()

# --- Data Verification ---
print("\n--- Data Head (First 5 Rows) ---")
print(data.head())
print("\n--- Available Columns ---")
print(data.columns.tolist())
print("\n" + "="*60 + "\n")

# Set global plot style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10

# Define colors
colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C']
joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6']

#====================================================================================================================#
# Plot 1: Contact Force Components (X, Y, Z) + Magnitude
#====================================================================================================================#
print("Generating Plot 1: Contact Force Components...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['contact_force_x'], color='r', label='Contact Force X', linewidth=1.5)
plt.plot(data['step'], data['contact_force_y'], color='g', label='Contact Force Y', linewidth=1.5)
plt.plot(data['step'], data['contact_force_z'], color='b', label='Contact Force Z', linewidth=1.5)
plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude',
         linewidth=2, linestyle='--', alpha=0.7)

plt.title('Plot 1: Contact Force Components (X, Y, Z) and Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot1_contact_force_components.png', dpi=150)
print("   Saved: hdr35_plot1_contact_force_components.png")
plt.show()

#====================================================================================================================#
# Plot 2: Flange F/T Sensor Force (X, Y, Z)
#====================================================================================================================#
print("Generating Plot 2: Flange F/T Sensor Force...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['flange_force_x'], color='#FF6B6B', label='Flange Force X', linewidth=1.5)
plt.plot(data['step'], data['flange_force_y'], color='#4ECDC4', label='Flange Force Y', linewidth=1.5)
plt.plot(data['step'], data['flange_force_z'], color='#45B7D1', label='Flange Force Z', linewidth=1.5)
plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude',
         linewidth=2, linestyle=':', alpha=0.7)

plt.title('Plot 2: Flange F/T Sensor Force vs Contact Force Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot2_flange_force.png', dpi=150)
print("   Saved: hdr35_plot2_flange_force.png")
plt.show()

#====================================================================================================================#
# Plot 3: Flange F/T Sensor Torque (X, Y, Z)
#====================================================================================================================#
print("Generating Plot 3: Flange F/T Sensor Torque...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['flange_torque_x'], color='#C44569', label='Flange Torque X', linewidth=1.5)
plt.plot(data['step'], data['flange_torque_y'], color='#F8B500', label='Flange Torque Y', linewidth=1.5)
plt.plot(data['step'], data['flange_torque_z'], color='#A29BFE', label='Flange Torque Z', linewidth=1.5)
plt.plot(data['step'], data['contact_force_magnitude'] * 0.01, color='k', label='Contact Force Magnitude * 0.01',
         linewidth=2, linestyle=':', alpha=0.7)

plt.title('Plot 3: Flange F/T Sensor Torque vs Contact Force', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot3_flange_torque.png', dpi=150)
print("   Saved: hdr35_plot3_flange_torque.png")
plt.show()

#====================================================================================================================#
# Plot 4: Joint 2 Torque Comparison (Commanded vs External)
#====================================================================================================================#
print("Generating Plot 4: Joint 2 Torque Comparison...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude',
         linewidth=2, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['joint2_commanded'], color='#C44569', label='Joint2 Commanded', linewidth=1.5)
plt.plot(data['step'], data['joint2_external'], color='#F8B500', label='Joint2 External', linewidth=1.5)
plt.plot(data['step'], data['joint2_external_g'], color="#B41FBE", label='Joint2 External (with G)', linewidth=1.5, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j2'], color="#28D04D", label='OSC Torque J2', linewidth=1.2, alpha=0.8)

plt.title('Plot 4: Joint 2 Torque Comparison (Commanded, External, OSC)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot4_joint2_torque.png', dpi=150)
print("   Saved: hdr35_plot4_joint2_torque.png")
plt.show()

#====================================================================================================================#
# Plot 5: Joint 4 Torque Comparison (Commanded vs External)
#====================================================================================================================#
print("Generating Plot 5: Joint 4 Torque Comparison...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude',
         linewidth=2, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['joint4_commanded'], color='#C44569', label='Joint4 Commanded', linewidth=1.5)
plt.plot(data['step'], data['joint4_external'], color='#F8B500', label='Joint4 External', linewidth=1.5)
plt.plot(data['step'], data['joint4_external_g'], color="#B41FBE", label='Joint4 External (with G)', linewidth=1.5, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j4'], color="#28D04D", label='OSC Torque J4', linewidth=1.2, alpha=0.8)

plt.title('Plot 5: Joint 4 Torque Comparison (Commanded, External, OSC)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot5_joint4_torque.png', dpi=150)
print("   Saved: hdr35_plot5_joint4_torque.png")
plt.show()

#====================================================================================================================#
# Plot 6: Joint 6 Torque Comparison (Commanded vs External)
#====================================================================================================================#
print("Generating Plot 6: Joint 6 Torque Comparison...")
plt.figure(figsize=(14, 7))

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude',
         linewidth=2, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['joint6_commanded'], color='#C44569', label='Joint6 Commanded', linewidth=1.5)
plt.plot(data['step'], data['joint6_external'], color='#F8B500', label='Joint6 External', linewidth=1.5)
plt.plot(data['step'], data['joint6_external_g'], color="#B41FBE", label='Joint6 External (with G)', linewidth=1.5, linestyle=':', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j6'], color="#28D04D", label='OSC Torque J6', linewidth=1.2, alpha=0.8)

plt.title('Plot 6: Joint 6 Torque Comparison (Commanded, External, OSC)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot6_joint6_torque.png', dpi=150)
print("   Saved: hdr35_plot6_joint6_torque.png")
plt.show()

#====================================================================================================================#
# Plot 7: OSC Joint Torques (J1-J6)
#====================================================================================================================#
print("Generating Plot 7: OSC Joint Torques (All Joints)...")
plt.figure(figsize=(14, 7))

for i in range(6):
    col_name = f'osc_torque_j{i+1}'
    if col_name in data.columns:
        plt.plot(data['step'], data[col_name],
                 color=colors[i], label=f'OSC Torque {joint_labels[i]}',
                 linewidth=1.2, alpha=0.8)

plt.plot(data['step'], data['contact_force_magnitude'],
         color='k', label='Contact Force Magnitude',
         linewidth=2.5, linestyle=':', alpha=0.9)

plt.title('Plot 7: OSC Joint Torques (J1-J6) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=9, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot7_osc_joint_torques.png', dpi=150)
print("   Saved: hdr35_plot7_osc_joint_torques.png")
plt.show()

#====================================================================================================================#
# Plot 8: Flange vs Contact Force Comparison (Component-wise)
#====================================================================================================================#
print("Generating Plot 8: Flange vs Contact Force Comparison...")
plt.figure(figsize=(14, 7))

# Contact Force (dashed)
plt.plot(data['step'], data['contact_force_x'],
         color='#E74C3C', label='Contact Force X',
         linewidth=2, linestyle='--', alpha=0.9)
plt.plot(data['step'], data['contact_force_y'],
         color='#27AE60', label='Contact Force Y',
         linewidth=2, linestyle='--', alpha=0.9)
plt.plot(data['step'], data['contact_force_z'],
         color='#3498DB', label='Contact Force Z',
         linewidth=2, linestyle='--', alpha=0.9)

# Flange Force (solid)
plt.plot(data['step'], data['flange_force_x'],
         color='#FF69B4', label='Flange Force X',
         linewidth=1.5, linestyle='-', alpha=0.8)
plt.plot(data['step'], data['flange_force_y'],
         color='#32CD32', label='Flange Force Y',
         linewidth=1.5, linestyle='-', alpha=0.8)
plt.plot(data['step'], data['flange_force_z'],
         color='#FFA500', label='Flange Force Z',
         linewidth=1.5, linestyle='-', alpha=0.8)

plt.title('Plot 8: Contact Force vs Flange Force (Component-wise)', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10, ncol=2)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hdr35_plot8_contact_vs_flange_force.png', dpi=150)
print("   Saved: hdr35_plot8_contact_vs_flange_force.png")
plt.show()

#====================================================================================================================#
# Plot 9: All Link F/T Forces Comparison (if available)
#====================================================================================================================#
link_names = ['base_body_link', 'lower_frame_link', 'upper_frame_link',
              'arm_link', 'wrist_body_link', 'wrist_holder_link', 'flange_link']

# Check if link data exists
available_links = []
for link in link_names:
    if f'{link}_force_x' in data.columns:
        available_links.append(link)

if available_links:
    print("Generating Plot 9: All Link Forces Z-axis Comparison...")
    plt.figure(figsize=(14, 7))

    link_colors = plt.cm.viridis(np.linspace(0, 1, len(available_links)))

    for i, link in enumerate(available_links):
        plt.plot(data['step'], data[f'{link}_force_z'],
                 color=link_colors[i], label=f'{link} Force Z',
                 linewidth=1.5, alpha=0.8)

    plt.plot(data['step'], data['contact_force_magnitude'],
             color='k', label='Contact Force Magnitude',
             linewidth=2.5, linestyle=':', alpha=0.9)

    plt.title('Plot 9: All Link Forces (Z-axis) vs Contact Force', fontsize=14, fontweight='bold')
    plt.xlabel('Step', fontsize=12)
    plt.ylabel('Force (N)', fontsize=12)
    plt.legend(loc='best', fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hdr35_plot9_all_link_forces_z.png', dpi=150)
    print("   Saved: hdr35_plot9_all_link_forces_z.png")
    plt.show()

#====================================================================================================================#
# Summary Statistics
#====================================================================================================================#
print("\n" + "="*60)
print("Summary Statistics - HDR35_20 F/T Sensor Data")
print("="*60)

print("\n1. Contact Force:")
print(f"   Mean Magnitude: {data['contact_force_magnitude'].mean():.3f} N")
print(f"   Max Magnitude:  {data['contact_force_magnitude'].max():.3f} N")
print(f"   Std Dev:        {data['contact_force_magnitude'].std():.3f} N")

print("\n2. Flange F/T Sensor Force:")
print(f"   Mean X: {data['flange_force_x'].mean():.3f} N, Std: {data['flange_force_x'].std():.3f} N")
print(f"   Mean Y: {data['flange_force_y'].mean():.3f} N, Std: {data['flange_force_y'].std():.3f} N")
print(f"   Mean Z: {data['flange_force_z'].mean():.3f} N, Std: {data['flange_force_z'].std():.3f} N")

print("\n3. Flange F/T Sensor Torque:")
print(f"   Mean X: {data['flange_torque_x'].mean():.3f} Nm, Std: {data['flange_torque_x'].std():.3f} Nm")
print(f"   Mean Y: {data['flange_torque_y'].mean():.3f} Nm, Std: {data['flange_torque_y'].std():.3f} Nm")
print(f"   Mean Z: {data['flange_torque_z'].mean():.3f} Nm, Std: {data['flange_torque_z'].std():.3f} Nm")

print("\n4. OSC Joint Torques (J1-J6):")
for i in range(6):
    col_name = f'osc_torque_j{i+1}'
    if col_name in data.columns:
        mean_val = data[col_name].mean()
        std_val = data[col_name].std()
        max_val = data[col_name].max()
        min_val = data[col_name].min()
        print(f"   Joint {i+1}: Mean={mean_val:8.2f} Nm, Std={std_val:7.2f} Nm, Range=[{min_val:8.2f}, {max_val:8.2f}] Nm")

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
print("\nGenerated files:")
print("  - hdr35_plot1_contact_force_components.png")
print("  - hdr35_plot2_flange_force.png")
print("  - hdr35_plot3_flange_torque.png")
print("  - hdr35_plot4_joint2_torque.png")
print("  - hdr35_plot5_joint4_torque.png")
print("  - hdr35_plot6_joint6_torque.png")
print("  - hdr35_plot7_osc_joint_torques.png")
print("  - hdr35_plot8_contact_vs_flange_force.png")
if available_links:
    print("  - hdr35_plot9_all_link_forces_z.png")
print("\nAnalysis complete!")
