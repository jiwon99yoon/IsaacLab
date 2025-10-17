# # 👨‍💻 analyze_log_enhanced.py
# # Enhanced version for analyzing simulation_log_enhanced.csv

# import pandas as pd
# import matplotlib.pyplot as plt

# # --- File Loading ---
# file_path = 'simulation_log_enhanced.csv'

# try:
#     data = pd.read_csv(file_path)
#     print(f"✅ Successfully loaded: {file_path}")
#     print(f"   Total records: {len(data)}")
# except FileNotFoundError:
#     print(f"❌ Error: The file '{file_path}' was not found.")
#     print("   Please check the filename and path.")
#     exit()

# # --- Data Verification ---
# print("\n--- Data Head (First 5 Rows) ---")
# print(data.head())
# print("\n--- Available Columns ---")
# print(data.columns.tolist())
# print("\n" + "="*60 + "\n")

# # Set global plot style
# plt.style.use('default')
# plt.rcParams['figure.figsize'] = (14, 7)
# plt.rcParams['font.size'] = 10

# # --- Plot 1: Contact Force Components (X, Y, Z) + Magnitude ---
# print("Generating Plot 1: Contact Force Components...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['contact_force_x'], color='r', label='Contact Force X', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_y'], color='g', label='Contact Force Y', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_z'], color='b', label='Contact Force Z', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle='--', alpha=0.7)

# plt.title('Plot 1: Contact Force Components (X, Y, Z) and Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot1_contact_force_components.png', dpi=150)
# print("   ✅ Saved: plot1_contact_force_components.png")
# plt.show()

# # --- Plot 2: FT Sensor Force (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 2: FT Sensor Force (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_force_x_raw'], color='#FF6B6B', label='FS Force X (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_y_raw'], color='#4ECDC4', label='FS Force Y (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_z_raw'], color='#45B7D1', label='FS Force Z (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 2: FT Sensor Force (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot2_ft_force_raw.png', dpi=150)
# print("   ✅ Saved: plot2_ft_force_raw.png")
# plt.show()

# # --- Plot 3: FT Sensor Force (FILTERED) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 3: FT Sensor Force (Filtered)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_force_x'], color='#FF6B6B', label='FS Force X (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_y'], color='#4ECDC4', label='FS Force Y (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_z'], color='#45B7D1', label='FS Force Z (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 3: FT Sensor Force (Filtered, Outliers Removed) vs Contact Force', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot3_ft_force_filtered.png', dpi=150)
# print("   ✅ Saved: plot3_ft_force_filtered.png")
# plt.show()

# # --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 4: FT Sensor Torque (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_torque_x_raw'], color='#C44569', label='FS Torque X (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_y_raw'], color='#F8B500', label='FS Torque Y (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_z_raw'], color='#A29BFE', label='FS Torque Z (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 4: FT Sensor Torque (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot4_ft_torque_raw.png', dpi=150)
# print("   ✅ Saved: plot4_ft_torque_raw.png")
# plt.show()

# # --- Plot 5: FT Sensor Torque (FILTERED) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 5: FT Sensor Torque (Filtered)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_torque_x'], color='#C44569', label='FS Torque X (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_y'], color='#F8B500', label='FS Torque Y (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_z'], color='#A29BFE', label='FS Torque Z (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 5: FT Sensor Torque (Filtered, Outliers Removed) vs Contact Force', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot5_ft_torque_filtered.png', dpi=150)
# print("   ✅ Saved: plot5_ft_torque_filtered.png")
# plt.show()

# # --- Plot 6: OSC Joint Torques (J1-J7) + Contact Force Magnitude ---
# print("Generating Plot 6: OSC Joint Torques (All Joints)...")
# plt.figure(figsize=(14, 7))

# # Define colors for each joint
# colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
# joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

# for i in range(7):
#     plt.plot(data['step'], data[f'osc_torque_j{i+1}'], 
#              color=colors[i], label=f'OSC Torque {joint_labels[i]}', 
#              linewidth=1.2, alpha=0.8)

# # Add contact force magnitude for reference
# plt.plot(data['step'], data['contact_force_magnitude'], 
#          color='k', label='Contact Force Magnitude', 
#          linewidth=2.5, linestyle=':', alpha=0.9)

# plt.title('Plot 6: OSC Joint Torques (J1-J7) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=9, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot6_osc_joint_torques_all.png', dpi=150)
# print("   ✅ Saved: plot6_osc_joint_torques_all.png")
# plt.show()

# # --- Summary Statistics ---
# print("\n" + "="*60)
# print("📊 Summary Statistics")
# print("="*60)

# print("\n1. Contact Force:")
# print(f"   Mean Magnitude: {data['contact_force_magnitude'].mean():.3f} N")
# print(f"   Max Magnitude:  {data['contact_force_magnitude'].max():.3f} N")
# print(f"   Std Dev:        {data['contact_force_magnitude'].std():.3f} N")

# print("\n2. FT Sensor Force (Filtered):")
# print(f"   Mean X: {data['fs_force_x'].mean():.3f} N, Std: {data['fs_force_x'].std():.3f} N")
# print(f"   Mean Y: {data['fs_force_y'].mean():.3f} N, Std: {data['fs_force_y'].std():.3f} N")
# print(f"   Mean Z: {data['fs_force_z'].mean():.3f} N, Std: {data['fs_force_z'].std():.3f} N")

# print("\n3. FT Sensor Torque (Filtered):")
# print(f"   Mean X: {data['fs_torque_x'].mean():.3f} Nm, Std: {data['fs_torque_x'].std():.3f} Nm")
# print(f"   Mean Y: {data['fs_torque_y'].mean():.3f} Nm, Std: {data['fs_torque_y'].std():.3f} Nm")
# print(f"   Mean Z: {data['fs_torque_z'].mean():.3f} Nm, Std: {data['fs_torque_z'].std():.3f} Nm")

# print("\n4. OSC Joint Torques:")
# for i in range(7):
#     mean_val = data[f'osc_torque_j{i+1}'].mean()
#     std_val = data[f'osc_torque_j{i+1}'].std()
#     max_val = data[f'osc_torque_j{i+1}'].max()
#     min_val = data[f'osc_torque_j{i+1}'].min()
#     print(f"   Joint {i+1}: Mean={mean_val:6.2f} Nm, Std={std_val:5.2f} Nm, Range=[{min_val:6.2f}, {max_val:6.2f}] Nm")

# print("\n5. Outlier Statistics:")
# outlier_force = ((data['fs_force_x_raw'].abs() > 15).sum() + 
#                  (data['fs_force_y_raw'].abs() > 15).sum() + 
#                  (data['fs_force_z_raw'].abs() > 15).sum())
# outlier_torque = ((data['fs_torque_x_raw'].abs() > 0.4).sum() + 
#                   (data['fs_torque_y_raw'].abs() > 0.4).sum() + 
#                   (data['fs_torque_z_raw'].abs() > 0.4).sum())
# print(f"   Force outliers removed (>15N):   {outlier_force}")
# print(f"   Torque outliers removed (>0.4Nm): {outlier_torque}")
# print(f"   Total data points per axis:       {len(data)}")
# print(f"   Outlier rate (Force):             {outlier_force/(len(data)*3)*100:.2f}%")
# print(f"   Outlier rate (Torque):            {outlier_torque/(len(data)*3)*100:.2f}%")

# print("\n" + "="*60)
# print("✅ All plots generated successfully!")
# print("="*60)
# print("\nGenerated files:")
# print("  - plot1_contact_force_components.png")
# print("  - plot2_ft_force_raw.png")
# print("  - plot3_ft_force_filtered.png")
# print("  - plot4_ft_torque_raw.png")
# print("  - plot5_ft_torque_filtered.png")
# print("  - plot6_osc_joint_torques_all.png")
# print("\n🎉 Analysis complete!")

# 👨‍💻 analyze_log_enhanced.py
# Enhanced version for analyzing simulation_log_enhanced.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- File Loading ---
# file_path = 'simulation_log_enhanced_copy.csv'
#file_path = 'simulation_log_enhanced_357.csv'
file_path = 'simulation_log_enhanced.csv'
# file_path = 'simulation_log_enhanced_new.csv'
try:
    data = pd.read_csv(file_path)
    print(f"✅ Successfully loaded: {file_path}")
    print(f"   Total records: {len(data)}")
except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found.")
    print("   Please check the filename and path.")
    exit()

# --- Data Verification ---
print("\n--- Data Head (First 5 Rows) ---")
print(data.head())
print("\n--- Available Columns ---")
print(data.columns.tolist())
print("\n" + "="*60 + "\n")


#====================================================================================================================#
# Set global plot style
plt.style.use('default')
plt.rcParams['figure.figsize'] = (14, 7)
plt.rcParams['font.size'] = 10



# # --- Plot 1: Contact Force Components (X, Y, Z) + Magnitude ---
# print("Generating Plot 1: Contact Force Components...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['contact_force_x'], color='r', label='Contact Force X', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_y'], color='g', label='Contact Force Y', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_z'], color='b', label='Contact Force Z', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle='--', alpha=0.7)

# plt.title('Plot 1: Contact Force Components (X, Y, Z) and Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot1_contact_force_components.png', dpi=150)
# print("   ✅ Saved: plot1_contact_force_components.png")
# plt.show()

# #====================================================================================================================#

# # --- Plot 2: FT Sensor Force (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 1: Link2 force (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[link2_force_x_raw]'], color='#45B7D1', label='link2 force X(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['link_data[link2_force_y_raw]'], color='#FF6B6B', label='link2 force Y(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['link_data[link2_force_z_raw]'], color='#4ECDC4', label='link2 force Z(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 2: Link2 force (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot2_ft_force_raw.png', dpi=150)
# # print("   ✅ Saved: plot2_ft_force_raw.png")
# plt.show()

# # --- Plot 2: FT Sensor Force (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 1: Link4 force (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[link4_force_x_raw]'], color='#45B7D1', label='link4 force X(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['link_data[link4_force_y_raw]'], color='#FF6B6B', label='link4 force Y(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['link_data[link4_force_z_raw]'], color='#4ECDC4', label='link4 force Z(Raw)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)


# plt.title('Plot 2: Link4 force (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot2_ft_force_raw.png', dpi=150)
# # print("   ✅ Saved: plot2_ft_force_raw.png")
# plt.show()

# #====================================================================================================================#

# # --- Plot 1: Contact Force Components (X, Y, Z) + Magnitude ---
# print("Generating Plot 1: Contact Force Components...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['contact_force_x'], color='r', label='Contact Force X', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_y'], color='g', label='Contact Force Y', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_z'], color='b', label='Contact Force Z', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle='--', alpha=0.7)

# plt.title('Plot 1: Contact Force Components (X, Y, Z) and Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot1_contact_force_components.png', dpi=150)
# # print("   ✅ Saved: plot1_contact_force_components.png")
# plt.show()


# # #====================================================================================================================#

# # --- Plot 2: FT Sensor Force (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 2: FT Sensor Force (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_force_x_raw'], color='#FF6B6B', label='FS Force X (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_y_raw'], color="#5EAE1D", label='FS Force Y (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_z_raw'], color='#45B7D1', label='FS Force Z (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 2: FT Sensor Force (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# # plt.savefig('plot2_ft_force_raw.png', dpi=150)
# # print("   ✅ Saved: plot2_ft_force_raw.png")
# plt.show()


# # #====================================================================================================================#

# # --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 3: FT Sensor Torque (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_torque_x_raw'], color='#C44569', label='FS Torque X (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_y_raw'], color='#F8B500', label='FS Torque Y (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_z_raw'], color='#A29BFE', label='FS Torque Z (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['contact_force_magnitude']*0.01, color='k', label='Contact Force Magnitude * 0.01', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 3: FT Sensor Torque (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# # plt.savefig('plot4_ft_torque_raw.png', dpi=150)
# # print("   ✅ Saved: plot4_ft_torque_raw.png")
# plt.show()

# #====================================================================================================================#

# --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
print("Generating Plot 4: joint2 torque (RAW)...")
plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[joint2_torque_x_raw]'], color='#C44569', label='joint2 Torque X (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_y_raw]'], color='#F8B500', label='joint2 Torque Y (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_z_raw]'], color='#A29BFE', label='joint2 Torque Z (Raw)', linewidth=1.5, linestyle='-')

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
         linewidth=2, linestyle=':', alpha=0.7)

plt.plot(data['step'], data['joint2_commanded(applied)'], color='#C44569', label='joint2_commanded(applied)', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint2_external'], color='#F8B500', label='joint2_external', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint2_external_g'], color="#B41FBE", label='joint2_external_g', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint2_cot_tor_sim'], color="#B41FBE", label='joint2_cot_tor_sim', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint2_cot_tor'], color='b', label='joint2_cot_tor', linewidth=1.5, linestyle='--', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j2'], color="#28D04D", label='OSC Torque 2', linewidth=1.2, alpha=0.8)

# plt.plot(data['step'], data['joint2_acc'], color='#A29BFE', label='joint2_acc', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['joint_vel_j2'], color="#6C456E", label='joint_vel_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['joint_pos_j2'], color="#B41FBE", label='joint_pos_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['osc_torque_j2']- data['act_tor_j2'], color="#28D04D", label='error torque 2', linewidth=1.2, alpha=0.8)

plt.title('Plot 4: joint2 Torque comparison (commanded, external, actual) (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('plot4_ft_torque_raw.png', dpi=150)
#print("   ✅ Saved: plot4_ft_torque_raw.png")
plt.show()

# #====================================================================================================================#

# --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
print("Generating Plot 5: joint4 torque (RAW)...")
plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[joint2_torque_x_raw]'], color='#C44569', label='joint2 Torque X (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_y_raw]'], color='#F8B500', label='joint2 Torque Y (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_z_raw]'], color='#A29BFE', label='joint2 Torque Z (Raw)', linewidth=1.5, linestyle='-')

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
         linewidth=2, linestyle=':', alpha=0.7)

plt.plot(data['step'], data['joint4_commanded(applied)'], color='#C44569', label='joint4_commanded(applied)', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint4_external'], color='#F8B500', label='joint4_external', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint4_external_g'], color="#B41FBE", label='joint4_external_g', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint4_cot_tor_sim'], color="#B41FBE", label='joint4_cot_tor_sim', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint4_cot_tor'], color='b', label='joint4_cot_tor', linewidth=1.5, linestyle='--', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j4'], color="#28D04D", label='OSC Torque 4', linewidth=1.2, alpha=0.8)

# plt.plot(data['step'], data['joint2_acc'], color='#A29BFE', label='joint2_acc', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['joint_vel_j2'], color="#6C456E", label='joint_vel_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['joint_pos_j2'], color="#B41FBE", label='joint_pos_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['osc_torque_j2']- data['act_tor_j2'], color="#28D04D", label='error torque 2', linewidth=1.2, alpha=0.8)

plt.title('Plot 5: joint4 Torque comparison (commanded, external, actual) (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('plot4_ft_torque_raw.png', dpi=150)
#print("   ✅ Saved: plot4_ft_torque_raw.png")
plt.show()

# --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
print("Generating Plot 6: joint6 torque (RAW)...")
plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[joint2_torque_x_raw]'], color='#C44569', label='joint2 Torque X (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_y_raw]'], color='#F8B500', label='joint2 Torque Y (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[joint2_torque_z_raw]'], color='#A29BFE', label='joint2 Torque Z (Raw)', linewidth=1.5, linestyle='-')

plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
         linewidth=2, linestyle=':', alpha=0.7)

plt.plot(data['step'], data['joint6_commanded(applied)'], color='#C44569', label='joint6_commanded(applied)', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint6_external'], color='#F8B500', label='joint6_external', linewidth=1.5, linestyle='-')
plt.plot(data['step'], data['joint6_external_g'], color="#B41FBE", label='joint6_external_g', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint6_cot_tor_sim'], color="#B41FBE", label='joint6_cot_tor_sim', linewidth=1.5, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['joint6_cot_tor'], color='b', label='joint6_cot_tor', linewidth=1.5, linestyle='--', alpha=0.7)
plt.plot(data['step'], data['osc_torque_j6'], color="#28D04D", label='OSC Torque 6', linewidth=1.2, alpha=0.8)

# plt.plot(data['step'], data['joint2_acc'], color='#A29BFE', label='joint2_acc', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['joint_vel_j2'], color="#6C456E", label='joint_vel_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['joint_pos_j2'], color="#B41FBE", label='joint_pos_j2', linewidth=1.5, linestyle='--', alpha=0.7)
# plt.plot(data['step'], data['osc_torque_j2']- data['act_tor_j2'], color="#28D04D", label='error torque 2', linewidth=1.2, alpha=0.8)

plt.title('Plot 6: joint6 Torque comparison (commanded, external, actual) (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
plt.xlabel('Step', fontsize=12)
plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)
plt.tight_layout()
#plt.savefig('plot4_ft_torque_raw.png', dpi=150)
#print("   ✅ Saved: plot4_ft_torque_raw.png")
plt.show()


# #====================================================================================================================#

# # --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 2: LInk2 torque (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[link2_torque_x_raw]'], color='#C44569', label='link2 Torque X (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[link2_torque_y_raw]'], color='#F8B500', label='link2 Torque Y (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[link2_torque_z_raw]'], color='#A29BFE', label='link2 Torque Z (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude * 0.01', 
#          linewidth=2, linestyle=':', alpha=0.7)
# plt.plot(data['step'], data['osc_torque_j3'], color='#2ECC71', label='OSC Torque 2', linewidth=1.2, alpha=0.8)
# plt.title('Plot 2: Link2 Torque (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot4_ft_torque_raw.png', dpi=150)
# #print("   ✅ Saved: plot4_ft_torque_raw.png")
# plt.show()

# # #====================================================================================================================#

# # --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 2: LInk4 torque (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['link_data[link4_torque_x_raw]'], color='#C44569', label='link4 Torque X (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[link4_torque_y_raw]'], color='#F8B500', label='link4 Torque Y (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['link_data[link4_torque_z_raw]'], color='#A29BFE', label='link4 Torque Z (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude * 0.01', 
#          linewidth=2, linestyle=':', alpha=0.7)
# # colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
# # joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']
# plt.plot(data['step'], data['osc_torque_j5'], color='#2ECC71', label='OSC Torque 4', linewidth=1.2, alpha=0.8)

# plt.title('Plot 2: Link4 Torque (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# #plt.savefig('plot4_ft_torque_raw.png', dpi=150)
# #print("   ✅ Saved: plot4_ft_torque_raw.png")
# plt.show()

# # --- Plot 3: FT Sensor Force (FILTERED) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 3: FT Sensor Force (Filtered)...")
# plt.figure(figsize=(14, 7))

# # plt.plot(data['step'], data['fs_force_x'], color='#FF6B6B', label='FS Force X (Filtered)', linewidth=1.5)
# # plt.plot(data['step'], data['fs_force_y'], color='#4ECDC4', label='FS Force Y (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_force_z'], color='#45B7D1', label='FS Force Z (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['contact_force_magnitude'], color='k', label='Contact Force Magnitude', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 3: FT Sensor Force (Filtered, Outliers Removed) vs Contact Force', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot3_ft_force_filtered.png', dpi=150)
# print("   ✅ Saved: plot3_ft_force_filtered.png")
# plt.show()

# #====================================================================================================================#

# # --- Plot 4: FT Sensor Torque (RAW) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 4: FT Sensor Torque (RAW)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_torque_x_raw'], color='#C44569', label='FS Torque X (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_y_raw'], color='#F8B500', label='FS Torque Y (Raw)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_z_raw'], color='#A29BFE', label='FS Torque Z (Raw)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['contact_force_magnitude']*0.01, color='k', label='Contact Force Magnitude * 0.01', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 4: FT Sensor Torque (RAW) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot4_ft_torque_raw.png', dpi=150)
# print("   ✅ Saved: plot4_ft_torque_raw.png")
# plt.show()

# #====================================================================================================================#

# # --- Plot 5: FT Sensor Torque (FILTERED) - X, Y, Z + Contact Force Magnitude ---
# print("Generating Plot 5: FT Sensor Torque (Filtered)...")
# plt.figure(figsize=(14, 7))

# plt.plot(data['step'], data['fs_torque_x'], color='#C44569', label='FS Torque X (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_y'], color='#F8B500', label='FS Torque Y (Filtered)', linewidth=1.5)
# plt.plot(data['step'], data['fs_torque_z'], color='#A29BFE', label='FS Torque Z (Filtered)', linewidth=1.5, linestyle='-')
# plt.plot(data['step'], data['contact_force_magnitude']*0.01, color='k', label='Contact Force Magnitude * 0.01', 
#          linewidth=2, linestyle=':', alpha=0.7)

# plt.title('Plot 5: FT Sensor Torque (Filtered, Outliers Removed) vs Contact Force', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot5_ft_torque_filtered.png', dpi=150)
# print("   ✅ Saved: plot5_ft_torque_filtered.png")
# plt.show()

# #====================================================================================================================#

# # --- Plot 6: Contact Force vs FT Sensor Force Component Comparison ---
# print("Generating Plot 6: Contact Force vs FT Sensor Force (Component-wise)...")
# plt.figure(figsize=(14, 7))

# # Contact Force (점선, 진한 색상)
# plt.plot(data['step'], data['contact_force_x'], 
#          color='#E74C3C', label='Contact Force X', 
#          linewidth=2, linestyle='--', alpha=0.9)
# plt.plot(data['step'], data['contact_force_y'], 
#          color='#27AE60', label='Contact Force Y', 
#          linewidth=2, linestyle='--', alpha=0.9)
# plt.plot(data['step'], data['contact_force_z'], 
#          color='#3498DB', label='Contact Force Z', 
#          linewidth=2, linestyle='--', alpha=0.9)

# # FT Sensor Force RAW (실선, 대비되는 색상)
# plt.plot(data['step'], data['fs_force_x_raw'], 
#          color='#FF69B4', label='FS Force X (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_force_y_raw'], 
#          color='#32CD32', label='FS Force Y (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_force_z_raw'], 
#          color='#FFA500', label='FS Force Z (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)

# plt.title('Plot 6: Contact Force vs FT Sensor Force (Component-wise Comparison)', 
#           fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot6_contact_vs_fs_force_comparison.png', dpi=150)
# print("   ✅ Saved: plot6_contact_vs_fs_force_comparison.png")
# plt.show()


# # --- Plot 6: Contact Force vs FT Sensor Force Component Comparison ---
# print("Generating Plot 6+: Contact Force * 0.02 vs FT Sensor torque (Component-wise)...")
# plt.figure(figsize=(14, 7))

# # Contact Force (점선, 진한 색상)
# plt.plot(data['step'], data['contact_force_x']*0.02, 
#          color='#E74C3C', label='Contact Force X', 
#          linewidth=2, linestyle='--', alpha=0.9)
# plt.plot(data['step'], data['contact_force_y']*0.02, 
#          color='#27AE60', label='Contact Force Y', 
#          linewidth=2, linestyle='--', alpha=0.9)
# plt.plot(data['step'], data['contact_force_z']*0.02, 
#          color='#3498DB', label='Contact Force Z', 
#          linewidth=2, linestyle='--', alpha=0.9)

# # FT Sensor Force RAW (실선, 대비되는 색상)
# plt.plot(data['step'], data['fs_torque_x_raw'], 
#          color='#FF69B4', label='FS Torque X (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_torque_y_raw'], 
#          color='#32CD32', label='FS Torque Y (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_torque_z_raw'], 
#          color='#FFA500', label='FS Torque Z (Raw)', 
#          linewidth=1.5, linestyle='dashdot', alpha=0.8)

# plt.title('Plot 6: Contact Force vs FT Sensor Force (Component-wise Comparison)', 
#           fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot6_contact_force *0.02 vs fs_torque_comparison.png', dpi=150)
# print("   ✅ Saved: plot6_contact_vs_fs_force_comparison.png")
# plt.show()


#====================================================================================================================#

# --- Plot 7: OSC Joint Torques (J1-J7) + Contact Force Magnitude ---
# print("Generating Plot 7: OSC Joint Torques (All Joints)...")
# plt.figure(figsize=(14, 7))

# Define colors for each joint
# colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6', '#1ABC9C', '#E67E22']
# joint_labels = ['J1', 'J2', 'J3', 'J4', 'J5', 'J6', 'J7']

# for i in range(7):
#     if i == 3:
#         continue
#     else :
#         plt.plot(data['step'], data[f'osc_torque_j{i+1}'], 
#              color=colors[i], label=f'OSC Torque {joint_labels[i]}', 
#              linewidth=1.2, alpha=0.8)

# for i in range(7):
#     if i in [1, 3]:
#         plt.plot(data['step'], data[f'osc_torque_j{i+1}'], 
#              color=colors[i], label=f'OSC Torque {joint_labels[i]}', 
#              linewidth=1.2, alpha=0.8)
#     else :
#         continue

# Add contact force magnitude for reference
# plt.plot(data['step'], data['contact_force_magnitude'], 
#          color='k', label='Contact Force Magnitude', 
#          linewidth=2.5, linestyle=':', alpha=0.9)

# plt.title('Plot 7: OSC Joint Torques (J1-J7) vs Contact Force Magnitude', fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Torque (Nm) / Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=9, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot7_osc_joint_torques_2_4.png', dpi=150)
# print("   ✅ Saved: plot7_osc_joint_torques_all.png")
# plt.show()

#====================================================================================================================#

# # --- Plot 6: Contact Force vs FT Sensor Force Component Comparison ---
# print("Generating Plot 8: Contact Force * 0.02 vs FT Sensor torque with joint 7 torque (Component-wise)...")
# plt.figure(figsize=(14, 7))

# # Contact Force (점선, 진한 색상)
# plt.plot(data['step'], data['contact_force_x'], 
#          color='#E74C3C', label='Contact Force X', 
#          linewidth=2, linestyle='--', alpha=0.9)
# # plt.plot(data['step'], data['contact_force_y']*0.02, 
# #          color='#27AE60', label='Contact Force Y', 
# #          linewidth=2, linestyle='--', alpha=0.9)
# # plt.plot(data['step'], data['contact_force_z']*0.02, 
# #          color='#3498DB', label='Contact Force Z', 
# #          linewidth=2, linestyle='--', alpha=0.9)

# # FT Sensor Force RAW (실선, 대비되는 색상)
# plt.plot(data['step'], data['fs_torque_x_raw'], 
#          color='#FF69B4', label='FS Torque X (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_torque_y_raw'], 
#          color='#32CD32', label='FS Torque Y (Raw)', 
#          linewidth=1.5, linestyle='-', alpha=0.8)
# plt.plot(data['step'], data['fs_torque_z_raw'], 
#          color='#FFA500', label='FS Torque Z (Raw)', 
#          linewidth=1.5, linestyle='dashdot', alpha=0.8)

# plt.plot(data['step'], data[f'osc_torque_j{6}'], 
#              color=colors[6], label=f'OSC Torque {joint_labels[6]}', 
#              linewidth=1.2, alpha=0.8, linestyle='dashdot')

# ft_torque_magnitude = np.sqrt(
#     data['fs_torque_x_raw']**2 + 
#     data['fs_torque_y_raw']**2 + 
#     data['fs_torque_z_raw']**2
# )

# plt.plot(data['step'], ft_torque_magnitude, label='FT Torque Magnitude')

# plt.title('Plot 8: Contact Force vs FT Sensor Force with joint7 torque(Component-wise Comparison)', 
#           fontsize=14, fontweight='bold')
# plt.xlabel('Step', fontsize=12)
# plt.ylabel('Force (N)', fontsize=12)
# plt.legend(loc='best', fontsize=10, ncol=2)
# plt.grid(True, alpha=0.3)
# plt.tight_layout()
# plt.savefig('plot8_contact_force *0.02 vs fs_torque_comparison with 7_joint torque .png', dpi=150)
# print("   ✅ Saved: plot6_contact_vs_fs_force_comparison.png")
# plt.show()

# # --- Plot 8: OSC Joint Torques J7 + ft sensor  ---
# print("Generating Plot 8: OSC Joint Torques J7 + ft sensor ")
# plt.figure(figsize=(14, 7))


#====================================================================================================================#

# --- Summary Statistics ---
# print("\n" + "="*60)
# print("📊 Summary Statistics")
# print("="*60)

# print("\n1. Contact Force:")
# print(f"   Mean Magnitude: {data['contact_force_magnitude'].mean():.3f} N")
# print(f"   Max Magnitude:  {data['contact_force_magnitude'].max():.3f} N")
# print(f"   Std Dev:        {data['contact_force_magnitude'].std():.3f} N")

# print("\n2. FT Sensor Force (Filtered):")
# print(f"   Mean X: {data['fs_force_x'].mean():.3f} N, Std: {data['fs_force_x'].std():.3f} N")
# print(f"   Mean Y: {data['fs_force_y'].mean():.3f} N, Std: {data['fs_force_y'].std():.3f} N")
# print(f"   Mean Z: {data['fs_force_z'].mean():.3f} N, Std: {data['fs_force_z'].std():.3f} N")

# print("\n3. FT Sensor Torque (Filtered):")
# print(f"   Mean X: {data['fs_torque_x'].mean():.3f} Nm, Std: {data['fs_torque_x'].std():.3f} Nm")
# print(f"   Mean Y: {data['fs_torque_y'].mean():.3f} Nm, Std: {data['fs_torque_y'].std():.3f} Nm")
# print(f"   Mean Z: {data['fs_torque_z'].mean():.3f} Nm, Std: {data['fs_torque_z'].std():.3f} Nm")

# print("\n4. OSC Joint Torques:")
# for i in range(7):
#     mean_val = data[f'osc_torque_j{i+1}'].mean()
#     std_val = data[f'osc_torque_j{i+1}'].std()
#     max_val = data[f'osc_torque_j{i+1}'].max()
#     min_val = data[f'osc_torque_j{i+1}'].min()
#     print(f"   Joint {i+1}: Mean={mean_val:6.2f} Nm, Std={std_val:5.2f} Nm, Range=[{min_val:6.2f}, {max_val:6.2f}] Nm")

# print("\n5. Outlier Statistics:")
# outlier_force = ((data['fs_force_x_raw'].abs() > 15).sum() + 
#                  (data['fs_force_y_raw'].abs() > 15).sum() + 
#                  (data['fs_force_z_raw'].abs() > 15).sum())
# outlier_torque = ((data['fs_torque_x_raw'].abs() > 0.4).sum() + 
#                   (data['fs_torque_y_raw'].abs() > 0.4).sum() + 
#                   (data['fs_torque_z_raw'].abs() > 0.4).sum())
# print(f"   Force outliers removed (>15N):   {outlier_force}")
# print(f"   Torque outliers removed (>0.4Nm): {outlier_torque}")
# print(f"   Total data points per axis:       {len(data)}")
# print(f"   Outlier rate (Force):             {outlier_force/(len(data)*3)*100:.2f}%")
# print(f"   Outlier rate (Torque):            {outlier_torque/(len(data)*3)*100:.2f}%")

# print("\n" + "="*60)
# print("✅ All plots generated successfully!")
# print("="*60)
# print("\nGenerated files:")
# print("  - plot1_contact_force_components.png")
# print("  - plot2_ft_force_raw.png")
# print("  - plot3_ft_force_filtered.png")
# print("  - plot4_ft_torque_raw.png")
# print("  - plot5_ft_torque_filtered.png")
# print("  - plot6_osc_joint_torques_all.png")
# print("\n🎉 Analysis complete!")