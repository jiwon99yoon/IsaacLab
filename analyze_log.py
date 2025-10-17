# 👨‍💻 analyze_log.py (English Version)

import pandas as pd
import matplotlib.pyplot as plt

# --- File Loading ---
# Set the path to your CSV file.
file_path = 'simulation_log_enhanced.csv'

# Read the CSV file into a Pandas DataFrame.
try:
    data = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found. Please check the filename and path.")
    exit()

# --- Data Verification ---
# Print the first 5 rows and column names to verify.
print("--- Data Head (First 5 Rows) ---")
print(data.head())
print("\n--- Available Columns ---")
print(data.columns.tolist())


# --- Graph Generation ---

# Figure 1: Contact Force Magnitude over Time
plt.figure(figsize=(12, 6))
plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
plt.plot(data['step'], data['fs_force_x'], color='y', label='fs_force_x')
plt.plot(data['step'], data['fs_force_y'], color='g', label='fs_force_y')
plt.plot(data['step'], data['fs_force_z'], color='b', label='fs_force_z', linestyle='--')

plt.title('Figure 1: Contact Force Magnitude / FT sensor force over Simulation Steps')
plt.xlabel('Step')
plt.ylabel('Force (N) / Torque (Nm)')
plt.legend()
plt.grid(True)
plt.show()

#################################################################################################
# Figure 2: Contact Force Magnitude over Time
plt.figure(figsize=(12, 6))
#plt.plot(data['step'], data['contact_force_magnitude'], color='y', label='Contact Force', linestyle='--')
plt.plot(data['step'], data['fs_torque_x'], color='c', label='fs_torque_x')
plt.plot(data['step'], data['fs_torque_y'], color='m', label='fs_torque_y')
plt.plot(data['step'], data['fs_torque_z'], color='k', label='fs_torque_z')

plt.title('Figure 2: Contact Force Magnitude / FT sensor torque over Simulation Steps')
plt.xlabel('Step')
plt.ylabel('Force (N) / Torque (Nm)')
plt.legend()
plt.grid(True)
plt.show()


##########################################################################################################
# Figure 3: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
plt.figure(figsize=(12, 6))
# Plot both torque types on the same graph for comparison.
plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
plt.plot(data['step'], data['joint2_force_x'], color='y', label='joint2_force_x')
plt.plot(data['step'], data['joint2_force_y'], color='g', label='joint2_force_y')
plt.plot(data['step'], data['joint2_force_z'], color='b', label='joint2_force_z')

plt.title('Figure 3: Joint 2 force sensor data')
plt.xlabel('Step')
plt.ylabel('Force (N)')
plt.legend() # Display labels to identify each line
plt.grid(True)
plt.show()

##########################################################################################################
# Figure 4: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
plt.figure(figsize=(12, 6))
# Plot both torque types on the same graph for comparison.
plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
plt.plot(data['step'], data['joint2_torque_x'], color='c', label='joint2_torque_x')
plt.plot(data['step'], data['joint2_torque_y'], color='m', label='joint2_torque_y')
plt.plot(data['step'], data['joint2_torque_z'], color='k', label='joint2_torque_z')

plt.title('Figure 3: Joint 2 torque sensor data')
plt.xlabel('Step')
plt.ylabel('Torque (Nm)')
plt.legend() # Display labels to identify each line
plt.grid(True)
plt.show()

##########################################################################################################
# Figure 3: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
plt.figure(figsize=(12, 6))
# Plot both torque types on the same graph for comparison.
plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
plt.plot(data['step'], data['joint4_force_x'], color='y', label='joint4_force_x')
plt.plot(data['step'], data['joint4_force_y'], color='g', label='joint4_force_y')
plt.plot(data['step'], data['joint4_force_z'], color='b', label='joint4_force_z')

plt.title('Figure 3: Joint 4 force sensor data')
plt.xlabel('Step')
plt.ylabel('Force (N)')
plt.legend() # Display labels to identify each line
plt.grid(True)
plt.show()

##########################################################################################################
# Figure 4: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
plt.figure(figsize=(12, 6))
# Plot both torque types on the same graph for comparison.
plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
plt.plot(data['step'], data['joint4_torque_x'], color='c', label='joint4_torque_x')
plt.plot(data['step'], data['joint4_torque_y'], color='m', label='joint4_torque_y')
plt.plot(data['step'], data['joint4_torque_z'], color='k', label='joint4_torque_z')

plt.title('Figure 3: Joint 2 torque sensor data')
plt.xlabel('Step')
plt.ylabel('Torque (Nm)')
plt.legend() # Display labels to identify each line
plt.grid(True)
plt.show()


# ##########################################################################################################
# # Figure 4: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j2'], linestyle='-', label='OSC Calculated Torque (j2)')
# plt.plot(data['step'], data['ft_torque_j2'], linestyle='--', label='F/T Sensor Measured Torque (j2)')
# plt.plot(data['step'], data['diff_torque_j2'], color='y', label='(j2) difference')

# plt.title('Figure 3: Joint 2 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()


# ##########################################################################################################
# # Figure 5: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j3'], linestyle='-', label='OSC Calculated Torque (j3)')
# plt.plot(data['step'], data['ft_torque_j3'], linestyle='--', label='F/T Sensor Measured Torque (j3)')
# plt.plot(data['step'], data['diff_torque_j3'], color='y', label='(j3) difference')

# plt.title('Figure 3: Joint 3 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()


# ##########################################################################################################
# # Figure 6: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j4'], linestyle='-', label='OSC Calculated Torque (j4)')
# plt.plot(data['step'], data['ft_torque_j4'], linestyle='--', label='F/T Sensor Measured Torque (j4)')
# plt.plot(data['step'], data['diff_torque_j4'], color='y', label='(j4) difference')

# plt.title('Figure 3: Joint 4 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()

# ##########################################################################################################
# # Figure 2: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j5'], linestyle='-', label='OSC Calculated Torque (j5)')
# plt.plot(data['step'], data['ft_torque_j5'], linestyle='--', label='F/T Sensor Measured Torque (j5)')
# plt.plot(data['step'], data['diff_torque_j5'], color='y', label='(j5) difference')

# plt.title('Figure 3: Joint 5 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()


# ##########################################################################################################
# # Figure 2: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j6'], linestyle='-', label='OSC Calculated Torque (j6)')
# plt.plot(data['step'], data['ft_torque_j6'], linestyle='--', label='F/T Sensor Measured Torque (j6)')
# plt.plot(data['step'], data['diff_torque_j6'], color='y', label='(j6) difference')

# plt.title('Figure 3: Joint 6 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()

# ##########################################################################################################
# # Figure 2: Joint 1 Torque Comparison (OSC vs. F/T Sensor)
# plt.figure(figsize=(12, 6))
# # Plot both torque types on the same graph for comparison.
# plt.plot(data['step'], data['contact_force_magnitude'], color='r', label='Contact Force', linestyle=':')
# plt.plot(data['step'], data['osc_torque_j7'], linestyle='-', label='OSC Calculated Torque (j7)')
# plt.plot(data['step'], data['ft_torque_j7'], linestyle='--', label='F/T Sensor Measured Torque (j7)')
# plt.plot(data['step'], data['diff_torque_j7'], color='y', label='(j7) difference')

# plt.title('Figure 3: Joint 7 Torque Comparison - OSC vs. F/T Sensor')
# plt.xlabel('Step')
# plt.ylabel('Torque (Nm)')
# plt.legend() # Display labels to identify each line
# plt.grid(True)
# plt.show()
