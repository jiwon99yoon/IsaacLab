# 👨‍💻 analyze_log.py (Modified for txt file)

import pandas as pd
import matplotlib.pyplot as plt

# --- File Loading ---
# Set the path to your txt file.
file_path = 'advanced_hw2/logging_hw2_1.txt'

# Read the txt file into a Pandas DataFrame.
# The file is space-separated, so we use sep='\s+' (whitespace separator)
try:
    data = pd.read_csv(file_path, sep='\s+')
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

# Figure 1: Target Position and Current Position over Time
plt.figure(figsize=(12, 6))
plt.plot(data['playtime'], data['target_pos(x)'], color='r', label='Target X', linestyle='-')
plt.plot(data['playtime'], data['target_pos(y)'], color='g', label='Target Y', linestyle='-')
plt.plot(data['playtime'], data['target_pos(z)'], color='b', label='Target Z', linestyle='-')
plt.plot(data['playtime'], data['current_pos(x)'], color='r', label='Current X', linestyle='--')
plt.plot(data['playtime'], data['current_pos(y)'], color='g', label='Current Y', linestyle='--')
plt.plot(data['playtime'], data['current_pos(z)'], color='b', label='Current Z', linestyle='--')

plt.title('HW2_1: Target vs Current Position of end effector over Time')
plt.xlabel('Playtime (s)')
plt.ylabel('Position (m)')
plt.legend()
plt.grid(True)
plt.show()


# Figure 2: Desired Joint Angles (q_desired_1 to q_desired_7) over Time
plt.figure(figsize=(14, 7))
plt.plot(data['playtime'], data['q_desired_1'], label='q_desired_1')
plt.plot(data['playtime'], data['q_desired_2'], label='q_desired_2')
plt.plot(data['playtime'], data['q_desired_3'], label='q_desired_3')
plt.plot(data['playtime'], data['q_desired_4'], label='q_desired_4')
plt.plot(data['playtime'], data['q_desired_5'], label='q_desired_5')
plt.plot(data['playtime'], data['q_desired_6'], label='q_desired_6')
plt.plot(data['playtime'], data['q_desired_7'], label='q_desired_7')

plt.title('HW2_1: Desired Joint Angles over Time')
plt.xlabel('Playtime (s)')
plt.ylabel('Joint Angle (rad)')
plt.legend()
plt.grid(True)
plt.show()