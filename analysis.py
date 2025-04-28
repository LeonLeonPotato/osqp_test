import re
import matplotlib.pyplot as plt
import numpy as np

# Your raw data as a multi-line string
data = open('data.txt', 'r').read()

# Variable to choose which Target value to plot: 0 for first (e.g., 29.28), 1 for second (e.g., 16.50)
which_value = 0  # <-- Change to 1 to plot the second value

# Lists to hold extracted data
target_values = []
current_speeds = []

# Regex pattern to find Target and Current speeds
pattern = re.compile(r"Target: \[.*\| ([\-\d\.]+), ([\-\d\.]+)]\s+Curernt speeds: ([\-\d\.]+), ([\-\d\.]+)")

# Parse the data
for match in pattern.finditer(data):
    target_val1 = float(match.group(1))
    target_val2 = float(match.group(2))
    speed1 = float(match.group(3))
    speed2 = float(match.group(4))

    target_values.append((target_val1, target_val2))
    current_speeds.append((speed1, speed2))

# Separate the selected Target values
selected_target_values = [t[which_value] for t in target_values]
selected_speed_values = [s[which_value] for s in current_speeds]

X = np.linspace(0, len(selected_target_values), len(selected_target_values))

# Plot
plt.figure()
plt.scatter(X, selected_speed_values, label='Current Speed')
plt.scatter(X, selected_target_values, label='Desired speed')
plt.xlabel('Target Value ({})'.format('First' if which_value == 0 else 'Second'))
plt.ylabel('Current Speeds')
plt.title('Target Value vs Current Speeds')
plt.legend()
plt.grid(True)
plt.show()
