import pandas as pd
import matplotlib.pyplot as plt

def parse_omega_results(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('=') or line.startswith('-') or line.startswith('(') or line.startswith('Timing'):
                continue
            parts = line.split()
            if len(parts) == 5:
                method = parts[0]
                exp_type = parts[1]
                try:
                    omega = int(parts[2])
                    time_ms = float(parts[3])
                    error = float(parts[4])
                    data.append({
                        'method': method,
                        'exp_type': exp_type,
                        'omega': omega,
                        'time': time_ms,
                        'error': error
                    })
                except ValueError:
                    pass  # Skip invalid lines
    return pd.DataFrame(data)

# Parse the file
df = parse_omega_results('omega_results.txt')

# Sort by omega for each group
df = df.sort_values(['method', 'exp_type', 'omega'])

# Plot 1: Error vs Omega Number (log scale)
fig1, ax1 = plt.subplots(figsize=(10, 6))
for (method, exp_type), group in df.groupby(['method', 'exp_type']):
    ax1.plot(group['omega'], group['error'], label=f"{method} {exp_type}", marker='o')
ax1.set_yscale('log')
ax1.set_xlabel('Omega Number')
ax1.set_ylabel('Error (log scale)')
ax1.set_title('Error vs Omega Number')
ax1.legend()
ax1.grid(True)

# Plot 2: Time vs Omega Number (linear scale)
fig2, ax2 = plt.subplots(figsize=(10, 6))
for (method, exp_type), group in df.groupby(['method', 'exp_type']):
    ax2.plot(group['omega'], group['time'], label=f"{method} {exp_type}", marker='o')
ax2.set_xlabel('Omega Number')
ax2.set_ylabel('Time (ms)')
ax2.set_title('Time vs Omega Number')
ax2.legend()
ax2.grid(True)

# Show plots
plt.show()