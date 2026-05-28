import matplotlib.pyplot as plt

filepath = 'experiment8_macro_dt_self_conv.txt'
dts, diffs = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'macro_dt' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2 and parts[1] != 'N/A':
                dts.append(float(parts[0]))
                diffs.append(float(parts[1]))

plt.figure(figsize=(9, 6))
plt.loglog(dts, diffs, 'o-', color='darkgreen')
plt.plot(dts[-1], diffs[-1], 'ro', markersize=12, label='здесь багулина')

plt.gca().invert_xaxis()
plt.xlabel('Макро-шаг (macro_dt)')
plt.ylabel('Разность с предыдущим шагом')
plt.title('Exp 8: Само-сходимость Piecewise Магнуса')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('plot_exp8.png', dpi=300)
plt.show()
