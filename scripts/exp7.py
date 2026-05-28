import matplotlib.pyplot as plt
import numpy as np

filepath = 'experiment7_rk4_convergence.txt'
dts, errs = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'dt' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 2:
                dts.append(float(parts[0]))
                errs.append(float(parts[1]))

plt.figure(figsize=(9, 6))
plt.loglog(dts, errs, 'o-', color='crimson', label='Фактическая ошибка РК4')

# Генерируем линию O(dt^4)
dts_np = np.array(dts)
ref_err = (dts_np ** 4) * (errs[2] / (dts_np[2]**4)) * 0.1 # Смещение для красоты
plt.loglog(dts_np, ref_err, 'k--', label='Теоретический наклон O(dt^4)')

plt.gca().invert_xaxis() # Ось X в обратном порядке (уменьшение шага)
plt.xlabel('Шаг интегрирования (dt)')
plt.ylabel('Ошибка')
plt.title('Exp 7: Сходимость метода РК4')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('plot_exp7.png', dpi=300)
plt.show()
