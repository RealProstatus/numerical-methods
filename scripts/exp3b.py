import matplotlib.pyplot as plt

filepath = 'experiment3b_work_precision.txt'
rk_time, rk_err = [], []
mag_time, mag_err = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'Method' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 4:
                method, err, time = parts[0], float(parts[2]), float(parts[3])
                
                if method == 'RK4' and err < 1e5: # Убираем "взорвавшиеся" точки РК4
                    rk_time.append(time)
                    rk_err.append(err)
                elif method == 'Magnus':
                    mag_time.append(time)
                    mag_err.append(err)

plt.figure(figsize=(9, 6))
plt.loglog(rk_time, rk_err, 'o-', label='Рунге-Кутта 4', color='red', markersize=8)
plt.loglog(mag_time, mag_err, 's-', label='Piecewise Magnus', color='blue', markersize=8)
plt.xlabel('Время вычислений (мс)')
plt.ylabel('Глобальная ошибка')
plt.title('Exp 3b: Диаграмма Работа-Точность (Work-Precision)')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('plot_exp3b.png', dpi=300)
plt.show()
