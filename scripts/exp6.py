import matplotlib.pyplot as plt

filepath = 'experiment6_results.txt'
rk_k, rk_e = [], []
mag_k, mag_e = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'Kappa' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                kappa, method, err = float(parts[0]), parts[1], float(parts[2])
                if 'Runge-Kutta' in method:
                    rk_k.append(kappa)
                    rk_e.append(err)
                elif 'Magnus' in method:
                    mag_k.append(kappa)
                    mag_e.append(err)

plt.figure(figsize=(8, 5))
plt.loglog(rk_k, rk_e, 'o-', label='Рунге-Кутта (dt=1e-4)', color='red')
plt.loglog(mag_k, mag_e, 's-', label='Магнус (macro_dt=0.01)', color='blue')
plt.xlabel('Число обусловленности (Kappa)')
plt.ylabel('Глобальная ошибка U(T)')
plt.title('Exp 6: Глобальная ошибка от жесткости системы')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('plot_exp6.png', dpi=300)
plt.show()
