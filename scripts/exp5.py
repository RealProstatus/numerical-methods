import matplotlib.pyplot as plt

filepath = 'experiment5_results.txt'
taylor_k, taylor_e = [], []
cheb_k, cheb_e = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'Method' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                kappa, method, err = float(parts[0]), parts[1], float(parts[2])
                if method == 'Taylor':
                    taylor_k.append(kappa)
                    taylor_e.append(err)
                elif method == 'Chebyshev':
                    cheb_k.append(kappa)
                    cheb_e.append(err)

plt.figure(figsize=(8, 5))
plt.loglog(taylor_k, taylor_e, 'o-', label='Тейлор', color='brown')
plt.loglog(cheb_k, cheb_e, 's-', label='Чебышев', color='purple')
plt.ylim(1e-17, 1e-14) # Сужаем график, так как ошибки около 1e-16
plt.xlabel('Число обусловленности (Kappa)')
plt.ylabel('Ошибка экспоненты')
plt.title('Exp 5: Стабильность экспоненты от жесткости')
plt.grid(True, which='both', ls='--')
plt.legend()
plt.savefig('plot_exp5.png', dpi=300)
plt.show()
