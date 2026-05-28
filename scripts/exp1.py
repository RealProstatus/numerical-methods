import matplotlib.pyplot as plt

filepath = 'magnus_experiment_results.txt'
methods = []
errors = []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'Method' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 4:
                # Объединяем имя метода и его порядок k
                methods.append(f"{parts[0]} (k={parts[1]})")
                errors.append(float(parts[2]))

plt.figure(figsize=(10, 6))
plt.barh(methods, errors, color='skyblue', edgecolor='black')
plt.axvline(1e-16, color='red', linestyle='--', label='Машинный ноль (~1e-16)')
plt.xscale('log')
plt.xlabel('Max Element Diff (Лог. шкала)')
plt.title('Exp 1: Ошибка вычисления матрицы Omega (dt=1e-5)')
plt.legend()
plt.tight_layout()
plt.savefig('plot_exp1.png', dpi=300)
plt.show()
