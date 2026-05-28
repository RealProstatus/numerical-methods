import matplotlib.pyplot as plt
import numpy as np

filepath = 'experiment3_results.txt'
methods, errors, times = [], [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line:
            parts = [p.strip() for p in line.split('|')]
            methods.append(parts[0])
            errors.append(float(parts[1]))
            times.append(float(parts[2]))

fig, ax1 = plt.subplots(figsize=(8, 5))
x = np.arange(len(methods))
width = 0.35

ax1.bar(x - width/2, times, width, color='blue', label='Время (мс)')
ax1.set_ylabel('Время (мс)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.bar(x + width/2, errors, width, color='green', label='Ошибка')
ax2.set_ylabel('Ошибка', color='green')
ax2.set_yscale('log')
ax2.tick_params(axis='y', labelcolor='green')

ax1.set_xticks(x)
ax1.set_xticklabels(methods)
plt.title('Exp 3: Сравнение РК4 и Магнуса (Точечный зарезервированный тест)')
fig.tight_layout()
plt.savefig('plot_exp3.png', dpi=300)
plt.show()
