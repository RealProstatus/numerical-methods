import matplotlib.pyplot as plt
import pandas as pd

filepath = 'magnus_experiment_results.txt'

# Чтение и парсинг данных
data = []
with open(filepath, 'r') as f:
    for line in f:
        # Игнорируем строки без разделителя или заголовки
        if '|' in line and 'Method' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5 and parts[0] != '':
                method = parts[0]
                k = int(parts[1])
                err = float(parts[2])
                time = float(parts[4])
                data.append({'Method': method, 'K': k, 'Error': err, 'Time': time})

df = pd.DataFrame(data)

# Построение графика
plt.figure(figsize=(10, 6))

methods = df['Method'].unique()
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
markers = ['o', 's', '^', 'D']

# Отрисовка точек для каждого метода
for i, method in enumerate(methods):
    subset = df[df['Method'] == method]
    plt.scatter(subset['Time'], subset['Error'], 
                label=method, s=100, alpha=0.8, edgecolors='k',
                color=colors[i % len(colors)], marker=markers[i % len(markers)])
    
    # Добавление подписей порядка 'k' рядом с каждой точкой
    for _, row in subset.iterrows():
        plt.annotate(f"k={int(row['K'])}", (row['Time'], row['Error']), 
                     xytext=(8, -3), textcoords='offset points', fontsize=9)

# Настройка осей и внешнего вида
plt.xscale('log')
plt.yscale('log')
plt.xlabel('Время работы, Time (ms) [Лог. шкала]')
plt.ylabel('Ошибка, Max Element Diff [Лог. шкала]')
plt.title('Эксперимент 1: Соотношение Ошибки и Времени работы')

plt.grid(True, which="both", ls="--", alpha=0.3)
plt.legend(title="Методы")
plt.tight_layout()

# Сохранение и отображение
plt.savefig('plot_exp1_scatter.png', dpi=300)
plt.show()