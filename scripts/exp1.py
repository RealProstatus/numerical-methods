import matplotlib.pyplot as plt
import pandas as pd

filepath = 'magnus_experiment_results.txt'

# 1. Чтение и парсинг файла (с защитой от разорванных строк)
data = []
with open(filepath, 'r') as f:
    text = f.read()

text = text.replace('|\n', '| ')

for line in text.split('\n'):
    if '|' in line and 'Method' not in line and '----' not in line:
        parts = [p.strip() for p in line.split('|')]
        if len(parts) >= 5 and parts[0] != '':
            try:
                method = parts[0]
                k = int(parts[1])
                err = float(parts[2])
                time = float(parts[4])
                data.append({'Method': method, 'K': k, 'Error': err, 'Time': time})
            except ValueError:
                continue

df = pd.DataFrame(data)

# 2. Создаем удобную подпись для оси X вида "ACC (k=1)"
df['Label'] = df['Method'] + ' (k=' + df['K'].astype(str) + ')'

# 3. Создаем фигуру с двумя графиками (один под другим)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# --- ВЕРХНИЙ ГРАФИК: Ошибка ---
ax1.bar(df['Label'], df['Error'], color='#87CEFA', edgecolor='black')
ax1.set_yscale('log')
ax1.set_ylabel('Max Element Diff (Лог. шкала)')
ax1.set_title('Ошибка вычисления матрицы (Меньше = Лучше)')
ax1.grid(axis='y', ls='--', alpha=0.7)

# --- НИЖНИЙ ГРАФИК: Время работы ---
ax2.bar(df['Label'], df['Time'], color='#F08080', edgecolor='black')
ax2.set_yscale('log')
ax2.set_ylabel('Время работы, ms (Лог. шкала)')
ax2.set_title('Затраченное время (Меньше = Лучше)')
ax2.grid(axis='y', ls='--', alpha=0.7)

# Наклоняем подписи по оси X, чтобы они не сливались
ax2.set_xticklabels(df['Label'], rotation=45, ha='right')

plt.tight_layout()
plt.savefig('plot_exp1_bars.png', dpi=300)
plt.show()