import matplotlib.pyplot as plt
from matplotlib.patches import Patch

filepath = 'experiment9_adaptive_inner.txt'
segments, halvings = [], []

with open(filepath, 'r') as f:
    for line in f:
        if '|' in line and 'Segment' not in line and '----' not in line:
            parts = [p.strip() for p in line.split('|')]
            if len(parts) == 6:
                segments.append(int(parts[0]))
                halvings.append(int(parts[4]))

plt.figure(figsize=(10, 5))
bars = plt.bar(segments, halvings, color='#17becf', edgecolor='black')

# Подсвечиваем участки, где метод сэкономил шаги
for i, h in enumerate(halvings):
    if h < 9:
        bars[i].set_color('#ff7f0e')
        bars[i].set_edgecolor('black')

plt.xticks(segments)
plt.yticks(range(0, 11))
plt.xlabel('Номер сегмента времени')
plt.ylabel('Количество делений шага (Halvings)')
plt.title('Exp 9: Работа адаптивного алгоритма интеграции')

legend_elements = [
    Patch(facecolor='#17becf', edgecolor='black', label='9 делений пополам'),
    Patch(facecolor='#ff7f0e', edgecolor='black', label='8 делений пополам')
]
plt.legend(handles=legend_elements, loc='lower right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('plot_exp9.png', dpi=300)
plt.show()
