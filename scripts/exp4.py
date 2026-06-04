import os
import matplotlib.pyplot as plt

# === ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ ===
FILE_PATH = 'experiment4_results.txt' # поменяйте путь, если файлы лежат в той же папке
SAVE_FIG = True
FIG_NAME = 'exp4_stability_split.png'
FIG_SIZE = (9, 8) # Сделали график повыше для двух панелей
FONT_SIZE = 14

# Настройка стилей. 
# ВАЖНО: Recursive сделан пунктирным ('--'), чтобы из-под него было видно Classic
METHODS_STYLE = {
    'Classic':      {'color': '#2ca02c', 'marker': 'o', 'linestyle': '-'},
    'Recursive':    {'color': '#1f77b4', 'marker': '^', 'linestyle': '--'}, 
    'ACC':          {'color': '#ff7f0e', 'marker': 's', 'linestyle': '-'},
    'Formula 3.14': {'color': '#d62728', 'marker': 'D', 'linestyle': '-'}
}
LINE_WIDTH = 2
MARKER_SIZE = 8
# ===============================

plt.rcParams.update({'font.size': FONT_SIZE, 'axes.grid': True})

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Ошибка: Файл {FILE_PATH} не найден.")
        return

    # Подготовка словаря для данных
    data_dict = {method: {'kappas': [], 'errors': []} for method in METHODS_STYLE.keys()}

    with open(FILE_PATH, 'r') as f:
        for line in f:
            if '|' in line and 'Kappa' not in line and '----' not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    kappa = float(parts[0])
                    method = parts[1]
                    error = float(parts[2])
                    
                    if method in data_dict:
                        data_dict[method]['kappas'].append(kappa)
                        data_dict[method]['errors'].append(error)

    # Создаем фигуру с двумя графиками (один над другим), общая ось X
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIG_SIZE, sharex=True)
    
    for method, data in data_dict.items():
        if not data['kappas']:
            continue
            
        style = METHODS_STYLE[method]
        
        # Формулу 3.14 рисуем на верхнем графике (ax1)
        if method == 'Formula 3.14':
            ax1.loglog(data['kappas'], data['errors'], marker=style['marker'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=method)
        # Остальные рисуем на нижнем графике (ax2)
        else:
            ax2.loglog(data['kappas'], data['errors'], marker=style['marker'], 
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label=method)
    
    # Настройка верхнего графика (Формула 3.14)
    ax1.set_title('Стабильность методов разложения Магнуса')
    ax1.set_ylabel('Ошибка (Max Element Diff)')
    ax1.legend(loc='lower left')
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    # Настройка нижнего графика (Методы около машинного нуля)
    ax2.set_xlabel('Число обусловленности матрицы (\u03BA)')
    ax2.set_ylabel('Ошибка (Уровень маш. нуля)')
    ax2.legend(loc='lower left')
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    # Убираем лишнее пространство между графиками
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(FIG_NAME, dpi=300)
        print(f"График сохранен как {FIG_NAME}")
    plt.show()

if __name__ == "__main__":
    main()