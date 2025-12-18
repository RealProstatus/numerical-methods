import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os

def safe_log_values(values, min_val=1e-20):
    """Безопасное преобразование значений для логарифмической шкалы"""
    safe_values = []
    for v in values:
        if v <= 0:
            safe_values.append(min_val)
        elif v < min_val:
            safe_values.append(min_val)
        else:
            safe_values.append(v)
    return np.array(safe_values)

def parse_experiment1_results(filename):
    """Парсинг результатов эксперимента 1 из файла magnus_experiment_results.txt"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
    except UnicodeDecodeError:
        # Если UTF-8 не работает, пробуем cp1251 (Windows)
        with open(filename, 'r', encoding='cp1251') as f:
            content = f.read()

    print(f"File read, length: {len(content)} characters")
    print(f"First 200 characters: {content[:200]}")

    results = {}

    # Находим начало эксперимента 1
    exp1_start = content.find("EXPERIMENT 1:")
    exp1_end = content.find("EXPERIMENT 2:")

    if exp1_start != -1 and exp1_end != -1:
        exp1_content = content[exp1_start:exp1_end]
        lines = exp1_content.strip().split('\n')

        in_table = False
        for line in lines:
            if "Method | K | Max Element Diff | Max Eig Diff | Time (ms)" in line:
                in_table = True
                continue

            if in_table and "---" not in line and line.strip() and not line.startswith("="):
                parts = line.split('|')
                if len(parts) >= 5:
                    method = parts[0].strip()
                    k = int(parts[1].strip())
                    max_el = float(parts[2].strip())
                    max_eig = float(parts[3].strip())
                    time = float(parts[4].strip())

                    if method not in results:
                        results[method] = {'k': [], 'max_el': [], 'max_eig': [], 'time': []}

                    results[method]['k'].append(k)
                    results[method]['max_el'].append(max_el)
                    results[method]['max_eig'].append(max_eig)
                    results[method]['time'].append(time)

    return results

def plot_experiment1(results):
    """Графики для эксперимента 1: Сравнение методов Магнуса"""
    if not results:
        print("No data for experiment 1")
        return

    # Создаем фигуру
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Эксперимент 1: Сравнение методов Магнуса\n(Точность вычисления Ω для постоянного гамильтониана)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # График 1: Точность по элементам
    ax1 = plt.subplot(2, 3, 1)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax1.plot(data['k'], data['max_el'], 'o-', label=method,
                color=color, linewidth=2, markersize=6)

    ax1.set_xlabel('Порядок метода (K)')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность: разность элементов')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax1.set_yscale('log')

    # График 2: Точность по собственным значениям
    ax2 = plt.subplot(2, 3, 2)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax2.plot(data['k'], data['max_eig'], 's--', label=method,
                color=color, linewidth=2, markersize=6)

    ax2.set_xlabel('Порядок метода (K)')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность: разность собственных значений')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.set_yscale('log')

    # График 3: Время выполнения
    ax3 = plt.subplot(2, 3, 3)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax3.plot(data['k'], data['time'], '^-', label=method,
                color=color, linewidth=2, markersize=7)

    ax3.set_xlabel('Порядок метода (K)')
    ax3.set_ylabel('Время выполнения (мс)')
    ax3.set_title('Производительность')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='best')

    # График 4: Компромисс точность-время (по элементам)
    ax4 = plt.subplot(2, 3, 4)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]

        safe_times = np.array(data['time'])
        safe_errors = safe_log_values(data['max_el'])

        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax4.scatter(safe_times[mask], safe_errors[mask],
                       label=method, color=color, s=60, alpha=0.8)

            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax4.plot(safe_times[mask][sorted_indices],
                        safe_errors[mask][sorted_indices],
                        '--', color=color, alpha=0.6, linewidth=2)

    ax4.set_xlabel('Время выполнения (мс)')
    ax4.set_ylabel('Макс. разность элементов')
    ax4.set_title('Компромисс: время vs точность (элементы)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=9, loc='best')

    # График 5: Компромисс точность-время (по собственным значениям)
    ax5 = plt.subplot(2, 3, 5)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]

        safe_times = np.array(data['time'])
        safe_errors = safe_log_values(data['max_eig'])

        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax5.scatter(safe_times[mask], safe_errors[mask],
                       label=method, color=color, s=60, alpha=0.8)

            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax5.plot(safe_times[mask][sorted_indices],
                        safe_errors[mask][sorted_indices],
                        '--', color=color, alpha=0.6, linewidth=2)

    ax5.set_xlabel('Время выполнения (мс)')
    ax5.set_ylabel('Макс. разность собств. значений')
    ax5.set_title('Компромисс: время vs точность (собств. значения)')
    ax5.set_xscale('log')
    ax5.set_yscale('log')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=9, loc='best')

    # График 6: Сравнение методов при максимальном порядке
    ax6 = plt.subplot(2, 3, 6)

    methods = list(results.keys())
    max_errors_el = []
    max_errors_eig = []
    max_times = []

    for method in methods:
        data = results[method]
        last_idx = -1
        max_errors_el.append(data['max_el'][last_idx])
        max_errors_eig.append(data['max_eig'][last_idx])
        max_times.append(data['time'][last_idx])

    x = np.arange(len(methods))
    width = 0.25

    bars1 = ax6.bar(x - width, max_errors_el, width, label='Элементы',
                   alpha=0.8, color='skyblue')
    bars2 = ax6.bar(x, max_errors_eig, width, label='Собств. знач.',
                   alpha=0.8, color='lightcoral')
    bars3 = ax6.bar(x + width, max_times, width, label='Время (мс)',
                   alpha=0.8, color='lightgreen')

    ax6.set_xlabel('Метод')
    ax6.set_ylabel('Значение')
    ax6.set_title('Сравнение при максимальном порядке K')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bars, data in [(bars1, max_errors_el), (bars2, max_errors_eig), (bars3, max_times)]:
        for bar, val in zip(bars, data):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height*1.05,
                    f'{val:.1e}', ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig('experiment1_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment1_analysis.png")
    plt.show()

    # Additional statistics
    print("\n=== Experiment 1 Statistics ===")
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Order K: {data['k']}")
        print(".2e")
        print(".2e")
        print(".2f")
        print(".2e")
        print(".2e")

def main():
    print("Analysis of Experiment 1 results: Magnus methods")
    print("=" * 50)

    # Get filename from command line arguments or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "magnus_experiment_results.txt"

    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    print(f"Looking for file: {filename}")
    print(f"Full path: {os.path.abspath(filename)}")
    print(f"File exists: {os.path.exists(filename)}")

    if not os.path.exists(filename):
        print("\nFile not found! Possible solutions:")
        print("1. Run script from Methods of optimisation directory")
        print("2. Specify full path: python graph_experiment_1.py /path/to/file")
        print("3. Use run_graphs.bat")
        return

    try:
        results = parse_experiment1_results(filename)

        if results:
            print(f"Found {len(results)} methods for analysis:")
            for method in results.keys():
                print(f"  - {method}")

            plot_experiment1(results)
        else:
            print("Experiment 1 data not found in file!")

    except FileNotFoundError:
        print(f"Error: file {filename} not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
