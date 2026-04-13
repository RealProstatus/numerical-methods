import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os

def parse_experiment6_results(filename):
    """Парсинг результатов эксперимента 6 из файла experiment6_results.txt"""
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

    # Находим начало эксперимента 6
    exp6_start = content.find("EXPERIMENT 6:")
    if exp6_start == -1:
        print("EXPERIMENT 6 not found in file!")
        return results

    # Если эксперимент заканчивается концом файла
    exp6_end = content.find("EXPERIMENT", exp6_start + 1)
    if exp6_end == -1:
        exp6_content = content[exp6_start:]
    else:
        exp6_content = content[exp6_start:exp6_end]

    lines = exp6_content.strip().split('\n')

    in_table = False
    for line in lines:
        # Ищем заголовок таблицы
        if "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)" in line:
            in_table = True
            continue

        # Пропускаем разделительные линии
        if in_table and ("---" in line or line.strip() == "" or line.startswith("=")):
            continue

        # Парсим строки данных
        if in_table and '|' in line and line.strip():
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 5:
                try:
                    kappa = float(parts[0])
                    method = parts[1]
                    max_el = float(parts[2])
                    max_eig = float(parts[3])
                    time = float(parts[4])

                    if method not in results:
                        results[method] = {'kappa': [], 'max_el': [], 'max_eig': [], 'time': []}

                    results[method]['kappa'].append(kappa)
                    results[method]['max_el'].append(max_el)
                    results[method]['max_eig'].append(max_eig)
                    results[method]['time'].append(time)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue

    return results

def plot_experiment6(results):
    """Графики для эксперимента 6: Стабильность решения ОДУ"""
    if not results:
        print("No data for experiment 6")
        return

    # Создаем фигуру
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle('Эксперимент 6: Стабильность решения ОДУ U\'(t) = -iH·U(t)\n(Зависимость точности от обусловленности матрицы)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = ['blue', 'green', 'red', 'purple']

    methods = list(results.keys())

    # График 1: Точность по элементам vs обусловленность
    ax1 = plt.subplot(2, 2, 1)
    for i, method in enumerate(methods):
        data = results[method]
        if data['kappa'] and data['max_el']:
            color = colors[i % len(colors)]
            ax1.plot(data['kappa'], data['max_el'], 'o-', label=method.replace('dt=1e-5', '(dt=10⁻⁵)'),
                    color=color, linewidth=2, markersize=8)

    ax1.set_xlabel('Число обусловленности (κ)')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность по элементам')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9, loc='best')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xticks([10, 100, 1000, 10000])
    ax1.set_xticklabels(['10', '100', '1000', '10000'])

    # График 2: Точность по собственным значениям vs обусловленность
    ax2 = plt.subplot(2, 2, 2)
    for i, method in enumerate(methods):
        data = results[method]
        if data['kappa'] and data['max_eig']:
            color = colors[i % len(colors)]
            ax2.plot(data['kappa'], data['max_eig'], 's--', label=method.replace('dt=1e-5', '(dt=10⁻⁵)'),
                    color=color, linewidth=2, markersize=8)

    ax2.set_xlabel('Число обусловленности (κ)')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность по собственным значениям')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, loc='best')
    ax2.set_xscale('log')
    ax2.set_yscale('log')
    ax2.set_xticks([10, 100, 1000, 10000])
    ax2.set_xticklabels(['10', '100', '1000', '10000'])

    # График 3: Время выполнения vs обусловленность
    ax3 = plt.subplot(2, 2, 3)
    for i, method in enumerate(methods):
        data = results[method]
        if data['kappa'] and data['time']:
            color = colors[i % len(colors)]
            ax3.plot(data['kappa'], data['time'], '^-', label=method.replace('dt=1e-5', '(dt=10⁻⁵)'),
                    color=color, linewidth=2, markersize=8)

    ax3.set_xlabel('Число обусловленности (κ)')
    ax3.set_ylabel('Время выполнения (мс)')
    ax3.set_title('Производительность')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='best')
    ax3.set_xscale('log')
    ax3.set_yscale('log')
    ax3.set_xticks([10, 100, 1000, 10000])
    ax3.set_xticklabels(['10', '100', '1000', '10000'])

    # График 4: Соотношение точность/время для каждого метода
    ax4 = plt.subplot(2, 2, 4)

    # Для каждого kappa покажем точность обоих методов
    kappas = []
    rk_errors = []
    magnus_errors = []

    for method in methods:
        data = results[method]
        if 'Runge-Kutta' in method:
            kappas = data['kappa']
            rk_errors = data['max_el']
        elif 'Magnus' in method:
            magnus_errors = data['max_el']

    if kappas and rk_errors and magnus_errors:
        # Вычислим выигрыш в точности Магнуса
        accuracy_gain = np.array(rk_errors) / np.array(magnus_errors)

        ax4.plot(kappas, accuracy_gain, 'o-', color='purple', linewidth=2, markersize=8,
                label='Выигрыш Магнуса в точности')
        ax4.axhline(y=1, color='red', linestyle='--', alpha=0.7, label='Одинаковая точность')

        ax4.set_xlabel('Число обусловленности (κ)')
        ax4.set_ylabel('Кратность превосходства Магнуса')
        ax4.set_title('Относительная точность методов')
        ax4.grid(True, alpha=0.3)
        ax4.legend(fontsize=9, loc='best')
        ax4.set_xscale('log')
        ax4.set_yscale('log')
        ax4.set_xticks([10, 100, 1000, 10000])
        ax4.set_xticklabels(['10', '100', '1000', '10000'])

    plt.tight_layout()
    plt.savefig('experiment6_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment6_analysis.png")
    plt.show()

    # Additional statistics
    print("\n=== Experiment 6 Statistics ===")
    for method, data in results.items():
        if data['kappa']:
            print(f"\n{method}:")
            print(f"  Kappa range: {min(data['kappa'])} - {max(data['kappa'])}")
            print(".2e")
            print(".2e")
            print(".2f")

            # Показать тренд точности при росте kappa
            if len(data['kappa']) > 1:
                kappa_ratio = max(data['kappa']) / min(data['kappa'])
                error_ratio_el = max(data['max_el']) / min(data['max_el'])
                error_ratio_eig = max(data['max_eig']) / min(data['max_eig'])
                print(".1f")
                print(".1f")
                print(".1f")

def main():
    print("Analysis of Experiment 6 results: ODE stability")
    print("=" * 50)

    # Get filename from command line arguments or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "experiment6_results.txt"

    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    print(f"Looking for file: {filename}")
    print(f"Full path: {os.path.abspath(filename)}")
    print(f"File exists: {os.path.exists(filename)}")

    if not os.path.exists(filename):
        print("\nFile not found! Possible solutions:")
        print("1. Run script from Methods of optimisation directory")
        print("2. Specify full path: python graph_experiment_6.py /path/to/file")
        print("3. Use run_graphs.bat")
        return

    try:
        results = parse_experiment6_results(filename)

        if results:
            print(f"Found {len(results)} methods for analysis:")
            for method in results.keys():
                print(f"  - {method}")

            plot_experiment6(results)
        else:
            print("Experiment 6 data not found in file!")

    except FileNotFoundError:
        print(f"Error: file {filename} not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
