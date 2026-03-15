import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os


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
    if exp1_start == -1:
        print("EXPERIMENT 1 not found in file!")
        return results

    # Если эксперимент 2 не найден, берем до конца файла
    exp1_end = content.find("EXPERIMENT 2:")
    if exp1_end == -1:
        exp1_content = content[exp1_start:]
    else:
        exp1_content = content[exp1_start:exp1_end]

    lines = exp1_content.strip().split('\n')

    in_table = False
    for line in lines:
        # Ищем заголовок таблицы (может быть разбит на несколько строк)
        if "Method | K | Max Element Diff | Max Eig Diff | Time (ms)" in line:
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
                    method = parts[0]
                    k = int(parts[1])
                    max_el = float(parts[2])
                    max_eig = float(parts[3])
                    time = float(parts[4])

                    if method not in results:
                        results[method] = {'k': [], 'max_el': [], 'max_eig': [], 'time': []}

                    results[method]['k'].append(k)
                    results[method]['max_el'].append(max_el)
                    results[method]['max_eig'].append(max_eig)
                    results[method]['time'].append(time)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue

    return results

def plot_experiment1(results):
    """Графики для эксперимента 1: Сравнение методов Магнуса"""
    if not results:
        print("No data for experiment 1")
        return

    # Создаем фигуру
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Эксперимент 1: Сравнение методов Магнуса\n(Точность вычисления Ω для постоянного гамильтониана)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']

    # График 1: Точность по элементам
    ax1 = plt.subplot(1, 3, 1)
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
    ax2 = plt.subplot(1, 3, 2)
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
    ax3 = plt.subplot(1, 3, 3)
    for i, (method, data) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax3.plot(data['k'], data['time'], '^-', label=method,
                color=color, linewidth=2, markersize=7)

    ax3.set_xlabel('Порядок метода (K)')
    ax3.set_ylabel('Время выполнения (мс)')
    ax3.set_title('Производительность')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9, loc='best')

    plt.tight_layout()
    plt.savefig('experiment1_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment1_analysis.png")
    plt.show()

    # Additional statistics
    print("\n=== Experiment 1 Statistics ===")
    for method, data in results.items():
        print(f"\n{method}:")
        print(f"  Order K: {data['k']}")
        print(f"  Max element diff: min={min(data['max_el']):.2e}, max={max(data['max_el']):.2e}")
        print(f"  Max eigenvalue diff: min={min(data['max_eig']):.2e}, max={max(data['max_eig']):.2e}")
        print(f"  Time: min={min(data['time']):.2f}ms, max={max(data['time']):.2f}ms")
        print(f"  Best element accuracy: {min(data['max_el']):.2e}")
        print(f"  Best eigenvalue accuracy: {min(data['max_eig']):.2e}")

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
