import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os

def parse_experiment3_results(filename):
    """Парсинг результатов эксперимента 3 из файла experiment3_results.txt"""
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

    # Находим начало эксперимента 3
    exp3_start = content.find("EXPERIMENT 3:")
    if exp3_start == -1:
        print("EXPERIMENT 3 not found in file!")
        return results

    # Если эксперимент 4 не найден, берем до конца файла
    exp3_end = content.find("EXPERIMENT 4:")
    if exp3_end == -1:
        exp3_content = content[exp3_start:]
    else:
        exp3_content = content[exp3_start:exp3_end]

    lines = exp3_content.strip().split('\n')

    in_table = False
    for line in lines:
        # Ищем заголовок таблицы
        if "Method | Max Element Diff | Max Eig Diff | Time (ms)" in line:
            in_table = True
            continue

        # Пропускаем разделительные линии
        if in_table and ("---" in line or line.strip() == "" or line.startswith("=")):
            continue

        # Парсим строки данных
        if in_table and '|' in line and line.strip():
            parts = [part.strip() for part in line.split('|')]
            if len(parts) >= 4:
                try:
                    method = parts[0]
                    max_el = float(parts[1])
                    max_eig = float(parts[2])
                    time = float(parts[3])

                    if method not in results:
                        results[method] = {'max_el': [], 'max_eig': [], 'time': []}

                    results[method]['max_el'].append(max_el)
                    results[method]['max_eig'].append(max_eig)
                    results[method]['time'].append(time)

                except (ValueError, IndexError) as e:
                    print(f"Error parsing line: {line}")
                    print(f"Error: {e}")
                    continue

    return results

def plot_experiment3(results):
    """Графики для эксперимента 3: Сравнение методов решения ОДУ"""
    if not results:
        print("No data for experiment 3")
        return

    # Создаем фигуру
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('Эксперимент 3: Решение ОДУ U\'(t) = -iH·U(t)\n(Сравнение Рунге-Кутты и Магнуса с эталонным решением)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = ['blue', 'green', 'red', 'purple']

    methods = list(results.keys())
    max_el_errors = []
    max_eig_errors = []
    times = []
    display_names = []

    for method in methods:
        data = results[method]
        if data['max_el'] and data['max_eig'] and data['time']:
            max_el_errors.append(data['max_el'][0])
            max_eig_errors.append(data['max_eig'][0])
            times.append(data['time'][0])

            # Форматируем названия для отображения
            if 'Runge-Kutta' in method:
                display_names.append('Рунге-Кутта\n(dt=10⁻⁵)')
            elif 'Magnus' in method:
                display_names.append('Магнус ACC +\nЧебышев')

    # График 1: Сравнение точности по элементам
    ax1 = plt.subplot(1, 3, 1)
    bars1 = ax1.bar(range(len(methods)), max_el_errors, color=colors[:len(methods)], alpha=0.7)
    ax1.set_xlabel('Метод')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность по элементам')
    ax1.set_xticks(range(len(methods)))
    ax1.set_xticklabels(display_names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_yscale('log')

    # Добавляем значения на столбцы
    for bar, val in zip(bars1, max_el_errors):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height*1.1,
                '.1e', ha='center', va='bottom', fontsize=8)

    # График 2: Сравнение точности по собственным значениям
    ax2 = plt.subplot(1, 3, 2)
    bars2 = ax2.bar(range(len(methods)), max_eig_errors, color=colors[:len(methods)], alpha=0.7)
    ax2.set_xlabel('Метод')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность по собственным значениям')
    ax2.set_xticks(range(len(methods)))
    ax2.set_xticklabels(display_names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_yscale('log')

    # Добавляем значения на столбцы
    for bar, val in zip(bars2, max_eig_errors):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height*1.1,
                '.1e', ha='center', va='bottom', fontsize=8)

    # График 3: Сравнение времени выполнения
    ax3 = plt.subplot(1, 3, 3)
    bars3 = ax3.bar(range(len(methods)), times, color=colors[:len(methods)], alpha=0.7)
    ax3.set_xlabel('Метод')
    ax3.set_ylabel('Время выполнения (мс)')
    ax3.set_title('Производительность')
    ax3.set_xticks(range(len(methods)))
    ax3.set_xticklabels(display_names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    # Добавляем значения на столбцы
    for bar, val in zip(bars3, times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height*1.05,
                '.0f', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig('experiment3_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment3_analysis.png")
    plt.show()

    # Additional statistics
    print("\n=== Experiment 3 Statistics ===")
    for method, data in results.items():
        if data['max_el'] and data['max_eig'] and data['time']:
            print(f"\n{method}:")
            print(f"  Max element diff: {data['max_el'][0]:.2e}")
            print(f"  Max eigenvalue diff: {data['max_eig'][0]:.2e}")
            print(f"  Time: {data['time'][0]:.2f}ms")

def main():
    print("Analysis of Experiment 3 results: ODE solving")
    print("=" * 50)

    # Get filename from command line arguments or use default
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        filename = "experiment3_results.txt"

    print(f"Current directory: {os.getcwd()}")
    print(f"Files in directory: {os.listdir('.')}")
    print(f"Looking for file: {filename}")
    print(f"Full path: {os.path.abspath(filename)}")
    print(f"File exists: {os.path.exists(filename)}")

    if not os.path.exists(filename):
        print("\nFile not found! Possible solutions:")
        print("1. Run script from Methods of optimisation directory")
        print("2. Specify full path: python graph_experiment_3.py /path/to/file")
        print("3. Use run_graphs.bat")
        return

    try:
        results = parse_experiment3_results(filename)

        if results:
            print(f"Found {len(results)} methods for analysis:")
            for method in results.keys():
                print(f"  - {method}")

            plot_experiment3(results)
        else:
            print("Experiment 3 data not found in file!")

    except FileNotFoundError:
        print(f"Error: file {filename} not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
