import matplotlib.pyplot as plt
import numpy as np
import re

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

def parse_experiment4_results(filename):
    """Парсинг результатов эксперимента 4 из файла experiment4_results.txt"""
    with open(filename, 'r') as f:
        content = f.read()

    results = {}

    # Находим таблицу с результатами
    lines = content.strip().split('\n')

    in_table = False
    for line in lines:
        if "Kappa | Method | Max Element Diff | Max Eig Diff | Time (ms)" in line:
            in_table = True
            continue

        if in_table and "---" not in line and line.strip() and not line.startswith("="):
            parts = line.split('|')
            if len(parts) >= 5:
                kappa = float(parts[0].strip())
                method = parts[1].strip()
                max_el = float(parts[2].strip())
                max_eig = float(parts[3].strip())
                time = float(parts[4].strip())

                if kappa not in results:
                    results[kappa] = {}

                if method not in results[kappa]:
                    results[kappa][method] = {'max_el': [], 'max_eig': [], 'time': []}

                results[kappa][method]['max_el'].append(max_el)
                results[kappa][method]['max_eig'].append(max_eig)
                results[kappa][method]['time'].append(time)

    return results

def plot_experiment4(results):
    """Графики для эксперимента 4: Устойчивость методов Магнуса"""
    if not results:
        print("No data for experiment 4")
        return

    kappas = sorted(results.keys())
    methods = ['Classic', 'ACC', 'Recursive', 'Formula 3.14']

    # Создаем фигуру
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Эксперимент 4: Устойчивость методов Магнуса\n(Зависимость точности от числа обусловленности)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = {'Classic': 'blue', 'ACC': 'green', 'Recursive': 'red', 'Formula 3.14': 'purple'}

    # График 1: Точность по элементам vs число обусловленности
    ax1 = plt.subplot(2, 2, 1)
    for method in methods:
        el_errors = []
        for kappa in kappas:
            if method in results[kappa]:
                el_errors.append(results[kappa][method]['max_el'][0])  # берем первое значение
            else:
                el_errors.append(np.nan)

        ax1.plot(kappas, el_errors, 'o-', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax1.set_xlabel('Число обусловленности κ')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность по элементам')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    ax1.set_xscale('log')
    ax1.set_yscale('log')

    # График 2: Точность по собственным значениям vs число обусловленности
    ax2 = plt.subplot(2, 2, 2)
    for method in methods:
        eig_errors = []
        for kappa in kappas:
            if method in results[kappa]:
                eig_errors.append(results[kappa][method]['max_eig'][0])
            else:
                eig_errors.append(np.nan)

        ax2.plot(kappas, eig_errors, 's--', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax2.set_xlabel('Число обусловленности κ')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность по собственным значениям')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10)
    ax2.set_xscale('log')
    ax2.set_yscale('log')

    # График 3: Время выполнения vs число обусловленности
    ax3 = plt.subplot(2, 2, 3)
    for method in methods:
        times = []
        for kappa in kappas:
            if method in results[kappa]:
                times.append(results[kappa][method]['time'][0])
            else:
                times.append(np.nan)

        ax3.plot(kappas, times, '^-', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax3.set_xlabel('Число обусловленности κ')
    ax3.set_ylabel('Время выполнения (мс)')
    ax3.set_title('Производительность')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xscale('log')

    # График 4: Относительная точность (элементы/время)
    ax4 = plt.subplot(2, 2, 4)
    for method in methods:
        efficiency = []
        for kappa in kappas:
            if method in results[kappa]:
                el_err = results[kappa][method]['max_el'][0]
                time_val = results[kappa][method]['time'][0]
                if time_val > 0 and el_err > 0:
                    efficiency.append(el_err / time_val)
                else:
                    efficiency.append(np.nan)
            else:
                efficiency.append(np.nan)

        safe_eff = safe_log_values(efficiency)
        ax4.plot(kappas, safe_eff, 'o-', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax4.set_xlabel('Число обусловленности κ')
    ax4.set_ylabel('Ошибка/Время (эффективность)')
    ax4.set_title('Эффективность (меньше = лучше)')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=10)
    ax4.set_xscale('log')
    ax4.set_yscale('log')

    plt.tight_layout()
    plt.savefig('experiment4_stability_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Сохранен experiment4_stability_analysis.png")
    plt.show()

    # Дополнительная статистика
    print("\n=== Статистика Эксперимента 4 ===")
    print("Устойчивость методов Магнуса при разных числах обусловленности")
    print("Эталон: Ω = H × dt")

    for kappa in kappas:
        print(f"\nЧисло обусловленности κ = {kappa}:")
        for method in methods:
            if method in results[kappa]:
                data = results[kappa][method]
                print(".2e")
                print(".2e")
                print(".2f")
            else:
                print(f"  {method}: данные отсутствуют")

def main():
    print("Analysis of Experiment 4 results: Magnus methods stability")
    print("=" * 60)

    filename = "experiment4_results.txt"

    try:
        results = parse_experiment4_results(filename)

        if results:
            print(f"Найдены данные для {len(results)} значений обусловленности:")
            for kappa in sorted(results.keys()):
                methods_count = len(results[kappa])
                print(f"  κ = {kappa}: {methods_count} методов")

            plot_experiment4(results)
        else:
            print("Данные эксперимента 4 не найдены в файле!")

    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
