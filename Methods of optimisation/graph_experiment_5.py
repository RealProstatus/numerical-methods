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

def parse_experiment5_results(filename):
    """Парсинг результатов эксперимента 5 из файла experiment5_results.txt"""
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

def plot_experiment5(results):
    """Графики для эксперимента 5: Устойчивость методов экспоненцирования"""
    if not results:
        print("No data for experiment 5")
        return

    kappas = sorted(results.keys())
    methods = ['Taylor', 'Chebyshev', 'Eigen Pade']

    # Создаем фигуру
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Эксперимент 5: Устойчивость методов экспоненцирования\n(Зависимость точности exp(-i*H) от числа обусловленности)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = {'Taylor': 'blue', 'Chebyshev': 'red', 'Eigen Pade': 'green'}

    # График 1: Точность по элементам vs число обусловленности
    ax1 = plt.subplot(2, 2, 1)
    for method in methods:
        el_errors = []
        for kappa in kappas:
            if method in results[kappa]:
                el_errors.append(results[kappa][method]['max_el'][0])
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

    # График 4: Сравнение методов для каждого kappa
    ax4 = plt.subplot(2, 2, 4)

    bar_width = 0.25
    x = np.arange(len(kappas))

    for i, method in enumerate(methods):
        el_errors = []
        for kappa in kappas:
            if method in results[kappa]:
                el_errors.append(results[kappa][method]['max_el'][0])
            else:
                el_errors.append(0)

        ax4.bar(x + i*bar_width, el_errors, bar_width, label=method,
               alpha=0.8, color=colors[method])

    ax4.set_xlabel('Число обусловленности κ')
    ax4.set_ylabel('Макс. разность элементов')
    ax4.set_title('Сравнение методов по точности')
    ax4.set_xticks(x + bar_width)
    ax4.set_xticklabels([f'{k:g}' for k in kappas])
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_yscale('log')

    # Добавляем значения на столбцы
    for i, method in enumerate(methods):
        el_errors = []
        for kappa in kappas:
            if method in results[kappa]:
                el_errors.append(results[kappa][method]['max_el'][0])
            else:
                el_errors.append(0)

        for j, val in enumerate(el_errors):
            if val > 0:
                ax4.text(x[j] + i*bar_width, val*1.1, f'{val:.1e}',
                        ha='center', va='bottom', fontsize=7, rotation=90)

    plt.tight_layout()
    plt.savefig('experiment5_exponent_stability.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Сохранен experiment5_exponent_stability.png")
    plt.show()

    # Дополнительные графики: эффективность
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Эксперимент 5: Эффективность методов экспоненцирования', fontsize=14, fontweight='bold')

    # График 5: Эффективность по элементам
    ax5 = plt.subplot(1, 2, 1)
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
        ax5.plot(kappas, safe_eff, 'o-', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax5.set_xlabel('Число обусловленности κ')
    ax5.set_ylabel('Ошибка/Время (эффективность)')
    ax5.set_title('Эффективность по элементам')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    ax5.set_xscale('log')
    ax5.set_yscale('log')

    # График 6: Эффективность по собственным значениям
    ax6 = plt.subplot(1, 2, 2)
    for method in methods:
        efficiency = []
        for kappa in kappas:
            if method in results[kappa]:
                eig_err = results[kappa][method]['max_eig'][0]
                time_val = results[kappa][method]['time'][0]
                if time_val > 0 and eig_err > 0:
                    efficiency.append(eig_err / time_val)
                else:
                    efficiency.append(np.nan)
            else:
                efficiency.append(np.nan)

        safe_eff = safe_log_values(efficiency)
        ax6.plot(kappas, safe_eff, 's--', label=method,
                color=colors[method], linewidth=2, markersize=8)

    ax6.set_xlabel('Число обусловленности κ')
    ax6.set_ylabel('Ошибка/Время (эффективность)')
    ax6.set_title('Эффективность по собств. значениям')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)
    ax6.set_xscale('log')
    ax6.set_yscale('log')

    plt.tight_layout()
    plt.savefig('experiment5_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Сохранен experiment5_efficiency.png")
    plt.show()

    # Финальная статистика
    print("\n=== Статистика Эксперимента 5 ===")
    print("Устойчивость методов вычисления exp(-i*H) при разных числах обусловленности")
    print("Эталон: exp(-i*H) = V*exp(-i*D)*V^(-1)")

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

        # Сравнение методов для данного kappa
        print("  Сравнение методов:")
        best_el = min([results[kappa][m]['max_el'][0] for m in methods if m in results[kappa]])
        best_eig = min([results[kappa][m]['max_eig'][0] for m in methods if m in results[kappa]])

        for method in methods:
            if method in results[kappa]:
                el_ratio = results[kappa][method]['max_el'][0] / best_el if best_el > 0 else 1.0
                eig_ratio = results[kappa][method]['max_eig'][0] / best_eig if best_eig > 0 else 1.0
                print(".1f")

def main():
    print("Analysis of Experiment 5 results: Exponentiation stability")
    print("=" * 60)

    filename = "experiment5_results.txt"

    try:
        results = parse_experiment5_results(filename)

        if results:
            print(f"Найдены данные для {len(results)} значений обусловленности:")
            for kappa in sorted(results.keys()):
                methods_count = len(results[kappa])
                print(f"  κ = {kappa}: {methods_count} методов")

            plot_experiment5(results)
        else:
            print("Данные эксперимента 5 не найдены в файле!")

    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
