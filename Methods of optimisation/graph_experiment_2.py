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

def parse_experiment2_results(filename):
    """Парсинг результатов эксперимента 2 из файла experiment2_results.txt"""
    with open(filename, 'r') as f:
        content = f.read()

    results = {}

    # Находим таблицу с результатами
    lines = content.strip().split('\n')

    in_table = False
    for line in lines:
        if "Method | Param | Max Element Diff | Max Eig Diff | Time (ms)" in line:
            in_table = True
            continue

        if in_table and "---" not in line and line.strip() and not line.startswith("="):
            parts = line.split('|')
            if len(parts) >= 5:
                method = parts[0].strip()
                param = parts[1].strip()
                max_el = float(parts[2].strip())
                max_eig = float(parts[3].strip())
                time = float(parts[4].strip())

                if method not in results:
                    results[method] = {'param': [], 'max_el': [], 'max_eig': [], 'time': []}

                # Для Taylor и Chebyshev param - это число, для других - строка
                try:
                    param_val = int(param)
                except ValueError:
                    param_val = param

                results[method]['param'].append(param_val)
                results[method]['max_el'].append(max_el)
                results[method]['max_eig'].append(max_eig)
                results[method]['time'].append(time)

    return results

def plot_experiment2(results):
    """Графики для эксперимента 2: Сравнение методов экспоненцирования"""
    if not results:
        print("No data for experiment 2")
        return

    # Создаем фигуру
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('Эксперимент 2: Сравнение методов вычисления exp(Ω)\n(Матричная экспонента для анти-эрмитовой матрицы)', fontsize=14, fontweight='bold')

    # Цвета для разных методов
    colors = {'Taylor': 'blue', 'Chebyshev': 'red', 'Eigen Pade': 'green'}

    # Разделяем методы на Taylor, Chebyshev и reference
    taylor_data = results.get('Taylor', {})
    cheb_data = results.get('Chebyshev', {})

    # График 1: Точность Taylor метода по элементам
    ax1 = plt.subplot(2, 3, 1)
    if taylor_data:
        ax1.plot(taylor_data['param'], taylor_data['max_el'], 'o-',
                color=colors['Taylor'], linewidth=2, markersize=6)
        ax1.set_xlabel('Количество членов ряда Тейлора (K)')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Taylor: точность по элементам')
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax1.set_yscale('log')

    # График 2: Точность Chebyshev метода по элементам
    ax2 = plt.subplot(2, 3, 2)
    if cheb_data:
        ax2.plot(cheb_data['param'], cheb_data['max_el'], 's-',
                color=colors['Chebyshev'], linewidth=2, markersize=6)
        ax2.set_xlabel('Количество членов Чебышева (M)')
    ax2.set_ylabel('Макс. разность элементов')
    ax2.set_title('Chebyshev: точность по элементам')
    ax2.grid(True, alpha=0.3)
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax2.set_yscale('log')

    # График 3: Сравнение точности методов по элементам
    ax3 = plt.subplot(2, 3, 3)
    if taylor_data:
        ax3.plot(taylor_data['param'], taylor_data['max_el'], 'o-',
                label='Taylor', color=colors['Taylor'], linewidth=2, markersize=6)
    if cheb_data:
        ax3.plot(cheb_data['param'], cheb_data['max_el'], 's--',
                label='Chebyshev', color=colors['Chebyshev'], linewidth=2, markersize=6)
    ax3.set_xlabel('Количество членов (K/M)')
    ax3.set_ylabel('Макс. разность элементов')
    ax3.set_title('Сравнение: точность по элементам')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax3.set_yscale('log')

    # График 4: Точность Taylor метода по собственным значениям
    ax4 = plt.subplot(2, 3, 4)
    if taylor_data:
        ax4.plot(taylor_data['param'], taylor_data['max_eig'], 'o-',
                color=colors['Taylor'], linewidth=2, markersize=6)
        ax4.set_xlabel('Количество членов ряда Тейлора (K)')
    ax4.set_ylabel('Макс. разность собств. значений')
    ax4.set_title('Taylor: точность по собств. значениям')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax4.set_yscale('log')

    # График 5: Точность Chebyshev метода по собственным значениям
    ax5 = plt.subplot(2, 3, 5)
    if cheb_data:
        ax5.plot(cheb_data['param'], cheb_data['max_eig'], 's-',
                color=colors['Chebyshev'], linewidth=2, markersize=6)
        ax5.set_xlabel('Количество членов Чебышева (M)')
    ax5.set_ylabel('Макс. разность собств. значений')
    ax5.set_title('Chebyshev: точность по собств. значениям')
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    ax5.set_yscale('log')

    # График 6: Сравнение производительности
    ax6 = plt.subplot(2, 3, 6)
    if taylor_data:
        ax6.plot(taylor_data['param'], taylor_data['time'], 'o-',
                label='Taylor', color=colors['Taylor'], linewidth=2, markersize=6)
    if cheb_data:
        ax6.plot(cheb_data['param'], cheb_data['time'], 's--',
                label='Chebyshev', color=colors['Chebyshev'], linewidth=2, markersize=6)
    ax6.set_xlabel('Количество членов (K/M)')
    ax6.set_ylabel('Время выполнения (мс)')
    ax6.set_title('Сравнение производительности')
    ax6.grid(True, alpha=0.3)
    ax6.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('experiment2_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment2_analysis.png")
    plt.show()

    # Дополнительные графики: компромисс точность-время
    fig2, (ax7, ax8) = plt.subplots(1, 2, figsize=(15, 6))
    fig2.suptitle('Эксперимент 2: Компромисс точность-производительность', fontsize=14, fontweight='bold')

    # График 7: Элементы - время vs точность
    ax7 = plt.subplot(1, 2, 1)
    if taylor_data:
        safe_times = np.array(taylor_data['time'])
        safe_errors = safe_log_values(taylor_data['max_el'])
        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax7.scatter(safe_times[mask], safe_errors[mask],
                       label='Taylor', color=colors['Taylor'], s=80, alpha=0.8, marker='o')
            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax7.plot(safe_times[mask][sorted_indices], safe_errors[mask][sorted_indices],
                        '-', color=colors['Taylor'], alpha=0.6, linewidth=2)

    if cheb_data:
        safe_times = np.array(cheb_data['time'])
        safe_errors = safe_log_values(cheb_data['max_el'])
        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax7.scatter(safe_times[mask], safe_errors[mask],
                       label='Chebyshev', color=colors['Chebyshev'], s=80, alpha=0.8, marker='s')
            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax7.plot(safe_times[mask][sorted_indices], safe_errors[mask][sorted_indices],
                        '--', color=colors['Chebyshev'], alpha=0.6, linewidth=2)

    ax7.set_xlabel('Время выполнения (мс)')
    ax7.set_ylabel('Макс. разность элементов')
    ax7.set_title('Компромисс: время vs точность (элементы)')
    ax7.set_xscale('log')
    ax7.set_yscale('log')
    ax7.grid(True, alpha=0.3)
    ax7.legend(fontsize=10)

    # График 8: Собственные значения - время vs точность
    ax8 = plt.subplot(1, 2, 2)
    if taylor_data:
        safe_times = np.array(taylor_data['time'])
        safe_errors = safe_log_values(taylor_data['max_eig'])
        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax8.scatter(safe_times[mask], safe_errors[mask],
                       label='Taylor', color=colors['Taylor'], s=80, alpha=0.8, marker='o')
            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax8.plot(safe_times[mask][sorted_indices], safe_errors[mask][sorted_indices],
                        '-', color=colors['Taylor'], alpha=0.6, linewidth=2)

    if cheb_data:
        safe_times = np.array(cheb_data['time'])
        safe_errors = safe_log_values(cheb_data['max_eig'])
        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax8.scatter(safe_times[mask], safe_errors[mask],
                       label='Chebyshev', color=colors['Chebyshev'], s=80, alpha=0.8, marker='s')
            if np.sum(mask) > 1:
                sorted_indices = np.argsort(safe_times[mask])
                ax8.plot(safe_times[mask][sorted_indices], safe_errors[mask][sorted_indices],
                        '--', color=colors['Chebyshev'], alpha=0.6, linewidth=2)

    ax8.set_xlabel('Время выполнения (мс)')
    ax8.set_ylabel('Макс. разность собств. значений')
    ax8.set_title('Компромисс: время vs точность (собств. значения)')
    ax8.set_xscale('log')
    ax8.set_yscale('log')
    ax8.grid(True, alpha=0.3)
    ax8.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('experiment2_efficiency.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved experiment2_efficiency.png")
    plt.show()

    # Final statistics
    print("\n=== Experiment 2 Statistics ===")
    print("Matrix exponentiation methods comparison for anti-Hermitian matrix Ω = -i×H")
    print("Reference: Eigen Pade approximation")

    for method, data in results.items():
        if not data['param']:
            continue

        print(f"\n{method}:")
        if method in ['Taylor', 'Chebyshev']:
            param_name = 'K' if method == 'Taylor' else 'M'
            print(f"  {param_name}: from {min(data['param'])} to {max(data['param'])}")

        print(".2e")
        print(".2e")
        print(".2f")

        # Лучший результат
        min_el_idx = np.argmin(data['max_el'])
        min_eig_idx = np.argmin(data['max_eig'])
        print(".2e")
        print(".2e")

def main():
    print("Analysis of Experiment 2 results: Matrix exponentiation methods")
    print("=" * 55)

    filename = "experiment2_results.txt"

    try:
        results = parse_experiment2_results(filename)

        if results:
            print(f"Found {len(results)} methods for analysis:")
            for method in results.keys():
                print(f"  - {method}")

            plot_experiment2(results)
        else:
            print("Experiment 2 data not found in file!")

    except FileNotFoundError:
        print(f"Error: file {filename} not found!")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
