import matplotlib.pyplot as plt
import numpy as np
import re

def parse_results(filename):
    """Парсинг всех результатов из файла"""
    with open(filename, 'r') as f:
        content = f.read()
    
    results = {'exp1': {}, 'exp3': {'Taylor': [], 'Cheb': []}}
    
    # --- Парсинг эксперимента 1 ---
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
                    
                    if method not in results['exp1']:
                        results['exp1'][method] = {'k': [], 'max_el': [], 'max_eig': [], 'time': []}
                    
                    results['exp1'][method]['k'].append(k)
                    results['exp1'][method]['max_el'].append(max_el)
                    results['exp1'][method]['max_eig'].append(max_eig)
                    results['exp1'][method]['time'].append(time)
    
    # --- Парсинг эксперимента 3 ---
    exp3_start = content.find("EXPERIMENT 3:")
    
    if exp3_start != -1:
        exp3_content = content[exp3_start:]
        lines = exp3_content.strip().split('\n')
        
        in_table = False
        for line in lines:
            if "Method | K/M | Max Elem Diff | Max Eig Diff | Time (ms)" in line:
                in_table = True
                continue
            
            if in_table and "---" not in line and line.strip() and not line.startswith("="):
                parts = line.split('|')
                if len(parts) >= 5:
                    method = parts[0].strip()
                    km = int(parts[1].strip())
                    max_el = float(parts[2].strip())
                    max_eig = float(parts[3].strip())
                    time = float(parts[4].strip())
                    
                    if method == "Taylor":
                        results['exp3']['Taylor'].append({
                            'k': km, 'max_el': max_el, 'max_eig': max_eig, 'time': time
                        })
                    elif method == "Cheb":
                        results['exp3']['Cheb'].append({
                            'm': km, 'max_el': max_el, 'max_eig': max_eig, 'time': time
                        })
    
    return results

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

def plot_experiment1(results):
    """Графики для эксперимента 1"""
    if not results['exp1']:
        print("Нет данных для эксперимента 1")
        return
    
    # Создаем фигуру с меньшим размером
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Эксперимент 1: Сравнение методов Magnus', fontsize=14, fontweight='bold')
    
    # Цвета для разных методов
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown']
    
    # График 1: Точность (макс. разность элементов) - ЛИНЕЙНАЯ шкала
    ax1 = plt.subplot(2, 2, 1)
    for i, (method, data) in enumerate(results['exp1'].items()):
        color = colors[i % len(colors)]
        ax1.plot(data['k'], data['max_el'], 'o-', label=method, 
                color=color, linewidth=2, markersize=5)
    
    ax1.set_xlabel('Порядок (K)')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность: разность элементов (линейная шкала)')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # График 2: Точность (собственные значения) - ЛИНЕЙНАЯ шкала
    ax2 = plt.subplot(2, 2, 2)
    for i, (method, data) in enumerate(results['exp1'].items()):
        color = colors[i % len(colors)]
        ax2.plot(data['k'], data['max_eig'], 's--', label=method,
                color=color, linewidth=2, markersize=5)
    
    ax2.set_xlabel('Порядок (K)')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность: разность собственных значений (линейная шкала)')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8, loc='best')
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # График 3: Время выполнения
    ax3 = plt.subplot(2, 2, 3)
    for i, (method, data) in enumerate(results['exp1'].items()):
        color = colors[i % len(colors)]
        ax3.plot(data['k'], data['time'], '^-', label=method,
                color=color, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Порядок (K)')
    ax3.set_ylabel('Время (мс)')
    ax3.set_title('Время выполнения')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc='best')
    
    # График 4: Компромисс точность-время (с безопасными значениями)
    ax4 = plt.subplot(2, 2, 4)
    for i, (method, data) in enumerate(results['exp1'].items()):
        color = colors[i % len(colors)]
        # Используем минимальную ошибку
        min_error = np.minimum(data['max_el'], data['max_eig'])
        
        # Безопасное преобразование для логарифмической шкалы
        safe_times = np.array(data['time'])
        safe_errors = safe_log_values(min_error)
        
        # Отфильтруем нулевые и очень маленькие значения
        mask = (safe_times > 0) & (safe_errors > 0)
        if np.any(mask):
            ax4.scatter(safe_times[mask], safe_errors[mask], 
                       label=method, color=color, s=50, alpha=0.7)
            
            # Соединяем точки линиями только если есть более одной точки
            if np.sum(mask) > 1:
                # Сортируем по времени для красивого отображения
                sorted_indices = np.argsort(safe_times[mask])
                ax4.plot(safe_times[mask][sorted_indices], 
                        safe_errors[mask][sorted_indices], 
                        '--', color=color, alpha=0.5, linewidth=1)
    
    ax4.set_xlabel('Время (мс)')
    ax4.set_ylabel('Минимальная ошибка')
    ax4.set_title('Компромисс точность-время (логарифмическая шкала)')
    ax4.set_xscale('log')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8, loc='best')
    
    plt.tight_layout()
    plt.savefig('experiment1.png', dpi=150, bbox_inches='tight')
    print("Сохранен experiment1.png")
    plt.show()
    
    # Дополнительный график: сравнение методов - БЕЗ логарифмической шкалы
    fig2, (ax5, ax6) = plt.subplots(1, 2, figsize=(12, 5))
    fig2.suptitle('Эксперимент 1: Сравнение методов', fontsize=12, fontweight='bold')
    
    # Максимальный порядок для каждого метода
    methods = list(results['exp1'].keys())
    max_errors = []
    max_times = []
    
    for method in methods:
        data = results['exp1'][method]
        last_idx = -1
        max_errors.append(min(data['max_el'][last_idx], data['max_eig'][last_idx]))
        max_times.append(data['time'][last_idx])
    
    x = range(len(methods))
    
    # Точность при максимальном порядке - линейная шкала
    bars1 = ax5.bar(x, max_errors, color=colors[:len(methods)], alpha=0.7)
    ax5.set_xlabel('Метод')
    ax5.set_ylabel('Минимальная ошибка')
    ax5.set_title('Точность при максимальном K')
    ax5.set_xticks(x)
    ax5.set_xticklabels(methods, rotation=45, ha='right')
    ax5.grid(True, alpha=0.3, axis='y')
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # Добавляем значения на столбцы
    for bar in bars1:
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{height:.2e}', ha='center', va='bottom', fontsize=8)
    
    # Время при максимальном порядке
    bars2 = ax6.bar(x, max_times, color=colors[:len(methods)], alpha=0.7)
    ax6.set_xlabel('Метод')
    ax6.set_ylabel('Время (мс)')
    ax6.set_title('Время при максимальном K')
    ax6.set_xticks(x)
    ax6.set_xticklabels(methods, rotation=45, ha='right')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения на столбцы
    for bar in bars2:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height*1.05,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('experiment1_comparison.png', dpi=150, bbox_inches='tight')
    print("Сохранен experiment1_comparison.png")
    plt.show()

def plot_experiment3(results):
    """Графики для эксперимента 3"""
    if not results['exp3']['Taylor'] or not results['exp3']['Cheb']:
        print("Нет данных для эксперимента 3")
        return
    
    # Подготовка данных
    taylor = results['exp3']['Taylor']
    cheb = results['exp3']['Cheb']
    
    taylor_k = [d['k'] for d in taylor]
    taylor_el = [d['max_el'] for d in taylor]
    taylor_eig = [d['max_eig'] for d in taylor]
    taylor_time = [d['time'] for d in taylor]
    
    cheb_m = [d['m'] for d in cheb]
    cheb_el = [d['max_el'] for d in cheb]
    cheb_eig = [d['max_eig'] for d in cheb]
    cheb_time = [d['time'] for d in cheb]
    
    # Создаем фигуру
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Эксперимент 3: Сравнение методов вычисления exp(Ω)', 
                 fontsize=14, fontweight='bold')
    
    # График 1: Точность по элементам - ЛИНЕЙНАЯ шкала
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(taylor_k, taylor_el, 'bo-', label='Taylor', linewidth=2, markersize=5)
    ax1.plot(cheb_m, cheb_el, 'rs--', label='Chebyshev', linewidth=2, markersize=5)
    
    ax1.set_xlabel('Количество членов (K/M)')
    ax1.set_ylabel('Макс. разность элементов')
    ax1.set_title('Точность: разность элементов (линейная шкала)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # График 2: Точность по собственным значениям - ЛИНЕЙНАЯ шкала
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(taylor_k, taylor_eig, 'bo-', label='Taylor', linewidth=2, markersize=5)
    ax2.plot(cheb_m, cheb_eig, 'rs--', label='Chebyshev', linewidth=2, markersize=5)
    
    ax2.set_xlabel('Количество членов (K/M)')
    ax2.set_ylabel('Макс. разность собств. значений')
    ax2.set_title('Точность: разность собственных значений (линейная шкала)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # График 3: Время выполнения
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(taylor_k, taylor_time, 'bo-', label='Taylor', linewidth=2, markersize=5)
    ax3.plot(cheb_m, cheb_time, 'rs--', label='Chebyshev', linewidth=2, markersize=5)
    
    ax3.set_xlabel('Количество членов (K/M)')
    ax3.set_ylabel('Время (мс)')
    ax3.set_title('Время выполнения')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # График 4: Метод Тейлора отдельно
    ax4 = plt.subplot(2, 3, 4)
    ax4.plot(taylor_k, taylor_el, 'bo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Члены Тейлора (K)')
    ax4.set_ylabel('Макс. разность элементов')
    ax4.set_title('Метод Тейлора')
    ax4.grid(True, alpha=0.3)
    ax4.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # Отмечаем точки
    for i, (k, err) in enumerate(zip(taylor_k, taylor_el)):
        if i % 2 == 0:  # Отмечаем каждую вторую точку
            ax4.annotate(f'K={k}', (k, err), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    # График 5: Метод Чебышева отдельно
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(cheb_m, cheb_el, 'rs-', linewidth=2, markersize=6)
    ax5.set_xlabel('Члены Чебышева (M)')
    ax5.set_ylabel('Макс. разность элементов')
    ax5.set_title('Метод Чебышева')
    ax5.grid(True, alpha=0.3)
    ax5.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    
    # Отмечаем точки
    for i, (m, err) in enumerate(zip(cheb_m, cheb_el)):
        if i % 5 == 0:  # Отмечаем каждую пятую точку
            ax5.annotate(f'M={m}', (m, err), xytext=(5, 5), 
                        textcoords='offset points', fontsize=8)
    
    # График 6: Компромисс точность-время (с безопасными значениями)
    ax6 = plt.subplot(2, 3, 6)
    # Используем минимальную ошибку
    taylor_min = np.minimum(taylor_el, taylor_eig)
    cheb_min = np.minimum(cheb_el, cheb_eig)
    
    # Безопасное преобразование
    safe_taylor_times = np.array(taylor_time)
    safe_taylor_errors = safe_log_values(taylor_min)
    safe_cheb_times = np.array(cheb_time)
    safe_cheb_errors = safe_log_values(cheb_min)
    
    # Отфильтруем нулевые значения
    taylor_mask = (safe_taylor_times > 0) & (safe_taylor_errors > 0)
    cheb_mask = (safe_cheb_times > 0) & (safe_cheb_errors > 0)
    
    if np.any(taylor_mask):
        ax6.scatter(safe_taylor_times[taylor_mask], safe_taylor_errors[taylor_mask], 
                   label='Taylor', color='blue', s=50, alpha=0.7)
        if np.sum(taylor_mask) > 1:
            sorted_idx = np.argsort(safe_taylor_times[taylor_mask])
            ax6.plot(safe_taylor_times[taylor_mask][sorted_idx], 
                    safe_taylor_errors[taylor_mask][sorted_idx], 
                    'b-', alpha=0.5, linewidth=1)
    
    if np.any(cheb_mask):
        ax6.scatter(safe_cheb_times[cheb_mask], safe_cheb_errors[cheb_mask], 
                   label='Chebyshev', color='red', s=50, alpha=0.7)
        if np.sum(cheb_mask) > 1:
            sorted_idx = np.argsort(safe_cheb_times[cheb_mask])
            ax6.plot(safe_cheb_times[cheb_mask][sorted_idx], 
                    safe_cheb_errors[cheb_mask][sorted_idx], 
                    'r--', alpha=0.5, linewidth=1)
    
    ax6.set_xlabel('Время (мс)')
    ax6.set_ylabel('Минимальная ошибка')
    ax6.set_title('Компромисс точность-время')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.tight_layout()
    plt.savefig('experiment3.png', dpi=150, bbox_inches='tight')
    print("Сохранен experiment3.png")
    plt.show()
    
    # Дополнительный график: сравнение методов
    fig2, (ax7, ax8, ax9) = plt.subplots(1, 3, figsize=(14, 5))
    fig2.suptitle('Эксперимент 3: Прямое сравнение методов', fontsize=12, fontweight='bold')
    
    # Берем последние 5 точек для сравнения
    n_points = min(5, len(taylor), len(cheb))
    
    # Точность последних точек
    taylor_last_el = taylor_el[-n_points:]
    cheb_last_el = cheb_el[-n_points:]
    
    x_pos = np.arange(n_points)
    ax7.bar(x_pos - 0.2, taylor_last_el, width=0.4, label='Taylor', 
            alpha=0.7, color='blue')
    ax7.bar(x_pos + 0.2, cheb_last_el, width=0.4, label='Chebyshev', 
            alpha=0.7, color='red')
    
    ax7.set_xlabel('Последние измерения')
    ax7.set_ylabel('Макс. разность элементов')
    ax7.set_title('Точность последних измерений')
    ax7.set_xticks(x_pos)
    ax7.set_xticklabels([f'#{i+1}' for i in range(n_points)])
    ax7.legend()
    ax7.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения
    for i, (t_val, c_val) in enumerate(zip(taylor_last_el, cheb_last_el)):
        ax7.text(i - 0.2, t_val*1.05, f'{t_val:.2e}', ha='center', va='bottom', fontsize=7)
        ax7.text(i + 0.2, c_val*1.05, f'{c_val:.2e}', ha='center', va='bottom', fontsize=7)
    
    # Время последних точек
    taylor_last_time = taylor_time[-n_points:]
    cheb_last_time = cheb_time[-n_points:]
    
    ax8.bar(x_pos - 0.2, taylor_last_time, width=0.4, label='Taylor', 
            alpha=0.7, color='blue')
    ax8.bar(x_pos + 0.2, cheb_last_time, width=0.4, label='Chebyshev', 
            alpha=0.7, color='red')
    
    ax8.set_xlabel('Последние измерения')
    ax8.set_ylabel('Время (мс)')
    ax8.set_title('Время последних измерений')
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels([f'#{i+1}' for i in range(n_points)])
    ax8.legend()
    ax8.grid(True, alpha=0.3, axis='y')
    
    # Добавляем значения
    for i, (t_val, c_val) in enumerate(zip(taylor_last_time, cheb_last_time)):
        ax8.text(i - 0.2, t_val*1.05, f'{t_val:.1f}', ha='center', va='bottom', fontsize=8)
        ax8.text(i + 0.2, c_val*1.05, f'{c_val:.1f}', ha='center', va='bottom', fontsize=8)
    
    # Эффективность (ошибка/время) - меньше лучше
    ax9.bar(x_pos - 0.2, [e/t for e, t in zip(taylor_last_el, taylor_last_time)], 
            width=0.4, label='Taylor', alpha=0.7, color='blue')
    ax9.bar(x_pos + 0.2, [e/t for e, t in zip(cheb_last_el, cheb_last_time)], 
            width=0.4, label='Chebyshev', alpha=0.7, color='red')
    
    ax9.set_xlabel('Последние измерения')
    ax9.set_ylabel('Ошибка / Время')
    ax9.set_title('Эффективность (меньше = лучше)')
    ax9.set_xticks(x_pos)
    ax9.set_xticklabels([f'#{i+1}' for i in range(n_points)])
    ax9.legend()
    ax9.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiment3_comparison.png', dpi=150, bbox_inches='tight')
    print("Сохранен experiment3_comparison.png")
    plt.show()

def main():
    """Основная функция"""
    print("Анализ результатов экспериментов")
    print("=" * 40)
    
    filename = "magnus_experiment_results.txt"
    
    try:
        # Чтение и парсинг результатов
        print(f"Чтение файла: {filename}")
        results = parse_results(filename)
        
        # Статистика
        print(f"\nЭксперимент 1: найдено {len(results['exp1'])} методов")
        for method, data in results['exp1'].items():
            print(f"  - {method}: {len(data['k'])} точек, K от {min(data['k'])} до {max(data['k'])}")
            print(f"    Ошибки: мин={min(data['max_el']):.2e}, макс={max(data['max_el']):.2e}")
            print(f"    Время: мин={min(data['time']):.2f} мс, макс={max(data['time']):.2f} мс")
        
        print(f"\nЭксперимент 3:")
        print(f"  - Taylor: {len(results['exp3']['Taylor'])} точек, K от {min([d['k'] for d in results['exp3']['Taylor']])} до {max([d['k'] for d in results['exp3']['Taylor']])}")
        print(f"  - Chebyshev: {len(results['exp3']['Cheb'])} точек, M от {min([d['m'] for d in results['exp3']['Cheb']])} до {max([d['m'] for d in results['exp3']['Cheb']])}")
        
        # Построение графиков
        print("\nПостроение графиков...")
        
        if results['exp1']:
            print("Графики для эксперимента 1...")
            plot_experiment1(results)
        
        if results['exp3']['Taylor'] and results['exp3']['Cheb']:
            print("Графики для эксперимента 3...")
            plot_experiment3(results)
        
        print("\nГотово! Созданы файлы:")
        print("  - experiment1.png")
        print("  - experiment1_comparison.png")
        print("  - experiment3.png")
        print("  - experiment3_comparison.png")
        
    except FileNotFoundError:
        print(f"Ошибка: файл {filename} не найден!")
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()