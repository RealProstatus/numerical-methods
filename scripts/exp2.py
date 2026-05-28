import os
import matplotlib.pyplot as plt
import re

# === ПАРАМЕТРЫ ДЛЯ НАСТРОЙКИ ===
FILE_PATH = 'experiment2_results.txt' # Убедитесь, что путь к файлу верный
SAVE_FIG = True
FIG_NAME = 'exp2_exponentiation.png'
FIG_SIZE = (8, 6)
FONT_SIZE = 14

COLOR_TAYLOR = '#8c564b'    # Коричневый
COLOR_CHEBYSHEV = '#9467bd' # Фиолетовый
LINE_WIDTH = 2
MARKER_SIZE = 8
# ===============================

plt.rcParams.update({'font.size': FONT_SIZE, 'axes.grid': True})

def extract_terms(param_str):
    """Безопасно извлекает число из строк 'terms=10', 'M=5' или '--'"""
    match = re.search(r'\d+', param_str)
    # Если цифры найдены - возвращаем их, иначе возвращаем 0
    return int(match.group()) if match else 0

def main():
    if not os.path.exists(FILE_PATH):
        print(f"Ошибка: Файл {FILE_PATH} не найден.")
        return

    taylor_terms, taylor_err = [], []
    cheb_terms, cheb_err = [], []

    with open(FILE_PATH, 'r') as f:
        for line in f:
            if '|' in line and 'Method' not in line and '----' not in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 4:
                    method = parts[0]
                    terms = extract_terms(parts[1]) # Используем безопасную функцию
                    error = float(parts[2])
                    
                    if error <= 0: # Для логарифмической шкалы нули недопустимы
                        error = 1e-18 # Заглушка (машинный ноль)
                    
                    if method == 'Taylor':
                        taylor_terms.append(terms)
                        taylor_err.append(error)
                    elif method == 'Chebyshev':
                        cheb_terms.append(terms)
                        cheb_err.append(error)

    plt.figure(figsize=FIG_SIZE)
    
    # Используем semilogy, так как по оси X (число членов ряда) линейная шкала
    plt.semilogy(taylor_terms, taylor_err, marker='o', linestyle='-', color=COLOR_TAYLOR, 
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Разложение Тейлора')
    
    plt.semilogy(cheb_terms, cheb_err, marker='s', linestyle='-', color=COLOR_CHEBYSHEV, 
                 linewidth=LINE_WIDTH, markersize=MARKER_SIZE, label='Полиномы Чебышева')
    
    plt.title('Сходимость методов вычисления матричной экспоненты (Exp 2)')
    plt.xlabel('Количество членов ряда (K / M)')
    plt.ylabel('Погрешность вычисления $e^{\\Omega}$')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()

    if SAVE_FIG:
        plt.savefig(FIG_NAME, dpi=300)
        print(f"График сохранен как {FIG_NAME}")
    plt.show()

if __name__ == "__main__":
    main()
