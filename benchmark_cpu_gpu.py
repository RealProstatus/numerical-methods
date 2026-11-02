import os
import sys
import time
import ctypes
import numpy as np
import pymagnus_hpc

# ==============================================================================
# Утилиты управления потоками Intel MKL / OpenMP
# ==============================================================================

def set_mkl_openmp_threads(num_threads: int):
    """
    Устанавливает количество потоков для Intel MKL и OpenMP runtime.
    1 поток = строго последовательное выполнение на CPU.
    """
    for dll_name in ["mkl_intel_thread.2.dll", "libiomp5md.dll", "mkl_rt.2.dll"]:
        try:
            lib = ctypes.CDLL(dll_name)
            if hasattr(lib, "mkl_set_num_threads"):
                lib.mkl_set_num_threads(ctypes.c_int(num_threads))
            if hasattr(lib, "omp_set_num_threads"):
                lib.omp_set_num_threads(ctypes.c_int(num_threads))
        except Exception:
            pass

def generate_random_hermitian(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Генерация плотной случайной эрмитовой матрицы."""
    a = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    return 0.5 * (a + a.conj().T)

# ==============================================================================
# 1. Измерение накладных расходов интерфейса Python <-> C++
# ==============================================================================

def measure_bridge_overhead(dim: int, rng: np.random.Generator, repeats: int = 100):
    """
    Оценивает накладные расходы моста nanobind + аллокации памяти.
    Использует регрессию T(steps) = a * steps + T_bridge.
    При steps -> 0 остается только время вызова C++ и копирования буферов.
    """
    H0 = generate_random_hermitian(dim, rng)
    H1 = generate_random_hermitian(dim, rng)
    
    test_steps = [1, 2, 4, 8, 16]
    mean_times = []
    
    for s in test_steps:
        # Прогрев
        _ = pymagnus_hpc.solve_magnus(H0, H1, 0.0, 0.01 * s, s, backend="Cpu")
        
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = pymagnus_hpc.solve_magnus(H0, H1, 0.0, 0.01 * s, s, backend="Cpu")
        t_avg = (time.perf_counter() - t0) / repeats * 1e6 # в микросекундах (мкс)
        mean_times.append(t_avg)
    
    # Линейный МНК: y = a*x + b, где b = T_bridge
    x = np.array(test_steps, dtype=np.float64)
    y = np.array(mean_times, dtype=np.float64)
    a, b = np.polyfit(x, y, 1)
    t_bridge_us = max(0.0, b)
    t_step_us = max(0.0, a)
    
    # Полное время при 100 шагах
    t_100_us = t_bridge_us + 100 * t_step_us
    overhead_pct = (t_bridge_us / t_100_us) * 100.0 if t_100_us > 0 else 0.0
    
    return t_bridge_us, t_step_us, overhead_pct

# ==============================================================================
# Основной сценарий
# ==============================================================================

def main():
    rng = np.random.default_rng(42)
    t_start, t_final = 0.0, 1.0
    steps = 100
    eps = 0.2
    omega = 2.0 * np.pi
    max_cpu_threads = os.cpu_count() or 8

    print("=" * 85)
    print(" 1. НАКЛАДНЫЕ РАСХОДЫ ИНТЕРФЕЙСА PYTHON <-> C++ (NANOBIND / MEMCPY)")
    print("=" * 85)
    print("Оценка методом линейной экстраполяции T(steps) = T_bridge + steps * T_step:")
    print(f"{'Размер N':<10} | {'T_bridge (мкс)':<18} | {'T_step ядра (мкс)':<20} | {'Доля моста при S=100':<22}")
    print("-" * 85)

    for dim in [2, 8, 32, 64, 128, 256]:
        t_bridge, t_step, pct = measure_bridge_overhead(dim, rng, repeats=50 if dim <= 64 else 15)
        print(f"{dim:<10} | {t_bridge:<18.2f} | {t_step:<20.2f} | {pct:<21.2f}%")

    print("\n" + "=" * 85)
    print(" 2. СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ: SEQ CPU vs PAR CPU vs GPU (RTX 4060)")
    print("=" * 85)
    print(f"{'N':<5} | {'CPU (1 поток)':<15} | {'CPU (All cores)':<16} | {'GPU (CUDA)':<12} | {'Ускор. OpenMP':<14} | {'Ускор. GPU'}")
    print("-" * 85)

    # Прогрев GPU
    H0_dummy = generate_random_hermitian(2, rng)
    H1_dummy = generate_random_hermitian(2, rng)
    _ = pymagnus_hpc.solve_magnus(H0_dummy, H1_dummy, t_start, t_final, steps, epsilon=eps, omega=omega, backend="Cuda")

    dimensions = [2, 4, 8, 16, 32, 64, 128, 256]

    for dim in dimensions:
        H0 = generate_random_hermitian(dim, rng)
        H1 = generate_random_hermitian(dim, rng)
        repeats = 10 if dim <= 64 else 3

        # 1. Последовательный CPU (1 поток MKL)
        set_mkl_openmp_threads(1)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = pymagnus_hpc.solve_magnus(H0, H1, t_start, t_final, steps, epsilon=eps, omega=omega, backend="Cpu")
        t_seq = (time.perf_counter() - t0) / repeats * 1000.0

        # 2. Параллельный CPU (все потоки MKL OpenMP)
        set_mkl_openmp_threads(max_cpu_threads)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = pymagnus_hpc.solve_magnus(H0, H1, t_start, t_final, steps, epsilon=eps, omega=omega, backend="Cpu")
        t_par = (time.perf_counter() - t0) / repeats * 1000.0

        # 3. GPU (CUDA / cuBLAS)
        t0 = time.perf_counter()
        for _ in range(repeats):
            _ = pymagnus_hpc.solve_magnus(H0, H1, t_start, t_final, steps, epsilon=eps, omega=omega, backend="Cuda")
        t_gpu = (time.perf_counter() - t0) / repeats * 1000.0

        # Метрики ускорения
        speedup_omp = t_seq / t_par if t_par > 0 else 1.0
        speedup_gpu_vs_seq = t_seq / t_gpu if t_gpu > 0 else 1.0
        speedup_gpu_vs_par = t_par / t_gpu if t_gpu > 0 else 1.0

        gpu_desc = f"{speedup_gpu_vs_par:.2f}x (к OpenMP)" if speedup_gpu_vs_par >= 1.0 else f"{speedup_gpu_vs_par:.2f}x"

        print(f"{dim:<5} | {t_seq:<12.2f} мс | {t_par:<13.2f} мс | {t_gpu:<9.2f} мс | {speedup_omp:<11.2f}x | {gpu_desc}")

    print("=" * 85)
    
    # Возвращаем системные потоки в исходное состояние
    set_mkl_openmp_threads(max_cpu_threads)

if __name__ == "__main__":
    main()