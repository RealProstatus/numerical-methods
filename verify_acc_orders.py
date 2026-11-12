import numpy as np
import pymagnus_hpc

# 1. Генерация некоммутирующей тестовой системы N = 4
N = 4
rng = np.random.default_rng(42)
A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
H0 = 0.5 * (A + A.conj().T)
H1 = 0.5 * (B + B.conj().T)

eps = 0.5
omega = 2.0 * np.pi

print("=" * 80)
print(" ЧАСТЬ 1: ЛОКАЛЬНЫЙ ШАГ (h = 0.01, steps = 1)")
print("=" * 80)

t0, t1 = 0.0, 0.05
# Сверхточный эталон на мелкой сетке
u_ref_local = pymagnus_hpc.solve_magnus(
    H0, H1, t0, t1, steps=1000,
    epsilon=eps, omega=omega,
    backend="Cpu", magnus_method="Formula314_6"
)

print(f"{'Метод / Порядок':<20} | {'Погрешность ||U - U_ref||':<25} | {'Дефект унитарности':<20}")
print("-" * 80)

for method in ["Classic2", "Classic4", "Acc4", "Acc6", "Formula314_6"]:
    u_test = pymagnus_hpc.solve_magnus(
        H0, H1, t0, t1, steps=1,
        epsilon=eps, omega=omega,
        backend="Cpu", magnus_method=method
    )
    diff = np.max(np.abs(u_test - u_ref_local))
    unitarity = np.linalg.norm(u_test.conj().T @ u_test - np.eye(N))
    print(f"{method:<20} | {diff:<25.16e} | {unitarity:<20.16e}")

print("\n" + "=" * 80)
print(" ЧАСТЬ 2: КУСОЧНЫЙ МЕТОД (Piecewise Magnus, T = 1.0)")
print("=" * 80)

T_final = 1.0
# Эталон траектории на T = 1.0 (5000 шагов 6-го порядка)
print("Вычисление эталона U_ref(T=1.0) на сетке 5000 шагов...")
u_ref_global = pymagnus_hpc.solve_magnus(
    H0, H1, 0.0, T_final, steps=5000,
    epsilon=eps, omega=omega,
    backend="Cpu", magnus_method="Formula314_6"
)

print("-" * 80)
print(f"{'Шаги (macro_dt)':<20} | {'Метод':<15} | {'Погрешность U(T)':<20} | {'Дефект унитарности':<20}")
print("-" * 80)

for steps in [10, 20, 50, 100]:
    macro_dt = T_final / steps
    for method in ["Acc4", "Acc6", "Formula314_6"]:
        u_piecewise = pymagnus_hpc.solve_magnus(
            H0, H1, 0.0, T_final, steps=steps,
            epsilon=eps, omega=omega,
            backend="Cpu", magnus_method=method
        )
        diff = np.max(np.abs(u_piecewise - u_ref_global))
        unitarity = np.linalg.norm(u_piecewise.conj().T @ u_piecewise - np.eye(N))
        print(f"N={steps:<4} (dt={macro_dt:.3f})   | {method:<15} | {diff:<20.2e} | {unitarity:<20.2e}")
    print("-" * 80)