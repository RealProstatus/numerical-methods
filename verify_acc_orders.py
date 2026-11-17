import numpy as np
import pymagnus_hpc

# 1. Генерация эрмитовых матриц N = 100 со строгим диагональным преобладанием
N = 100
rng = np.random.default_rng(42)
A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))

H0_raw = 0.5 * (A + A.conj().T)
H1_raw = 0.5 * (B + B.conj().T)

def make_diagonally_dominant(H: np.ndarray, margin: float = 10.0, scale: float = 0.05) -> np.ndarray:
    H_off = H.copy()
    np.fill_diagonal(H_off, 0.0)
    off_row_sums = np.sum(np.abs(H_off), axis=1)
    # Гарантируем |H_ii| >= sum_{j != i} |H_ij| + margin в каждой строке
    diag_vals = np.maximum(np.abs(np.diag(H)), off_row_sums + margin)
    H_dom = H_off + np.diag(diag_vals)
    # Масштабируем спектр, чтобы ||H|| * dt < pi на грубых сетках (steps >= 10)
    return np.ascontiguousarray(scale * H_dom, dtype=np.complex128)

H0 = make_diagonally_dominant(H0_raw, margin=10.0, scale=0.05)
H1 = make_diagonally_dominant(H1_raw, margin=5.0, scale=0.02)

# Быстрая проверка эрмитовости и диагонального преобладания
assert np.allclose(H0, H0.conj().T) and np.allclose(H1, H1.conj().T)
off0 = np.sum(np.abs(H0 - np.diag(np.diag(H0))), axis=1)
assert np.all(np.abs(np.diag(H0)) > off0), "H0 не обладает строгим диагональным преобладанием"

eps = 0.5
omega = 2.0 * np.pi

print("=" * 80)
print(f" ЧАСТЬ 1: ЛОКАЛЬНЫЙ ШАГ (N = {N}x{N}, diag-dominant, h = 0.05, steps = 1)")
print("=" * 80)

t0, t1 = 0.0, 0.05
u_ref_local = pymagnus_hpc.solve_magnus(
    H0, H1, t0, t1, steps=1000,
    epsilon=eps, omega=omega,
    backend="Cpu", magnus_method="Formula314_6"
)

print(f"{'Метод / Порядок':<20} | {'Погрешность ||U - U_ref||':<25} | {'Дефект унитарности':<20}")
print("-" * 80)

for method in ["Classic2", "Classic4", "Acc4", "Acc6", "Formula314_6", "Recursive"]:
    u_test = pymagnus_hpc.solve_magnus(
        H0, H1, t0, t1, steps=1,
        epsilon=eps, omega=omega,
        backend="Cpu", magnus_method=method
    )
    diff = np.max(np.abs(u_test - u_ref_local))
    unitarity = np.linalg.norm(u_test.conj().T @ u_test - np.eye(N))
    print(f"{method:<20} | {diff:<25.16e} | {unitarity:<20.16e}")

print("\n" + "=" * 80)
print(f" ЧАСТЬ 2: КУСОЧНЫЙ МЕТОД (Piecewise Magnus, N = {N}x{N}, T = 1.0)")
print("=" * 80)

T_final = 1.0
print("Вычисление эталона U_ref(T=1.0) на сетке 5000 шагов...")
u_ref_global = pymagnus_hpc.solve_magnus(
    H0, H1, 0.0, T_final, steps=5000,
    epsilon=eps, omega=omega,
    backend="Cpu", magnus_method="Formula314_6"
)

print("-" * 80)
print(f"{'Шаги (macro_dt)':<20} | {'Метод':<15} | {'Погрешность U(T)':<20} | {'Дефект унитарности':<20}")
print("-" * 80)

methods_global = ["Classic2", "Classic4", "Acc4", "Acc6", "Formula314_6", "Recursive"]
for steps in [10, 20, 50, 100]:
    macro_dt = T_final / steps
    for method in methods_global:
        u_piecewise = pymagnus_hpc.solve_magnus(
            H0, H1, 0.0, T_final, steps=steps,
            epsilon=eps, omega=omega,
            backend="Cpu", magnus_method=method
        )
        diff = np.max(np.abs(u_piecewise - u_ref_global))
        unitarity = np.linalg.norm(u_piecewise.conj().T @ u_piecewise - np.eye(N))
        print(f"steps={steps:<4} (dt={macro_dt:.3f}) | {method:<15} | {diff:<20.2e} | {unitarity:<20.2e}")
    print("-" * 80)