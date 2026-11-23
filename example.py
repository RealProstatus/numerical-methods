# # # import numpy as np
# # # import qutip as qt
# # # import pymagnus_hpc.qutip_plugin  # Регистрирует "magnus_chebyshev" в qt.SESolver

# # # # 1. Задаём гамильтониан системы
# # # omega_0 = 1.0   # Собственная частота кубита
# # # eps = 0.2       # Амплитуда внешнего поля
# # # omega = 1.0     # Частота внешнего поля (резонанс)

# # # H0 = 0.5 * omega_0 * qt.sigmaz()   # Статичная часть
# # # H1 = eps * qt.sigmax()             # Оператор взаимодействия с полем

# # # def drive_coeff(t, args=None):
# # #     return np.cos(omega * t)

# # # # Стандартный формат нестационарного гамильтониана в QuTiP: H0 + f(t) * H1
# # # H = [H0, [H1, drive_coeff]]

# # # # 2. Задаём начальное состояние и сетку времени для записи наблюдаемых
# # # psi0 = qt.basis(2, 0)                # Начальное состояние |0>
# # # tlist = np.linspace(0.0, 20.0, 101)  # 100 интервалов вывода результатов

# # # # 3. Запускаем эволюцию через стандартный qt.sesolve
# # # result = qt.sesolve(
# # #     H,
# # #     psi0,
# # #     tlist,
# # #     e_ops=[qt.sigmaz(), qt.sigmax()],   # Что измеряем в каждой точке tlist
# # #     options={
# # #         "method": "magnus_chebyshev",    # Обязательно: включаем наш интегратор
# # #         "nsteps": 5,                     # Сколько шагов Магнуса делать между точками tlist
# # #         "magnus_method": "Formula314_6", # Схема Магнуса 6-го порядка
# # #         "expm_method": "Chebyshev",      # Матричная экспонента через полиномы Чебышёва
# # #         "chebyshev_degree": 16,          # Степень полинома Чебышёва
# # #         "backend": "Cpu",                # Где считать: "Cpu", "Cuda" или "Auto"
# # #     },
# # # )

# # # print("Населённость <sigma_z> в конце:", result.expect[0][-1])


# # import numpy as np
# # import qutip as qt
# # import pymagnus_hpc.qutip_plugin  # Регистрирует "magnus_chebyshev" в qt.SESolver[cite: 10]

# # # 1. Задаём гамильтониан системы (осцилляции Раби)
# # omega_0 = 1.0   # Собственная частота кубита
# # eps = 0.2       # Амплитуда внешнего поля
# # omega = 1.0     # Частота внешнего поля

# # H0 = 0.5 * omega_0 * qt.sigmaz()
# # H1 = eps * qt.sigmax()

# # def drive_coeff(t, args=None):
# #     return np.cos(omega * t)

# # H = [H0, [H1, drive_coeff]]

# # # 2. Начальное состояние и сетка времени
# # psi0 = qt.basis(2, 0)
# # tlist = np.linspace(0.0, 20.0, 101)

# # # 3. Расчёт через наш солвер Магнуса–Чебышёва[cite: 10]
# # res_magnus = qt.sesolve(
# #     H,
# #     psi0,
# #     tlist,
# #     e_ops=[qt.sigmaz(), qt.sigmax()],
# #     options={
# #         "method": "magnus_chebyshev",
# #         "atol": 1e-13,
# #         "rtol": 1e-13,
# #         "nsteps": 100000,
# #         "magnus_method": "Formula314_6",  # Можно проверить также "Classic4", "Acc4", "Classic2"
# #         "expm_method": "Chebyshev",
# #         "chebyshev_degree": 16,
# #         "backend": "Cpu",
# #         "store_final_state": True,
# #     },
# # )

# # # 4. Эталонный расчёт встроенным решателем QuTiP (метод Вернера 7-го порядка)
# # res_qutip = qt.sesolve(
# #     H,
# #     psi0,
# #     tlist,
# #     e_ops=[qt.sigmaz(), qt.sigmax()],
# #     options={
# #         "method": "vern7",
# #         "atol": 1e-13,
# #         "rtol": 1e-13,
# #         "nsteps": 100000,
# #         "store_final_state": True,
# #     },
# # )

# # # 5. Сравнение результатов
# # sz_magnus = np.asarray(res_magnus.expect[0])
# # sz_qutip = np.asarray(res_qutip.expect[0])

# # abs_err_final = abs(sz_magnus[-1] - sz_qutip[-1])
# # max_err_traj = np.max(np.abs(sz_magnus - sz_qutip))
# # state_diff = (res_magnus.final_state - res_qutip.final_state).norm()

# # print(f"Magnus-Chebyshev <sigma_z>(T): {sz_magnus[-1]:.15f}")
# # print(f"QuTiP (vern7)    <sigma_z>(T): {sz_qutip[-1]:.15f}")
# # print("-" * 55)
# # print(f"Ошибка <sigma_z> в конечной точке : {abs_err_final:.3e}")
# # print(f"Макс. ошибка <sigma_z> по всей сетке: {max_err_traj:.3e}")
# # print(f"Норма разности векторов ||d_psi||  : {state_diff:.3e}")

# import time
# import numpy as np
# import qutip as qt
# from qutip.solver.integrator import IntegratorException

# import pymagnus_hpc.qutip_plugin  # регистрирует "magnus_chebyshev"
# from pymagnus_hpc.solver import has_cuda_backend

# ALL_MAGNUS = (
#     "Classic2",
#     "Classic4",
#     "Acc4",
#     "Formula314_4",
#     "Acc6",
#     "Formula314_6",
#     "Recursive",
# )
# ALL_EXPM = ("Chebyshev", "PadeEigen", "Taylor")


# def run_ref(H_evo, psi0, tlist):
#     t0 = time.perf_counter()
#     res = qt.sesolve(
#         H_evo,
#         psi0,
#         tlist,
#         options={
#             "method": "vern7",
#             "atol": 1e-14,
#             "rtol": 1e-14,
#             "nsteps": 200000,
#             "store_final_state": True,
#         },
#     )
#     return res.final_state, (time.perf_counter() - t0) * 1e3


# def run_magnus(H_evo, psi0, tlist, **opts):
#     base_opts = {
#         "method": "magnus_chebyshev",
#         "nsteps": 200000,
#         "store_final_state": True,
#     }
#     base_opts.update(opts)
#     solver = qt.SESolver(H_evo, options=base_opts)
#     t0 = time.perf_counter()
#     res = solver.run(psi0, tlist)
#     dt_ms = (time.perf_counter() - t0) * 1e3
#     integ = solver._integrator
#     return res.final_state, dt_ms, integ.total_steps, integ.rejected_steps


# def section(title: str):
#     print(f"\n{'=' * 92}\n{title}\n{'=' * 92}")


# # ---------------------------------------------------------------------------
# # 1. Все 7 методов Магнуса (одноканальный Раби, CPU)
# # ---------------------------------------------------------------------------
# section("1. ALL 7 MAGNUS METHODS (Single-channel Rabi, N=2, CPU)")
# H_rabi = qt.QobjEvo([0.5 * qt.sigmaz(), [0.2 * qt.sigmax(), lambda t: np.cos(t)]])
# psi0_2 = qt.basis(2, 0)
# tlist_rabi = np.linspace(0.0, 20.0, 101)
# ref_rabi, ref_ms = run_ref(H_rabi, psi0_2, tlist_rabi)
# print(f"Reference vern7 (1e-14): {ref_ms:.2f} ms")

# for method in ALL_MAGNUS:
#     print(f"\n--- {method} ---")
#     for tol in (1e-6, 1e-9, 1e-12):
#         psi, ms, st, rej = run_magnus(
#             H_rabi, psi0_2, tlist_rabi, magnus_method=method, atol=tol, rtol=tol
#         )
#         err = (psi - ref_rabi).norm()
#         print(f"  tol={tol:7.1e} | ||d_psi||={err:9.3e} | steps={st:6d} (rej={rej:3d}) | {ms:7.2f} ms")


# # ---------------------------------------------------------------------------
# # 2. Все 7 методов Магнуса x все 3 метода матричной экспоненты (CPU)
# # ---------------------------------------------------------------------------
# section("2. ALL 7 MAGNUS x ALL 3 EXPM METHODS (Rabi, tol=1e-9, CPU)")
# print(f"  {'Method':12s} | {'Chebyshev (err / ms)':22s} | {'PadeEigen (err / ms)':22s} | {'Taylor (err / ms)':22s}")
# print("  " + "-" * 86)
# for method in ALL_MAGNUS:
#     cols = []
#     for expm_m in ALL_EXPM:
#         psi, ms, st, _ = run_magnus(
#             H_rabi,
#             psi0_2,
#             tlist_rabi,
#             magnus_method=method,
#             expm_method=expm_m,
#             atol=1e-9,
#             rtol=1e-9,
#         )
#         err = (psi - ref_rabi).norm()
#         cols.append(f"{err:8.2e} / {ms:6.2f} ms")
#     print(f"  {method:12s} | {cols[0]:22s} | {cols[1]:22s} | {cols[2]:22s}")


# # ---------------------------------------------------------------------------
# # 3. Узкий гауссов импульс (все 7 методов, проверка адаптивности)
# # ---------------------------------------------------------------------------
# section("3. NARROW GAUSSIAN PULSE (All 7 methods, sigma=0.25, max_step=0.25)")
# H_gauss = qt.QobjEvo([
#     0.5 * qt.sigmaz(),
#     [3.0 * qt.sigmax(), lambda t: np.exp(-((t - 5.0) / 0.25) ** 2) * np.cos(4.0 * t)],
# ])
# tlist_gauss = np.linspace(0.0, 10.0, 21)
# ref_gauss, ref_g_ms = run_ref(H_gauss, psi0_2, tlist_gauss)
# print(f"Reference vern7 (1e-14): {ref_g_ms:.2f} ms")

# for method in ALL_MAGNUS:
#     print(f"\n--- {method} ---")
#     for tol in (1e-6, 1e-9, 1e-11):
#         psi, ms, st, rej = run_magnus(
#             H_gauss,
#             psi0_2,
#             tlist_gauss,
#             magnus_method=method,
#             atol=tol,
#             rtol=tol,
#             max_step=0.25,
#         )
#         err = (psi - ref_gauss).norm()
#         print(f"  tol={tol:7.1e} | ||d_psi||={err:9.3e} | steps={st:6d} (rej={rej:3d}) | {ms:7.2f} ms")


# # ---------------------------------------------------------------------------
# # 4. Все уровни гамильтонианов: Stationary / Multi-channel / Black-box Oracle
# # ---------------------------------------------------------------------------
# section("4. HAMILTONIAN LEVELS (All 7 methods on Level 0, Level 2, Level 3)")

# # Level 0: Stationary (ожидается ровно 50 шагов)
# H_stat = qt.QobjEvo(0.5 * qt.sigmaz() + 0.3 * qt.sigmax())
# tlist_50 = np.linspace(0.0, 10.0, 51)
# ref_stat, _ = run_ref(H_stat, psi0_2, tlist_50)
# print("\n[Level 0: Stationary, tol=1e-12, expected steps=50]")
# for method in ALL_MAGNUS:
#     psi, ms, st, rej = run_magnus(
#         H_stat, psi0_2, tlist_50, magnus_method=method, atol=1e-12, rtol=1e-12
#     )
#     print(f"  {method:12s} | ||d_psi||={(psi - ref_stat).norm():9.3e} | steps={st:5d} (rej={rej:2d}) | {ms:6.2f} ms")

# # Level 2: Multi-channel (2 кубита, N=4)
# sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
# sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
# sx1 = qt.tensor(qt.sigmax(), qt.qeye(2))
# sy2 = qt.tensor(qt.qeye(2), qt.sigmay())
# zz = qt.tensor(qt.sigmaz(), qt.sigmaz())
# H_mc = qt.QobjEvo([
#     0.5 * sz1 + 0.7 * sz2 + 0.25 * zz,
#     [0.3 * sx1, lambda t: np.cos(1.3 * t)],
#     [0.2 * sy2, lambda t: np.sin(2.1 * t)],
# ])
# psi0_4 = qt.tensor(qt.basis(2, 0), qt.basis(2, 1))
# tlist_mc = np.linspace(0.0, 10.0, 51)
# ref_mc, ref_mc_ms = run_ref(H_mc, psi0_4, tlist_mc)
# print(f"\n[Level 2: Multi-channel N=4, tol=1e-10] (ref vern7: {ref_mc_ms:.2f} ms)")
# for method in ALL_MAGNUS:
#     psi, ms, st, rej = run_magnus(
#         H_mc, psi0_4, tlist_mc, magnus_method=method, atol=1e-10, rtol=1e-10
#     )
#     print(f"  {method:12s} | ||d_psi||={(psi - ref_mc).norm():9.3e} | steps={st:5d} (rej={rej:2d}) | {ms:6.2f} ms")

# # Level 3: Black-box oracle QobjEvo(callable)
# H0_q, H1_q, H2_q = 0.5 * qt.sigmaz(), 0.25 * qt.sigmax(), 0.15 * qt.sigmay()
# H_oracle = qt.QobjEvo(lambda t: H0_q + np.cos(1.7 * t) * H1_q + np.sin(0.9 * t) * H2_q)
# ref_or, ref_or_ms = run_ref(H_oracle, psi0_2, tlist_mc)
# print(f"\n[Level 3: Black-box Oracle N=2, tol=1e-10] (ref vern7: {ref_or_ms:.2f} ms)")
# for method in ALL_MAGNUS:
#     psi, ms, st, rej = run_magnus(
#         H_oracle, psi0_2, tlist_mc, magnus_method=method, atol=1e-10, rtol=1e-10
#     )
#     print(f"  {method:12s} | ||d_psi||={(psi - ref_or).norm():9.3e} | steps={st:5d} (rej={rej:2d}) | {ms:6.2f} ms")


# # ---------------------------------------------------------------------------
# # 5. Краевые случаи: коммутирующий H(t), фикс. шаг, предохранитель nsteps
# # ---------------------------------------------------------------------------
# section("5. EDGE CASES & QUTIP CONTRACT (All 7 methods)")

# # 5a. Коммутирующий гамильтониан [H0, H1] = 0
# H_comm = qt.QobjEvo([0.5 * qt.sigmaz(), [1.5 * qt.sigmaz(), lambda t: np.cos(5.0 * t)]])
# psi0_plus = (qt.basis(2, 0) + qt.basis(2, 1)).unit()
# tlist_comm = np.linspace(0.0, 10.0, 21)
# ref_comm, _ = run_ref(H_comm, psi0_plus, tlist_comm)
# print("\n[5a. Commuting H0, H1 (all commutators == 0), tol=1e-10]")
# for method in ALL_MAGNUS:
#     psi, ms, st, rej = run_magnus(
#         H_comm, psi0_plus, tlist_comm, magnus_method=method, atol=1e-10, rtol=1e-10
#     )
#     print(f"  {method:12s} | ||d_psi||={(psi - ref_comm).norm():9.3e} | steps={st:5d} (rej={rej:2d}) | {ms:6.2f} ms")

# # 5b. Обратная совместимость: фиксированный dt=0.05 и adaptive=False (nsteps=4)
# print("\n[5b. Fixed-step compatibility (expected steps=400)]")
# for method in ALL_MAGNUS:
#     psi_dt, _, st_dt, _ = run_magnus(H_rabi, psi0_2, tlist_rabi, magnus_method=method, dt=0.05)
#     psi_fix, _, st_fix, _ = run_magnus(
#         H_rabi, psi0_2, tlist_rabi, magnus_method=method, adaptive=False, nsteps=4
#     )
#     print(
#         f"  {method:12s} | dt=0.05: err={(psi_dt - ref_rabi).norm():8.2e} (steps={st_dt:3d}) | "
#         f"adaptive=False: err={(psi_fix - ref_rabi).norm():8.2e} (steps={st_fix:3d})"
#     )

# # 5c. Предохранитель nsteps (должен бросать IntegratorException для всех 7 методов)
# print("\n[5c. nsteps guard (nsteps=2, tol=1e-13 on [0, 20])]")
# tlist_guard = np.array([0.0, 20.0])
# for method in ALL_MAGNUS:
#     try:
#         run_magnus(H_rabi, psi0_2, tlist_guard, magnus_method=method, atol=1e-13, rtol=1e-13, nsteps=2)
#         print(f"  {method:12s} | FAIL: IntegratorException was NOT raised!")
#     except IntegratorException:
#         print(f"  {method:12s} | PASS: caught IntegratorException")


# # # ---------------------------------------------------------------------------
# # # 6. CUDA Backend: большая матрица N=256 (все 7 методов, CPU vs CUDA)
# # # ---------------------------------------------------------------------------
# # N_GPU = 512
# # section(f"6. CUDA BACKEND (All 7 methods, Single-channel N={N_GPU}, tol=1e-7)")
# # if not has_cuda_backend():
# #     print("  CUDA backend not available — skipped.")
# # else:
# #     rng = np.random.default_rng(42)
# #     M0 = rng.normal(size=(N_GPU, N_GPU)) + 1j * rng.normal(size=(N_GPU, N_GPU))
# #     M1 = rng.normal(size=(N_GPU, N_GPU)) + 1j * rng.normal(size=(N_GPU, N_GPU))
# #     H0_raw = 0.5 * (M0 + M0.conj().T)
# #     H1_raw = 0.5 * (M1 + M1.conj().T)
# #     # Нормируем по 1-норме (||H0||_1 = 1.0, ||H1||_1 = 0.25), чтобы масштаб шага не зависел от sqrt(N)
# #     H0_gpu = qt.Qobj(H0_raw / np.linalg.norm(H0_raw, 1))
# #     H1_gpu = qt.Qobj(0.25 * H1_raw / np.linalg.norm(H1_raw, 1))
# #     H_gpu = qt.QobjEvo([H0_gpu, [H1_gpu, lambda t: np.cos(1.5 * t)]])
# #     psi0_n = qt.basis(N_GPU, 0)
# #     tlist_gpu = np.linspace(0.0, 5.0, 6)  # 5 крупных интервалов

# #     ref_gpu, ref_gpu_ms = run_ref(H_gpu, psi0_n, tlist_gpu)
# #     print(f"Reference vern7 (N={N_GPU}, 1e-14): {ref_gpu_ms:.2f} ms")

# #     # Прогрев CUDA-контекста перед замерами времени
# #     _ = run_magnus(
# #         H_gpu, psi0_n, np.array([0.0, 0.1]),
# #         magnus_method="Formula314_6", backend="Cuda", adaptive=False, nsteps=1,
# #     )

# #     print(
# #         f"  {'Method':12s} | {'CPU err (ms)':21s} | {'CUDA err (ms)':21s} | "
# #         f"{'||CPU-CUDA||':12s} | {'steps':5s} | {'CPU/CUDA':8s}"
# #     )
# #     print("  " + "-" * 90)
# #     for method in ALL_MAGNUS:
# #         psi_c, ms_c, st_c, _ = run_magnus(
# #             H_gpu, psi0_n, tlist_gpu, magnus_method=method, backend="Cpu", atol=1e-7, rtol=1e-7
# #         )
# #         psi_g, ms_g, st_g, _ = run_magnus(
# #             H_gpu, psi0_n, tlist_gpu, magnus_method=method, backend="Cuda", atol=1e-7, rtol=1e-7
# #         )
# #         err_c = (psi_c - ref_gpu).norm()
# #         err_g = (psi_g - ref_gpu).norm()
# #         diff_cg = (psi_c - psi_g).norm()
# #         speedup = ms_c / ms_g if ms_g > 0 else float("nan")
# #         print(
# #             f"  {method:12s} | {err_c:8.2e} ({ms_c:7.1f} ms) | {err_g:8.2e} ({ms_g:7.1f} ms) | "
# #             f"{diff_cg:12.2e} | {st_g:5d} | {speedup:7.2f}x"
# #         )

"""
Комплексный стенд экспериментов на матрицах 100x100 (эрмитовы, строгое диагональное преобладание):
  Часть 1: Гармонический косинус — все 7 методов Магнуса + все 3 expm.
  Часть 2: Все остальные способы задания f(t) и A(t) (Level 0, 1B callable, 1B array, Level 2, Level 3, CUDA).
  Часть 3: Трассировка фактической маршрутизации — какой именно C++/CUDA-метод вызвался в каждом случае.
"""
from __future__ import annotations

import math
import time
from collections import Counter
from contextlib import contextmanager

import numpy as np
import qutip as qt
from scipy.interpolate import CubicSpline

import pymagnus_hpc.qutip_plugin  # регистрирует "magnus_chebyshev" в QuTiP
from pymagnus_hpc import _magnus_py
from pymagnus_hpc.solver import has_cuda_backend, solve_magnus

ALL_MAGNUS = (
    "Classic2",
    "Classic4",
    "Acc4",
    "Formula314_4",
    "Acc6",
    "Formula314_6",
    "Recursive",
)
ALL_EXPM = ("Chebyshev", "PadeEigen", "Taylor")


# =============================================================================
# 0. Генерация эрмитовых матриц 100x100 со строгим диагональным преобладанием
# =============================================================================
def make_diagonally_dominant(
    rng: np.random.Generator, N: int, margin: float, scale: float
) -> np.ndarray:
    raw = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H = 0.5 * (raw + raw.conj().T)
    H_off = H.copy()
    np.fill_diagonal(H_off, 0.0)
    off_row_sums = np.sum(np.abs(H_off), axis=1)
    diag_vals = np.maximum(np.abs(np.diag(H)), off_row_sums + margin)
    H_dom = H_off + np.diag(diag_vals)
    mat = np.ascontiguousarray(scale * H_dom, dtype=np.complex128)

    assert np.allclose(mat, mat.conj().T, atol=1e-14), "Матрица не эрмитова"
    off_check = np.sum(np.abs(mat - np.diag(np.diag(mat))), axis=1)
    assert np.all(np.abs(np.diag(mat)) > off_check), "Нарушено строгое диагональное преобладание"
    return mat


def make_test_system(N: int = 100, seed: int = 42):
    rng = np.random.default_rng(seed)
    H0 = make_diagonally_dominant(rng, N, margin=10.0, scale=0.05)
    H1 = make_diagonally_dominant(rng, N, margin=5.0, scale=0.02)
    H2 = make_diagonally_dominant(rng, N, margin=5.0, scale=0.015)

    # Плотный нормированный начальный вектор (задействует все столбцы пропагатора U)
    v = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    v /= np.linalg.norm(v)
    psi0_vec = np.ascontiguousarray(v, dtype=np.complex128)
    psi0_q = qt.Qobj(psi0_vec)
    return H0, H1, H2, psi0_vec, psi0_q


def run_vern7_ref(H_evo: qt.QobjEvo, psi0: qt.Qobj, tlist: np.ndarray, tol: float = 1e-13):
    t0 = time.perf_counter()
    res = qt.sesolve(
        H_evo,
        psi0,
        tlist,
        options={
            "method": "vern7",
            "atol": tol,
            "rtol": tol,
            "nsteps": 200000,
            "store_final_state": True,
        },
    )
    return res.final_state, (time.perf_counter() - t0) * 1e3


def run_qutip_magnus(H_evo: qt.QobjEvo, psi0: qt.Qobj, tlist: np.ndarray, **opts):
    base = {
        "method": "magnus_chebyshev",
        "nsteps": 200000,
        "store_final_state": True,
    }
    base.update(opts)
    solver = qt.SESolver(H_evo, options=base)
    t0 = time.perf_counter()
    res = solver.run(psi0, tlist)
    dt_ms = (time.perf_counter() - t0) * 1e3
    integ = solver._integrator
    return res.final_state, dt_ms, integ


def section(title: str) -> None:
    print(f"\n{'=' * 102}\n{title}\n{'=' * 102}")


# =============================================================================
# Трассировщик фактических вызовов C++/CUDA-ядер (_magnus_py)
# =============================================================================
def _enum_name(val) -> str:
    return getattr(val, "name", str(val).split(".")[-1])


def _is_uniform_grid(time_nodes: np.ndarray) -> bool:
    n_nodes = int(time_nodes.size)
    if n_nodes <= 2:
        return True
    diffs = np.diff(time_nodes)
    h0 = float(diffs[0])
    return bool(np.all(np.abs(diffs - h0) <= 1e-12 * max(1.0, abs(h0))))


def _resolve_cpu_zero_copy_target(
    N: int,
    time_nodes: np.ndarray,
    mod_grid: np.ndarray,
    is_harmonic: bool,
    omega: float,
    opts,
) -> str:
    """Точно воспроизводит ветвление module.cpp + magnus_bridge.cpp для отчёта."""
    method = _enum_name(opts.magnus_method)
    expm = _enum_name(opts.expm_method)
    backend = _enum_name(opts.backend)
    steps = int(time_nodes.size - 1)
    is_uniform = _is_uniform_grid(time_nodes)

    use_cuda_harmonic = (
        is_harmonic
        and is_uniform
        and (
            backend == "Cuda"
            or (
                backend == "Auto"
                and N >= 128
                and has_cuda_backend()
                and method == "Formula314_6"
                and expm == "Chebyshev"
            )
        )
    )
    if use_cuda_harmonic:
        return f"CUDA propagate_cuda_raw -> MagnusChebyshevGPU::solve (order 6, {expm})"

    if backend == "MpiCpu" and is_harmonic:
        return "MPI -> mpi_magnus_chebyshev_solver"

    if is_harmonic:
        if method == "Formula314_6" and expm == "Chebyshev" and is_uniform and omega > 0.0:
            return "CORE single_cpu::solve_formula314_chebyshev (uniform, Chebyshev)"
        if method == "Classic2":
            return f"CORE ClassicMagnus.cpp :: magnus_classic(order=1) + {expm}"
        if method == "Classic4":
            return f"CORE ClassicMagnus.cpp :: magnus_classic(order=2) + {expm}"
        if method == "Acc4":
            return f"CORE AccMagnus.cpp :: generate_samples + magnus_ACC(order=4) + {expm}"
        if method == "Acc6":
            return f"CORE AccMagnus.cpp :: generate_samples + magnus_ACC(order=6) + {expm}"
        if method == "Recursive":
            return f"CORE RecursiveMagnus.cpp :: magnus_expansion(max_n={opts.recursive_order}) + {expm}"
        if method == "Formula314_4":
            return f"BRIDGE magnus_bridge.cpp :: analytical Formula 3.14 (order 4) + {expm}"
        return f"CORE Formula314Magnus.cpp :: magnus_3_14 (non-uniform/custom expm) + {expm}"

    # Негармонический путь -> propagate_arbitrary_modulation_mkl
    if mod_grid.size == 3 * steps:
        if method == "Classic2":
            ord_tag = "GL3 order 2 (midpoint)"
        elif method in ("Classic4", "Acc4", "Formula314_4"):
            ord_tag = "GL3 Formula 3.14 (order 4)"
        else:
            ord_tag = "GL3 Formula 3.14 (order 6)"
        return f"BRIDGE propagate_arbitrary_modulation_mkl [{ord_tag}] + {expm}"

    if method in ("Acc4", "Acc6"):
        return f"CORE AccMagnus.cpp :: magnus_ACC(order={6 if method == 'Acc6' else 4}) + {expm}"
    return f"BRIDGE propagate_arbitrary_modulation_mkl [finite-diff 3.14] + {expm}"


def _resolve_general_gl_target(opts) -> str:
    method = _enum_name(opts.magnus_method)
    expm = _enum_name(opts.expm_method)
    if method == "Classic2":
        tag = "GL2 (midpoint h*A2)"
    elif method in ("Classic4", "Acc4", "Formula314_4"):
        tag = "GL4 (Blanes-Casas-Ros, coeff sqrt(15)/36)"
    else:
        tag = "GL6 (Blanes-Casas-Ros 6th order)"
    return f"BRIDGE propagate_general_gl_mkl [{tag}] + {expm}"


def _resolve_cuda_gl3_target(opts) -> str:
    method = _enum_name(opts.magnus_method)
    if method == "Classic2":
        tag = "GL3 order 2"
    elif method in ("Classic4", "Acc4", "Formula314_4"):
        tag = "GL3 order 4"
    else:
        tag = "GL3 order 6"
    return f"CUDA propagate_cuda_gl3_raw -> MagnusChebyshevGPU::solve_coeffs [{tag}]"


@contextmanager
def trace_magnus_dispatch():
    """Перехватывает все вызовы _magnus_py и собирает лог фактической диспетчеризации."""
    records: list[dict] = []
    originals = {
        "expm_cpu": _magnus_py.expm_cpu,
        "propagate_cpu_zero_copy": _magnus_py.propagate_cpu_zero_copy,
        "propagate_cuda_gl3": _magnus_py.propagate_cuda_gl3,
        "propagate_general_gl_cpu": _magnus_py.propagate_general_gl_cpu,
        "propagate_cuda_dlpack": _magnus_py.propagate_cuda_dlpack,
    }

    def wrap_expm_cpu(A, out, method=_magnus_py.ExpmMethod.Chebyshev, degree_or_terms=16):
        m_name = _enum_name(method)
        records.append({
            "binding": "expm_cpu",
            "target": f"CORE compute_expm_cpu -> matrix_ops::expm_{m_name.lower()}",
            "steps": 1,
            "is_harmonic": None,
            "uniform": True,
        })
        return originals["expm_cpu"](A, out, method, degree_or_terms)

    def wrap_cpu_zero_copy(H0, H1, time_nodes, mod_grid, is_harmonic, epsilon, omega, opts, U_out):
        t_arr = np.asarray(time_nodes, dtype=np.float64)
        m_arr = np.asarray(mod_grid, dtype=np.float64)
        records.append({
            "binding": "propagate_cpu_zero_copy",
            "target": _resolve_cpu_zero_copy_target(
                H0.shape[0], t_arr, m_arr, bool(is_harmonic), float(omega), opts
            ),
            "steps": int(t_arr.size - 1),
            "is_harmonic": bool(is_harmonic),
            "uniform": _is_uniform_grid(t_arr),
        })
        return originals["propagate_cpu_zero_copy"](
            H0, H1, time_nodes, mod_grid, is_harmonic, epsilon, omega, opts, U_out
        )

    def wrap_cuda_gl3(H0, H1, time_nodes, gl_nodes, opts, U_out):
        t_arr = np.asarray(time_nodes, dtype=np.float64)
        records.append({
            "binding": "propagate_cuda_gl3",
            "target": _resolve_cuda_gl3_target(opts),
            "steps": int(t_arr.size - 1),
            "is_harmonic": False,
            "uniform": _is_uniform_grid(t_arr),
        })
        return originals["propagate_cuda_gl3"](H0, H1, time_nodes, gl_nodes, opts, U_out)

    def wrap_general_gl_cpu(time_nodes, A_gl_nodes, opts, U_out):
        t_arr = np.asarray(time_nodes, dtype=np.float64)
        records.append({
            "binding": "propagate_general_gl_cpu",
            "target": _resolve_general_gl_target(opts),
            "steps": int(t_arr.size - 1),
            "is_harmonic": False,
            "uniform": _is_uniform_grid(t_arr),
        })
        return originals["propagate_general_gl_cpu"](time_nodes, A_gl_nodes, opts, U_out)

    def wrap_cuda_dlpack(d_H0, d_H1, t_start, t_final, steps, epsilon, omega, opts, d_U_out):
        records.append({
            "binding": "propagate_cuda_dlpack",
            "target": "CUDA propagate_cuda_raw -> MagnusChebyshevGPU::solve_to_device (DLPack)",
            "steps": int(steps),
            "is_harmonic": True,
            "uniform": True,
        })
        return originals["propagate_cuda_dlpack"](
            d_H0, d_H1, t_start, t_final, steps, epsilon, omega, opts, d_U_out
        )

    _magnus_py.expm_cpu = wrap_expm_cpu
    _magnus_py.propagate_cpu_zero_copy = wrap_cpu_zero_copy
    _magnus_py.propagate_cuda_gl3 = wrap_cuda_gl3
    _magnus_py.propagate_general_gl_cpu = wrap_general_gl_cpu
    _magnus_py.propagate_cuda_dlpack = wrap_cuda_dlpack
    try:
        yield records
    finally:
        for name, orig in originals.items():
            setattr(_magnus_py, name, orig)


# =============================================================================
# ГЛАВНЫЙ ПРОГОН
# =============================================================================
def main():
    N = 100
    T = 1.0
    eps = 0.5
    omega = 2.0 * np.pi
    tlist = np.linspace(0.0, T, 11)  # 10 интервалов вывода по 0.1

    H0, H1, H2, psi0_vec, psi0_q = make_test_system(N=N, seed=42)
    H0_q, H1_q, H2_q = qt.Qobj(H0), qt.Qobj(H1), qt.Qobj(H2)

    print(f"Система: N = {N}x{N}, эрмитовы матрицы со строгим диагональным преобладанием")
    print(f"  ||H0||_1 = {np.linalg.norm(H0, 1):.3f}, ||H1||_1 = {np.linalg.norm(H1, 1):.3f}, ||H2||_1 = {np.linalg.norm(H2, 1):.3f}")
    print(f"  T = {T}, eps = {eps}, omega = {omega:.4f}, CUDA available = {has_cuda_backend()}")

    # =========================================================================
    # ЧАСТЬ 1: КОСИНУС НА 100x100 (ВСЕ 7 МЕТОДОВ + ВСЕ 3 EXPM)
    # =========================================================================
    section("ЧАСТЬ 1: ГАРМОНИЧЕСКИЙ КОСИНУС H0 + eps*cos(omega*t)*H1 (N=100, QuTiP atol/rtol)")
    H_cos = qt.QobjEvo([H0_q, [H1_q, lambda t: eps * np.cos(omega * t)]])
    ref_cos, ref_cos_ms = run_vern7_ref(H_cos, psi0_q, tlist)
    print(f"Эталон QuTiP vern7 (atol=rtol=1e-13): {ref_cos_ms:.2f} ms")

    for use_core in (True, False):
        mode_label = (
            "use_harmonic_core=True  (родные C++ ядра из src/core/magnus/)"
            if use_core
            else "use_harmonic_core=False (универсальный GL3-мост)"
        )
        print(f"\n>>> Режим: {mode_label}")
        print(f"  {'Метод':13s} | {'tol':7s} | {'||d_psi||':10s} | {'| ||psi||-1 |':13s} | {'steps (rej)':13s} | {'Время':9s}")
        print("  " + "-" * 80)
        for method in ALL_MAGNUS:
            for tol in (1e-6, 1e-8):
                psi, ms, integ = run_qutip_magnus(
                    H_cos,
                    psi0_q,
                    tlist,
                    magnus_method=method,
                    atol=tol,
                    rtol=tol,
                    use_harmonic_core=use_core,
                )
                err = (psi - ref_cos).norm()
                norm_def = abs(psi.norm() - 1.0)
                print(
                    f"  {method:13s} | {tol:7.1e} | {err:10.3e} | {norm_def:13.3e} | "
                    f"{integ.total_steps:6d} ({integ.rejected_steps:3d})  | {ms:7.1f} ms"
                )

    print("\n>>> Все 3 метода матричной экспоненты на N=100 (Formula314_6, tol=1e-8, harmonic core)")
    for expm_m in ALL_EXPM:
        psi, ms, integ = run_qutip_magnus(
            H_cos,
            psi0_q,
            tlist,
            magnus_method="Formula314_6",
            expm_method=expm_m,
            atol=1e-8,
            rtol=1e-8,
        )
        err = (psi - ref_cos).norm()
        print(f"  {expm_m:10s} | ||d_psi||={err:9.3e} | | ||psi||-1 |={abs(psi.norm()-1.0):9.3e} | steps={integ.total_steps:4d} | {ms:7.1f} ms")

    # =========================================================================
    # ЧАСТЬ 2: ВСЕ ОСТАЛЬНЫЕ СПОСОБЫ ЗАДАНИЯ f(t) И МАТРИЦЫ A(t) НА 100x100
    # =========================================================================
    section("ЧАСТЬ 2: ВСЕ ОСТАЛЬНЫЕ СПОСОБЫ ЗАДАНИЯ f(t) И A(t) (N=100, tol=1e-6)")

    # 2.1 Level 0: Стационарный гамильтониан H0
    H_stat = qt.QobjEvo(H0_q)
    ref_stat, ref_stat_ms = run_vern7_ref(H_stat, psi0_q, tlist)
    print(f"\n[2.1 Level 0: Стационарный H0] (эталон vern7: {ref_stat_ms:.2f} ms)")
    for method in ALL_MAGNUS:
        psi, ms, integ = run_qutip_magnus(H_stat, psi0_q, tlist, magnus_method=method, atol=1e-8, rtol=1e-8)
        print(f"  {method:13s} | ||d_psi||={(psi - ref_stat).norm():9.3e} | steps={integ.total_steps:4d} (rej={integ.rejected_steps:2d}) | {ms:7.1f} ms")

    # 2.2 Level 1B (callable): Негармоническая функция f(t) — гауссов импульс с несущей
    f_gauss = lambda t: 0.8 * np.exp(-((t - 0.5) / 0.15) ** 2) * np.cos(8.0 * t)
    H_gauss = qt.QobjEvo([H0_q, [H1_q, f_gauss]])
    ref_gauss, ref_gauss_ms = run_vern7_ref(H_gauss, psi0_q, tlist)
    print(f"\n[2.2 Level 1B (callable): Гауссов импульс H0 + f_gauss(t)*H1] (эталон vern7: {ref_gauss_ms:.2f} ms)")
    for method in ALL_MAGNUS:
        psi, ms, integ = run_qutip_magnus(
            H_gauss, psi0_q, tlist, magnus_method=method, atol=1e-6, rtol=1e-6, max_step=0.1
        )
        print(f"  {method:13s} | ||d_psi||={(psi - ref_gauss).norm():9.3e} | steps={integ.total_steps:4d} (rej={integ.rejected_steps:2d}) | {ms:7.1f} ms")

    # 2.3 Level 1B (табулированный массив): CubicSpline в solve_magnus и в QuTiP
    t_tab = np.linspace(0.0, T, 101)
    f_tab = np.asarray([f_gauss(tv) for tv in t_tab], dtype=np.float64)
    cs_tab = CubicSpline(t_tab, f_tab, bc_type="natural")
    H_tab_qt = qt.QobjEvo([H0_q, [H1_q, lambda t: float(cs_tab(t))]])
    ref_tab, ref_tab_ms = run_vern7_ref(H_tab_qt, psi0_q, tlist)
    print(f"\n[2.3 Level 1B (табулированный массив f_tab[101]): QuTiP spline + прямой solve_magnus(f_mod=array)]")
    for method in ALL_MAGNUS:
        # 1) Через QuTiP (адаптивный шаг)
        psi_qt, ms_qt, integ = run_qutip_magnus(
            H_tab_qt, psi0_q, tlist, magnus_method=method, atol=1e-6, rtol=1e-6, max_step=0.1
        )
        # 2) Прямой вызов solve_magnus с numpy-массивом f_mod=f_tab (20 шагов)
        t0 = time.perf_counter()
        U_tab = solve_magnus(
            H0=H0, H1=H1, t_start=0.0, t_final=T, steps=20, f_mod=f_tab, magnus_method=method
        )
        ms_dir = (time.perf_counter() - t0) * 1e3
        err_dir = float(np.linalg.norm(U_tab @ psi0_vec - ref_tab.full().ravel()))
        unit_def = float(np.linalg.norm(U_tab.conj().T @ U_tab - np.eye(N)))
        print(
            f"  {method:13s} | QuTiP: err={(psi_qt - ref_tab).norm():8.2e} (st={integ.total_steps:3d}, {ms_qt:6.1f} ms) | "
            f"Direct array(20): err={err_dir:8.2e}, ||U*U-I||={unit_def:7.1e} ({ms_dir:6.1f} ms)"
        )

    # 2.4 Level 2: Многоканальный гамильтониан H0 + f1(t)*H1 + f2(t)*H2
    f1_mc = lambda t: 0.5 * np.cos(omega * t)
    f2_mc = lambda t: 0.4 * np.sin(3.5 * t)
    H_mc = qt.QobjEvo([H0_q, [H1_q, f1_mc], [H2_q, f2_mc]])
    ref_mc, ref_mc_ms = run_vern7_ref(H_mc, psi0_q, tlist)
    print(f"\n[2.4 Level 2: Многоканальный H0 + f1(t)*H1 + f2(t)*H2] (эталон vern7: {ref_mc_ms:.2f} ms)")
    for method in ALL_MAGNUS:
        psi, ms, integ = run_qutip_magnus(H_mc, psi0_q, tlist, magnus_method=method, atol=1e-6, rtol=1e-6)
        print(f"  {method:13s} | ||d_psi||={(psi - ref_mc).norm():9.3e} | steps={integ.total_steps:4d} (rej={integ.rejected_steps:2d}) | {ms:7.1f} ms")

    # 2.5 Level 3: Произвольный матричный оракул H(t) = H_func(t)
    H_oracle = qt.QobjEvo(lambda t: H0_q + (0.5 * np.cos(omega * t)) * H1_q + (0.4 * np.sin(3.5 * t)) * H2_q)
    ref_or, ref_or_ms = run_vern7_ref(H_oracle, psi0_q, tlist)
    print(f"\n[2.5 Level 3: Матричный оракул QobjEvo(lambda t: H(t))] (эталон vern7: {ref_or_ms:.2f} ms)")
    for method in ALL_MAGNUS:
        psi, ms, integ = run_qutip_magnus(H_oracle, psi0_q, tlist, magnus_method=method, atol=1e-6, rtol=1e-6)
        print(f"  {method:13s} | ||d_psi||={(psi - ref_or).norm():9.3e} | steps={integ.total_steps:4d} (rej={integ.rejected_steps:2d}) | {ms:7.1f} ms")

    # 2.6 CUDA на N=100 (если доступен)
    if has_cuda_backend():
        print("\n[2.6 CUDA Backend (N=100, tol=1e-6): Косинус и Гауссов импульс vs CPU GL3]")
        # Прогрев GPU
        _ = run_qutip_magnus(H_cos, psi0_q, np.array([0.0, 0.1]), backend="Cuda", adaptive=False, nsteps=1)
        for label, H_case, ref_case in (("Cosine", H_cos, ref_cos), ("Gauss ", H_gauss, ref_gauss)):
            for method in ("Classic2", "Classic4", "Formula314_6"):
                psi_c, ms_c, _ = run_qutip_magnus(
                    H_case, psi0_q, tlist, magnus_method=method, backend="Cpu",
                    atol=1e-6, rtol=1e-6, max_step=0.1, use_harmonic_core=False,
                )
                psi_g, ms_g, integ_g = run_qutip_magnus(
                    H_case, psi0_q, tlist, magnus_method=method, backend="Cuda",
                    atol=1e-6, rtol=1e-6, max_step=0.1,
                )
                print(
                    f"  {label} | {method:12s} | CPU={((psi_c - ref_case).norm()):8.2e} ({ms_c:6.1f} ms) | "
                    f"CUDA={((psi_g - ref_case).norm()):8.2e} ({ms_g:6.1f} ms) | "
                    f"||CPU-CUDA||={(psi_c - psi_g).norm():8.2e} | st={integ_g.total_steps}"
                )

    # =========================================================================
    # ЧАСТЬ 3: ЭКСПЕРИМЕНТ ПО ТРАССИРОВКЕ — КАКОЙ ИМЕННО МЕТОД ВЫЗВАЛСЯ
    # =========================================================================
    section("ЧАСТЬ 3: ТРАССИРОВКА ДИСПЕТЧЕРИЗАЦИИ — КАКОЙ C++/CUDA МЕТОД РЕАЛЬНО ВЫЗЫВАЕТСЯ")
    t_probe = np.array([0.0, 0.25])  # короткий интервал для быстрой трассировки

    def run_and_print_trace(case_name: str, fn):
        with trace_magnus_dispatch() as log:
            extra_info = fn()
        summary = Counter((r["binding"], r["target"]) for r in log)
        print(f"\n• {case_name}")
        if extra_info:
            print(f"    QuTiP state : {extra_info}")
        for (binding, target), cnt in summary.items():
            flags = [r for r in log if r["binding"] == binding and r["target"] == target][0]
            print(
                f"    Binding     : _magnus_py.{binding} (calls={cnt}, is_harmonic={flags['is_harmonic']}, "
                f"uniform={flags['uniform']}, steps={flags['steps']})"
            )
            print(f"    Actual C++  : {target}")

    def qt_case(H_evo, **opts):
        def _run():
            _, _, integ = run_qutip_magnus(H_evo, psi0_q, t_probe, **opts)
            return (
                f"channels={len(integ._channels)}, use_oracle={integ._use_oracle}, "
                f"harmonic_params={integ._harmonic_params}"
            )
        return _run

    # 3A. Все 7 методов в QuTiP на косинусе (адаптивный шаг + фиксированный шаг)
    for m in ALL_MAGNUS:
        run_and_print_trace(
            f"QuTiP | Косинус | adaptive (tol=1e-7) | {m}",
            qt_case(H_cos, magnus_method=m, atol=1e-7, rtol=1e-7),
        )
    run_and_print_trace(
        "QuTiP | Косинус | fixed dt=0.05 (uniform) | Formula314_6",
        qt_case(H_cos, magnus_method="Formula314_6", dt=0.05),
    )
    run_and_print_trace(
        "QuTiP | Косинус | use_harmonic_core=False | Formula314_6",
        qt_case(H_cos, magnus_method="Formula314_6", atol=1e-7, rtol=1e-7, use_harmonic_core=False),
    )
    run_and_print_trace(
        "QuTiP | Косинус | PadeEigen | Formula314_6",
        qt_case(H_cos, magnus_method="Formula314_6", expm_method="PadeEigen", dt=0.05),
    )

    # 3B. Все остальные способы задания H(t) через QuTiP
    run_and_print_trace(
        "QuTiP | Level 0: Стационарный H0 | Formula314_6",
        qt_case(H_stat, magnus_method="Formula314_6", atol=1e-7, rtol=1e-7),
    )
    H_sin = qt.QobjEvo([H0_q, [H1_q, lambda t: 0.5 * np.sin(omega * t)]])
    run_and_print_trace(
        "QuTiP | Level 1B: sin(omega*t) (не чистый cos) | Acc6",
        qt_case(H_sin, magnus_method="Acc6", atol=1e-6, rtol=1e-6),
    )
    for m in ("Classic2", "Acc4", "Formula314_6", "Recursive"):
        run_and_print_trace(
            f"QuTiP | Level 1B: Гауссов импульс | {m}",
            qt_case(H_gauss, magnus_method=m, atol=1e-6, rtol=1e-6, max_step=0.1),
        )
    for m in ("Classic2", "Acc4", "Formula314_6"):
        run_and_print_trace(
            f"QuTiP | Level 2: Многоканальный (2 канала) | {m}",
            qt_case(H_mc, magnus_method=m, atol=1e-6, rtol=1e-6),
        )
    for m in ("Classic2", "Classic4", "Formula314_6"):
        run_and_print_trace(
            f"QuTiP | Level 3: Матричный оракул QobjEvo(callable) | {m}",
            qt_case(H_oracle, magnus_method=m, atol=1e-6, rtol=1e-6),
        )

    # 3C. Прямые вызовы solve_magnus(...)
    run_and_print_trace(
        "Direct solve_magnus | Level 1A: Гармонический (uniform) | Formula314_6",
        lambda: (solve_magnus(H0, H1, 0.0, 0.25, steps=5, epsilon=eps, omega=omega, magnus_method="Formula314_6"), None)[1],
    )
    run_and_print_trace(
        "Direct solve_magnus | Level 1A: Гармонический (uniform) | Acc6",
        lambda: (solve_magnus(H0, H1, 0.0, 0.25, steps=5, epsilon=eps, omega=omega, magnus_method="Acc6"), None)[1],
    )
    run_and_print_trace(
        "Direct solve_magnus | Level 1A: Гармонический (non-uniform time_nodes) | Formula314_6",
        lambda: (
            solve_magnus(
                H0, H1, time_nodes=np.array([0.0, 0.05, 0.12, 0.25]),
                epsilon=eps, omega=omega, magnus_method="Formula314_6",
            ),
            None,
        )[1],
    )
    run_and_print_trace(
        "Direct solve_magnus | Level 1B: Табулированный массив f_mod=f_tab | Acc4",
        lambda: (solve_magnus(H0, H1, 0.0, 0.25, steps=5, f_mod=f_tab, magnus_method="Acc4"), None)[1],
    )
    run_and_print_trace(
        "Direct solve_magnus | Level 2: channels=[(H1, f1), (H2, f2)] | Formula314_6",
        lambda: (solve_magnus(H0=H0, t_start=0.0, t_final=0.25, steps=5, channels=[(H1, f1_mc), (H2, f2_mc)]), None)[1],
    )
    run_and_print_trace(
        "Direct solve_magnus | Level 3: H_func(t) | Classic4",
        lambda: (
            solve_magnus(
                t_start=0.0, t_final=0.25, steps=5,
                H_func=lambda t: H0 + 0.5 * np.cos(omega * t) * H1,
                magnus_method="Classic4",
            ),
            None,
        )[1],
    )

    # 3D. CUDA и BackendType.Auto (порог N >= 128 в module.cpp)
    if has_cuda_backend():
        run_and_print_trace(
            "QuTiP | CUDA | Косинус | Formula314_6",
            qt_case(H_cos, magnus_method="Formula314_6", backend="Cuda", atol=1e-6, rtol=1e-6),
        )
        run_and_print_trace(
            "QuTiP | CUDA | Гауссов импульс | Classic4",
            qt_case(H_gauss, magnus_method="Classic4", backend="Cuda", atol=1e-6, rtol=1e-6, max_step=0.1),
        )
        run_and_print_trace(
            "Direct solve_magnus | backend='Auto', N=100 (<128) | Formula314_6",
            lambda: (
                solve_magnus(H0, H1, 0.0, 0.25, steps=5, epsilon=eps, omega=omega, backend="Auto", magnus_method="Formula314_6"),
                None,
            )[1],
        )
        rng128 = np.random.default_rng(7)
        H0_128 = make_diagonally_dominant(rng128, 128, margin=10.0, scale=0.05)
        H1_128 = make_diagonally_dominant(rng128, 128, margin=5.0, scale=0.02)
        run_and_print_trace(
            "Direct solve_magnus | backend='Auto', N=128 (>=128) | Formula314_6",
            lambda: (
                solve_magnus(H0_128, H1_128, 0.0, 0.25, steps=5, epsilon=eps, omega=omega, backend="Auto", magnus_method="Formula314_6"),
                None,
            )[1],
        )
        run_and_print_trace(
            "Direct solve_magnus | backend='Auto', N=128 (>=128) | Acc6 (не должен улетать на CUDA)",
            lambda: (
                solve_magnus(H0_128, H1_128, 0.0, 0.25, steps=5, epsilon=eps, omega=omega, backend="Auto", magnus_method="Acc6"),
                None,
            )[1],
        )


if __name__ == "__main__":
    main()