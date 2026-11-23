# # import numpy as np
# # import qutip as qt
# # import pymagnus_hpc.qutip_plugin  # Регистрирует "magnus_chebyshev" в qt.SESolver

# # # 1. Задаём гамильтониан системы
# # omega_0 = 1.0   # Собственная частота кубита
# # eps = 0.2       # Амплитуда внешнего поля
# # omega = 1.0     # Частота внешнего поля (резонанс)

# # H0 = 0.5 * omega_0 * qt.sigmaz()   # Статичная часть
# # H1 = eps * qt.sigmax()             # Оператор взаимодействия с полем

# # def drive_coeff(t, args=None):
# #     return np.cos(omega * t)

# # # Стандартный формат нестационарного гамильтониана в QuTiP: H0 + f(t) * H1
# # H = [H0, [H1, drive_coeff]]

# # # 2. Задаём начальное состояние и сетку времени для записи наблюдаемых
# # psi0 = qt.basis(2, 0)                # Начальное состояние |0>
# # tlist = np.linspace(0.0, 20.0, 101)  # 100 интервалов вывода результатов

# # # 3. Запускаем эволюцию через стандартный qt.sesolve
# # result = qt.sesolve(
# #     H,
# #     psi0,
# #     tlist,
# #     e_ops=[qt.sigmaz(), qt.sigmax()],   # Что измеряем в каждой точке tlist
# #     options={
# #         "method": "magnus_chebyshev",    # Обязательно: включаем наш интегратор
# #         "nsteps": 5,                     # Сколько шагов Магнуса делать между точками tlist
# #         "magnus_method": "Formula314_6", # Схема Магнуса 6-го порядка
# #         "expm_method": "Chebyshev",      # Матричная экспонента через полиномы Чебышёва
# #         "chebyshev_degree": 16,          # Степень полинома Чебышёва
# #         "backend": "Cpu",                # Где считать: "Cpu", "Cuda" или "Auto"
# #     },
# # )

# # print("Населённость <sigma_z> в конце:", result.expect[0][-1])


# import numpy as np
# import qutip as qt
# import pymagnus_hpc.qutip_plugin  # Регистрирует "magnus_chebyshev" в qt.SESolver[cite: 10]

# # 1. Задаём гамильтониан системы (осцилляции Раби)
# omega_0 = 1.0   # Собственная частота кубита
# eps = 0.2       # Амплитуда внешнего поля
# omega = 1.0     # Частота внешнего поля

# H0 = 0.5 * omega_0 * qt.sigmaz()
# H1 = eps * qt.sigmax()

# def drive_coeff(t, args=None):
#     return np.cos(omega * t)

# H = [H0, [H1, drive_coeff]]

# # 2. Начальное состояние и сетка времени
# psi0 = qt.basis(2, 0)
# tlist = np.linspace(0.0, 20.0, 101)

# # 3. Расчёт через наш солвер Магнуса–Чебышёва[cite: 10]
# res_magnus = qt.sesolve(
#     H,
#     psi0,
#     tlist,
#     e_ops=[qt.sigmaz(), qt.sigmax()],
#     options={
#         "method": "magnus_chebyshev",
#         "atol": 1e-13,
#         "rtol": 1e-13,
#         "nsteps": 100000,
#         "magnus_method": "Formula314_6",  # Можно проверить также "Classic4", "Acc4", "Classic2"
#         "expm_method": "Chebyshev",
#         "chebyshev_degree": 16,
#         "backend": "Cpu",
#         "store_final_state": True,
#     },
# )

# # 4. Эталонный расчёт встроенным решателем QuTiP (метод Вернера 7-го порядка)
# res_qutip = qt.sesolve(
#     H,
#     psi0,
#     tlist,
#     e_ops=[qt.sigmaz(), qt.sigmax()],
#     options={
#         "method": "vern7",
#         "atol": 1e-13,
#         "rtol": 1e-13,
#         "nsteps": 100000,
#         "store_final_state": True,
#     },
# )

# # 5. Сравнение результатов
# sz_magnus = np.asarray(res_magnus.expect[0])
# sz_qutip = np.asarray(res_qutip.expect[0])

# abs_err_final = abs(sz_magnus[-1] - sz_qutip[-1])
# max_err_traj = np.max(np.abs(sz_magnus - sz_qutip))
# state_diff = (res_magnus.final_state - res_qutip.final_state).norm()

# print(f"Magnus-Chebyshev <sigma_z>(T): {sz_magnus[-1]:.15f}")
# print(f"QuTiP (vern7)    <sigma_z>(T): {sz_qutip[-1]:.15f}")
# print("-" * 55)
# print(f"Ошибка <sigma_z> в конечной точке : {abs_err_final:.3e}")
# print(f"Макс. ошибка <sigma_z> по всей сетке: {max_err_traj:.3e}")
# print(f"Норма разности векторов ||d_psi||  : {state_diff:.3e}")

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


# # ---------------------------------------------------------------------------
# # 6. CUDA Backend: большая матрица N=256 (все 7 методов, CPU vs CUDA)
# # ---------------------------------------------------------------------------
# N_GPU = 1024
# section(f"6. CUDA BACKEND (All 7 methods, Single-channel N={N_GPU}, tol=1e-7)")
# if not has_cuda_backend():
#     print("  CUDA backend not available — skipped.")
# else:
#     rng = np.random.default_rng(42)
#     M0 = rng.normal(size=(N_GPU, N_GPU)) + 1j * rng.normal(size=(N_GPU, N_GPU))
#     M1 = rng.normal(size=(N_GPU, N_GPU)) + 1j * rng.normal(size=(N_GPU, N_GPU))
#     H0_raw = 0.5 * (M0 + M0.conj().T)
#     H1_raw = 0.5 * (M1 + M1.conj().T)
#     # Нормируем по 1-норме (||H0||_1 = 1.0, ||H1||_1 = 0.25), чтобы масштаб шага не зависел от sqrt(N)
#     H0_gpu = qt.Qobj(H0_raw / np.linalg.norm(H0_raw, 1))
#     H1_gpu = qt.Qobj(0.25 * H1_raw / np.linalg.norm(H1_raw, 1))
#     H_gpu = qt.QobjEvo([H0_gpu, [H1_gpu, lambda t: np.cos(1.5 * t)]])
#     psi0_n = qt.basis(N_GPU, 0)
#     tlist_gpu = np.linspace(0.0, 5.0, 6)  # 5 крупных интервалов

#     ref_gpu, ref_gpu_ms = run_ref(H_gpu, psi0_n, tlist_gpu)
#     print(f"Reference vern7 (N={N_GPU}, 1e-14): {ref_gpu_ms:.2f} ms")

#     # Прогрев CUDA-контекста перед замерами времени
#     _ = run_magnus(
#         H_gpu, psi0_n, np.array([0.0, 0.1]),
#         magnus_method="Formula314_6", backend="Cuda", adaptive=False, nsteps=1,
#     )

#     print(
#         f"  {'Method':12s} | {'CPU err (ms)':21s} | {'CUDA err (ms)':21s} | "
#         f"{'||CPU-CUDA||':12s} | {'steps':5s} | {'CPU/CUDA':8s}"
#     )
#     print("  " + "-" * 90)
#     for method in ALL_MAGNUS:
#         psi_c, ms_c, st_c, _ = run_magnus(
#             H_gpu, psi0_n, tlist_gpu, magnus_method=method, backend="Cpu", atol=1e-7, rtol=1e-7
#         )
#         psi_g, ms_g, st_g, _ = run_magnus(
#             H_gpu, psi0_n, tlist_gpu, magnus_method=method, backend="Cuda", atol=1e-7, rtol=1e-7
#         )
#         err_c = (psi_c - ref_gpu).norm()
#         err_g = (psi_g - ref_gpu).norm()
#         diff_cg = (psi_c - psi_g).norm()
#         speedup = ms_c / ms_g if ms_g > 0 else float("nan")
#         print(
#             f"  {method:12s} | {err_c:8.2e} ({ms_c:7.1f} ms) | {err_g:8.2e} ({ms_g:7.1f} ms) | "
#             f"{diff_cg:12.2e} | {st_g:5d} | {speedup:7.2f}x"
#         )