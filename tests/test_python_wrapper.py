import time
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm as scipy_expm

import pymagnus_hpc as pm
from pymagnus_hpc.scipy_plugin import MagnusOdeSolver


def reference_propagator_dop853(H0: np.ndarray, H1: np.ndarray, f_scalar, t0: float, tf: float) -> np.ndarray:
    """Независимый эталонный пропагатор через SciPy DOP853 (rtol=1e-13, atol=1e-13)."""
    N = H0.shape[0]
    y0 = np.eye(N, dtype=np.complex128).ravel()

    def rhs(t, y_flat):
        U = y_flat.reshape((N, N))
        H_t = H0 + f_scalar(t) * H1
        dU = -1j * (H_t @ U)
        return dU.ravel()

    sol = solve_ivp(rhs, (t0, tf), y0, method="DOP853", rtol=1e-13, atol=1e-13)
    assert sol.success, f"DOP853 failed: {sol.message}"
    return sol.y[:, -1].reshape((N, N))


def make_test_hamiltonians(N: int = 4, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H0 = 0.5 * (A + A.conj().T)
    H1 = 0.5 * (B + B.conj().T)
    return np.ascontiguousarray(H0), np.ascontiguousarray(H1)


def test_expm_methods():
    N = 8
    H0, _ = make_test_hamiltonians(N)
    Omega = -1j * H0
    ref = scipy_expm(Omega)

    for method in ["Chebyshev", "PadeEigen", "Taylor"]:
        res = pm.expm_magnus(Omega, method=method, degree=20)
        err = np.linalg.norm(res - ref, ord="fro")
        assert err < 1e-13, f"expm_magnus({method}) error too high: {err}"
    print("[PASS] expm_magnus (Chebyshev, PadeEigen, Taylor) matches scipy.linalg.expm")


def test_harmonic_driving_and_orders():
    N = 4
    H0, H1 = make_test_hamiltonians(N)
    eps, omega = 0.6, 3.5
    t0, tf = 0.0, 1.5

    U_ref = reference_propagator_dop853(H0, H1, lambda t: eps * np.cos(omega * t), t0, tf)

    # Проверяем все реализованные методы Магнуса
    for method, steps, tol in [
        ("Classic2", 400, 1e-4),
        ("Classic4", 100, 1e-4),
        ("Acc4", 100, 1e-7),
        ("Acc6", 50, 1e-10),
        ("Formula314_6", 50, 2e-10),
        ("Recursive", 100, 1e-4),
    ]:
        U_num = pm.solve_magnus(
            H0, H1, t0, tf, steps=steps,
            epsilon=eps, omega=omega,
            magnus_method=method, expm_method="Chebyshev", backend="Cpu", chebyshev_degree=20
        )
        err = np.linalg.norm(U_num - U_ref, ord="fro")
        unit_def = np.linalg.norm(U_num.conj().T @ U_num - np.eye(N), ord="fro") / np.sqrt(N)
        assert err < tol, f"{method}: error {err:.2e} >= {tol:.2e}"
        assert unit_def < 1e-12, f"{method}: unitarity defect {unit_def:.2e}"

    # Проверка эмпирического порядка Formula314_6 (должен быть ~6)
    U_10 = pm.solve_magnus(H0, H1, t0, tf, steps=10, epsilon=eps, omega=omega, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=24)
    U_20 = pm.solve_magnus(H0, H1, t0, tf, steps=20, epsilon=eps, omega=omega, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=24)
    p_emp = np.log2(np.linalg.norm(U_10 - U_ref, "fro") / np.linalg.norm(U_20 - U_ref, "fro"))
    assert p_emp > 5.5, f"Expected ~6th order for Formula314_6, got p={p_emp:.2f}"
    print(f"[PASS] Harmonic driving verified against DOP853 (Formula314_6 empirical order p = {p_emp:.2f})")


def test_arbitrary_modulation():
    N = 4
    H0, H1 = make_test_hamiltonians(N)
    t0, tf = 0.0, 2.0

    # Гауссов импульс (негармоническая модуляция f_mod)
    f_gauss = lambda t: 0.9 * np.exp(-((t - 1.0) / 0.35) ** 2)
    U_ref = reference_propagator_dop853(H0, H1, f_gauss, t0, tf)

    U_num = pm.solve_magnus(
        H0, H1, t0, tf, steps=400,
        f_mod=f_gauss, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=20
    )
    err = np.linalg.norm(U_num - U_ref, ord="fro")
    assert err < 1e-7, f"Arbitrary modulation error too high: {err:.2e}"
    print(f"[PASS] Arbitrary modulation (Gaussian pulse) verified against DOP853 (err = {err:.2e})")


def test_zero_copy_and_overhead():
    N = 16
    H0, H1 = make_test_hamiltonians(N)
    time_nodes = np.linspace(0.0, 1.0, 11, dtype=np.float64)
    mod_grid = np.empty(0, dtype=np.float64)
    U_out = np.empty((N, N), dtype=np.complex128, order="C")

    ptr_before = U_out.__array_interface__["data"][0]
    opts = pm.SolverOptions()
    opts.backend = pm.BackendType.Cpu

    pm.propagate_cpu_zero_copy(H0, H1, time_nodes, mod_grid, True, 0.5, 2.0, opts, U_out)
    ptr_after = U_out.__array_interface__["data"][0]

    assert ptr_before == ptr_after, "Output buffer pointer changed (copy occurred!)"
    assert np.allclose(U_out.conj().T @ U_out, np.eye(N), atol=1e-13)

    # Замер накладных расходов вызова (500 прогонов)
    t_start = time.perf_counter()
    for _ in range(500):
        pm.propagate_cpu_zero_copy(H0, H1, time_nodes, mod_grid, True, 0.5, 2.0, opts, U_out)
    avg_us = (time.perf_counter() - t_start) / 500 * 1e6
    print(f"[PASS] Zero-copy buffer verified (avg call time for N=16, 10 steps: {avg_us:.1f} us)")


def test_cuda_backend_if_available():
    if not pm.has_cuda_backend():
        print("[SKIP] CUDA backend not available on this machine")
        return

    N = 128
    H0, H1 = make_test_hamiltonians(N)
    U_cpu = pm.solve_magnus(H0, H1, 0.0, 1.0, steps=100, epsilon=0.5, omega=3.0, backend="Cpu")
    U_gpu = pm.solve_magnus(H0, H1, 0.0, 1.0, steps=100, epsilon=0.5, omega=3.0, backend="Cuda")
    err = np.linalg.norm(U_cpu - U_gpu, ord="fro")
    assert err < 1e-12, f"CUDA vs CPU mismatch: {err:.2e}"
    print(f"[PASS] CUDA backend verified against CPU (N={N}, err = {err:.2e})")

    try:
        import cupy as cp
        d_H0 = cp.asarray(H0)
        d_H1 = cp.asarray(H1)
        d_U = pm.solve_magnus(d_H0, d_H1, 0.0, 1.0, steps=100, epsilon=0.5, omega=3.0, backend="Cuda")
        assert isinstance(d_U, cp.ndarray)
        err_dlpack = float(cp.linalg.norm(d_U - cp.asarray(U_cpu)))
        assert err_dlpack < 1e-12
        print(f"[PASS] CuPy DLPack GPU path verified (err = {err_dlpack:.2e})")
    except ImportError:
        print("[SKIP] CuPy not installed; skipping DLPack GPU test")


def test_ecosystem_plugins():
    N = 2
    H0 = np.array([[1.5, 0.0], [0.0, -1.5]], dtype=np.complex128)
    H1 = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    eps, omega, tf = 0.4, 3.0, 2.0
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    U_ref = reference_propagator_dop853(H0, H1, lambda t: eps * np.cos(omega * t), 0.0, tf)
    psi_ref = U_ref @ psi0

    # 1. SciPy OdeSolver plugin
    sol_scipy = solve_ivp(
        fun=lambda t, y: y,
        t_span=(0.0, tf),
        y0=psi0,
        method=MagnusOdeSolver,
        H0=H0,
        H1=H1,
        epsilon=eps,
        omega=omega,
        dt=0.05,
        backend="Cpu",
    )
    assert sol_scipy.success
    err_scipy = np.linalg.norm(sol_scipy.y[:, -1] - psi_ref)
    assert err_scipy < 5e-10, f"SciPy MagnusOdeSolver error: {err_scipy:.2e}"
    print(f"[PASS] SciPy plugin (MagnusOdeSolver) verified (err = {err_scipy:.2e})")

    # 2. QuTiP 5 plugin
    try:
        import qutip as qt
        H_qt = [qt.Qobj(H0), [qt.Qobj(H1), lambda t: eps * np.cos(omega * t)]]
        res_qt = qt.sesolve(
            H_qt,
            qt.Qobj(psi0),
            [0.0, tf],
            options={"method": "magnus_chebyshev", "dt": 0.05, "backend": "Cpu"},
        )
        psi_qt = res_qt.states[-1].full().ravel()
        err_qt = np.linalg.norm(psi_qt - psi_ref)
        assert err_qt < 5e-10, f"QuTiP plugin error: {err_qt:.2e}"
        print(f"[PASS] QuTiP 5 plugin ('magnus_chebyshev') verified (err = {err_qt:.2e})")
    except ImportError:
        print("[SKIP] QuTiP not installed; skipping QuTiP plugin test")

    # 3. Qiskit Dynamics plugin
    try:
        from qiskit_dynamics import Solver
        from qiskit_dynamics.signals import Signal
        from pymagnus_hpc.qiskit_plugin import solve_lmde_magnus_hpc

        qd_solver = Solver(
            static_hamiltonian=H0,
            hamiltonian_operators=[H1],
            hamiltonian_signals=[Signal(envelope=eps, carrier_freq=omega / (2 * np.pi))],
        )
        res_qd = qd_solver.solve(
            t_span=[0.0, tf],
            y0=psi0,
            method=solve_lmde_magnus_hpc,
            max_dt=0.05,
            backend="Cpu",
        )
        err_qd = np.linalg.norm(res_qd.y[-1] - psi_ref)
        assert err_qd < 5e-10, f"Qiskit Dynamics plugin error: {err_qd:.2e}"
        print(f"[PASS] Qiskit Dynamics plugin verified (err = {err_qd:.2e})")
    except ImportError:
        print("[SKIP] qiskit-dynamics not installed; skipping Qiskit plugin test")


if __name__ == "__main__":
    test_expm_methods()
    test_harmonic_driving_and_orders()
    test_arbitrary_modulation()
    test_zero_copy_and_overhead()
    test_cuda_backend_if_available()
    test_ecosystem_plugins()
    print("=== ALL PYTHON WRAPPER TESTS PASSED ===")