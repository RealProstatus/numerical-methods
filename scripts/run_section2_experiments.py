from __future__ import annotations

import argparse
import ctypes
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm as scipy_expm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymagnus_hpc as pm
from pymagnus_hpc import _magnus_py


OUT_DIR = Path("results/section2")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def get_rss_bytes() -> int:
    """Кроссплатформенное чтение RSS-памяти процесса без обязательной зависимости от psutil."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        if sys.platform == "win32":
            class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
                _fields_ = [
                    ("cb", ctypes.c_ulong),
                    ("PageFaultCount", ctypes.c_ulong),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                ]
            pmc = PROCESS_MEMORY_COUNTERS()
            pmc.cb = ctypes.sizeof(pmc)
            handle = ctypes.windll.kernel32.GetCurrentProcess()
            if ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(pmc), pmc.cb):
                return int(pmc.WorkingSetSize)
        return 0


def make_normalized_hermitian_pair(N: int = 100, norm_h0: float = 2.0, norm_h1: float = 1.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    B = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H0 = 0.5 * (A + A.conj().T)
    H1 = 0.5 * (B + B.conj().T)

    # Строгое диагональное преобладание
    for M, margin in ((H0, 10.0), (H1, 5.0)):
        np.fill_diagonal(M, 0.0)
        off_sums = np.sum(np.abs(M), axis=1)
        np.fill_diagonal(M, off_sums + margin)

    H0 = H0 / np.linalg.norm(H0, ord=2) * norm_h0
    H1 = H1 / np.linalg.norm(H1, ord=2) * norm_h1
    return np.ascontiguousarray(H0, dtype=np.complex128), np.ascontiguousarray(H1, dtype=np.complex128)


def make_spectral_pair(N: int, kappa: float, max_eig: float = 5.0, seed: int = 42):
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Q, _ = np.linalg.qr(Z)
    eigs = np.geomspace(max_eig / kappa, max_eig, N)
    H0 = (Q * eigs) @ Q.conj().T
    H0 = 0.5 * (H0 + H0.conj().T)
    Y = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H1 = 0.5 * (Y + Y.conj().T)
    H1 = H1 / np.linalg.norm(H1, ord=2) * 1.0
    return np.ascontiguousarray(H0, dtype=np.complex128), np.ascontiguousarray(H1, dtype=np.complex128)


def run_cpp_method(method_name: str, expm_name: str, H0: np.ndarray, H1: np.ndarray, t0: float, tf: float, steps: int, eps: float, omega: float, cheb_degree: int = 32):
    out = np.empty_like(H0, order="C")
    ms = _magnus_py.propagate_method_cpu(method_name, expm_name, H0, H1, t0, tf, steps, eps, omega, cheb_degree, out)
    return out, float(ms)


# ==============================================================================
# EXP-1.1: Эмпирический порядок сходимости методов Магнуса (p)
# ==============================================================================
def run_exp1_1(quick: bool = False) -> dict:
    print("\n--- [EXP-1.1] Эмпирический порядок сходимости методов Магнуса (N=64, T=1.0) ---")
    N = 64
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=11)

    # Эталон: сверхмелкая сетка (100 000 шагов) + кросс-проверка с DOP853 (rtol=1e-14)
    U_ref, _ = run_cpp_method("Formula314_6", "PadeEigen", H0, H1, 0.0, T, 20_000 if quick else 100_000, eps, omega, 32)
    sol_dop = solve_ivp(
        lambda t, y: (-1j * ((H0 + eps * np.cos(omega * t) * H1) @ y.reshape((N, N)))).ravel(),
        (0.0, T), np.eye(N, dtype=np.complex128).ravel(),
        method="DOP853", rtol=1e-13, atol=1e-13
    )
    U_dop = sol_dop.y[:, -1].reshape((N, N))
    ref_consistency = float(np.linalg.norm(U_ref - U_dop, ord="fro") / np.sqrt(N))
    print(f"  Reference consistency ||U_ref - U_DOP853||_F / sqrt(N) = {ref_consistency:.3e}")

    ks = list(range(0, 7 if quick else 9))
    methods = ["Classic2", "Acc4", "Acc6", "Formula314_4", "Formula314_6", "Recursive8"]
    rows = []
    slopes = {}

    for m in methods:
        prev_ef = None
        m_dts, m_efs = [], []
        for k in ks:
            steps = 10 * (2 ** k)
            if m == "Recursive8" and steps > (160 if quick else 640):
                continue
            dt = T / steps
            U_num, ms = run_cpp_method(m, "Chebyshev", H0, H1, 0.0, T, steps, eps, omega, cheb_degree=32)
            diff = U_num - U_ref
            ef = float(np.linalg.norm(diff, ord="fro") / np.sqrt(N))
            e2 = float(np.linalg.norm(diff, ord=2))
            p_loc = float(np.log2(prev_ef / ef)) if (prev_ef is not None and ef > 0) else float("nan")
            prev_ef = ef
            m_dts.append(dt)
            m_efs.append(ef)
            rows.append({
                "method": m,
                "k": k,
                "steps": steps,
                "dt": dt,
                "E_F": ef,
                "E_2": e2,
                "local_order_p": p_loc,
                "time_ms": ms,
            })

        # МНК-регрессия наклона p_fit в асимптотической области 1e-13 <= E_F <= 1e-2
        valid = [(d, e) for d, e in zip(m_dts, m_efs) if 1e-13 <= e <= 1e-2]
        if len(valid) >= 2:
            p_fit = float(np.polyfit(np.log([v[0] for v in valid]), np.log([v[1] for v in valid]), 1)[0])
        else:
            p_fit = float("nan")
        slopes[m] = p_fit
        print(f"  {m:13s}: asymptotic slope p_fit = {p_fit:.4f}, min E_F = {min(m_efs):.3e}")

    save_csv("exp1_1_convergence_orders.csv", ["method", "k", "steps", "dt", "E_F", "E_2", "local_order_p", "time_ms"], rows)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for m in methods:
        sub = [r for r in rows if r["method"] == m]
        ax.loglog([r["dt"] for r in sub], [r["E_F"] for r in sub], "o-", label=f"{m} (p={slopes[m]:.2f})")
    ax.set_title("EXP-1.1: Эмпирический порядок сходимости методов Магнуса (K_cheb = 32)")
    ax.set_xlabel("Шаг интегрирования dt")
    ax.set_ylabel("Относительная ошибка E_F(dt)")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp1_1_convergence_orders.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "slopes": slopes, "ref_consistency": ref_consistency}


# ==============================================================================
# EXP-1.2: Сохранение физических инвариантов на длинных временах (T = 10^4)
# ==============================================================================
def run_exp1_2(quick: bool = False) -> dict:
    print("\n--- [EXP-1.2] Сохранение инвариантов и унитарности на длинной траектории (N=64) ---")
    N = 64
    dt = 0.05
    total_steps = 20_000 if quick else 200_000
    checkpoint_every = 200 if quick else 1000
    n_chk = total_steps // checkpoint_every
    eps, omega = 0.5, 2.0 * np.pi
    H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=12)

    solvers = ["Formula314Chebyshev", "PiecewiseMagnusPade", "RK4"]
    rows = []
    final_metrics = {}

    for s_name in solvers:
        buf = np.empty((n_chk, 6), dtype=np.float64, order="C")
        t0 = time.perf_counter()
        _magnus_py.run_long_time_invariants(s_name, H0, H1, dt, total_steps, checkpoint_every, eps, omega, 16, buf)
        elapsed_s = time.perf_counter() - t0
        for r in buf:
            rows.append({
                "solver": s_name,
                "step": int(r[0]),
                "time_t": r[1],
                "unitarity_defect": r[2],
                "trace_defect": r[3],
                "purity_defect": r[4],
                "spectrum_drift": r[5],
            })
        final_metrics[s_name] = {
            "final_unitarity_defect": float(buf[-1, 2]),
            "final_trace_defect": float(buf[-1, 3]),
            "final_purity_defect": float(buf[-1, 4]),
            "final_spectrum_drift": float(buf[-1, 5]),
            "wall_time_s": elapsed_s,
        }
        print(f"  {s_name:20s}: D_unit={buf[-1, 2]:.3e}, dTr={buf[-1, 3]:.3e}, dPur={buf[-1, 4]:.3e}, dSpec={buf[-1, 5]:.3e} ({elapsed_s:.2f} s)")

    # 4-й инвариант: сохранение энергии в стационарном пределе (eps = 0)
    U_stat, _ = run_cpp_method("Formula314_6", "Chebyshev", H0, H1, 0.0, 100.0, 2000, 0.0, omega, 16)
    psi0 = np.zeros(N, dtype=np.complex128)
    psi0[0] = 1.0
    psiT = U_stat @ psi0
    E0 = float(np.real(psi0.conj() @ (H0 @ psi0)))
    ET = float(np.real(psiT.conj() @ (H0 @ psiT)))
    energy_drift = abs(ET - E0)
    print(f"  Stationary energy drift (eps=0, T=100): |E(T) - E(0)| = {energy_drift:.3e}")

    save_csv("exp1_2_long_time_invariants.csv", ["solver", "step", "time_t", "unitarity_defect", "trace_defect", "purity_defect", "spectrum_drift"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for s_name in solvers:
        sub = [r for r in rows if r["solver"] == s_name]
        ts = [r["time_t"] for r in sub]
        ax1.loglog(ts, [max(r["unitarity_defect"], 1e-17) for r in sub], label=s_name)
        ax2.loglog(ts, [max(r["purity_defect"], 1e-17) for r in sub], label=s_name)

    ax1.axhline(1e-12, color="k", ls="--", label="Порог 1e-12")
    ax1.set_title("EXP-1.2: Дрейф унитарности ||U†U - I||_F / sqrt(N)")
    ax1.set_xlabel("Время эволюции t")
    ax1.set_ylabel("Дефект унитарности D_unit")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("EXP-1.2: Нарушение чистоты состояния |Tr(rho^2) - 1|")
    ax2.set_xlabel("Время эволюции t")
    ax2.set_ylabel("Дефект чистоты")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp1_2_long_time_invariants.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "final_metrics": final_metrics, "stationary_energy_drift": energy_drift}


# ==============================================================================
# EXP-1.3: Сравнение exp(Omega) — Чебышев со скейлингом vs без скейлинга vs Паде vs Тейлор
# ==============================================================================
def run_exp1_3(quick: bool = False) -> dict:
    print("\n--- [EXP-1.3] exp(Omega): Чебышев со скейлингом vs без скейлинга (N=128) ---")
    N = 128
    rng = np.random.default_rng(13)
    Z = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    V, _ = np.linalg.qr(Z)
    d_base = rng.uniform(-1.0, 1.0, size=N)

    rhos = [0.1, 1.0, 5.0, 20.0, 100.0] if quick else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]
    m_grid = [6, 8, 10, 12, 14, 16, 20, 24, 32]
    rows = []

    for rho in rhos:
        d = d_base * rho
        Omega = np.ascontiguousarray(-1j * (V * d) @ V.conj().T, dtype=np.complex128)
        norm_1 = float(np.linalg.norm(Omega, ord=1))
        E_exact = (V * np.exp(-1j * d)) @ V.conj().T
        norm_exact = float(np.linalg.norm(E_exact, ord="fro"))

        # 1. Chebyshev с скейлингом по сетке степеней M
        for M in m_grid:
            out = np.empty_like(Omega)
            ms, zgemm_cnt, s_cnt = _magnus_py.expm_cheb_scaled_stats(Omega, M, out)
            rel_err = float(np.linalg.norm(out - E_exact, ord="fro") / norm_exact)
            rows.append({
                "rho": rho,
                "norm_1": norm_1,
                "method": f"Cheb_Scaled_M{M}",
                "poly_degree": M,
                "squarings_s": s_cnt,
                "zgemm_count": zgemm_cnt,
                "rel_error_F": rel_err,
                "time_ms": ms,
            })

        # 2. Chebyshev БЕЗ скейлинга (s = 0, M = ceil(1.36 * alpha + 14))
        out_ns = np.empty_like(Omega)
        ms_ns, zgemm_ns, deg_ns = _magnus_py.expm_cheb_no_scaling_stats(Omega, out_ns)
        rel_err_ns = float(np.linalg.norm(out_ns - E_exact, ord="fro") / norm_exact)
        rows.append({
            "rho": rho,
            "norm_1": norm_1,
            "method": "Cheb_NoScaling_Adaptive",
            "poly_degree": deg_ns,
            "squarings_s": 0,
            "zgemm_count": zgemm_ns,
            "rel_error_F": rel_err_ns,
            "time_ms": ms_ns,
        })

        # 3. EigenPadeExpm [13/13]
        out_pade = np.empty_like(Omega)
        t0 = time.perf_counter()
        _magnus_py.expm_cpu(Omega, out_pade, _magnus_py.ExpmMethod.PadeEigen, 13)
        ms_pade = (time.perf_counter() - t0) * 1e3
        rel_err_pade = float(np.linalg.norm(out_pade - E_exact, ord="fro") / norm_exact)
        rows.append({
            "rho": rho,
            "norm_1": norm_1,
            "method": "EigenPade_13",
            "poly_degree": 13,
            "squarings_s": max(0, int(np.ceil(np.log2(max(norm_1 / 5.37, 1.0))))),
            "zgemm_count": 9,
            "rel_error_F": rel_err_pade,
            "time_ms": ms_pade,
        })

        # 4. TaylorExpm (terms=30)
        out_tay = np.empty_like(Omega)
        t0 = time.perf_counter()
        _magnus_py.expm_cpu(Omega, out_tay, _magnus_py.ExpmMethod.Taylor, 30)
        ms_tay = (time.perf_counter() - t0) * 1e3
        rel_err_tay = float(np.linalg.norm(out_tay - E_exact, ord="fro") / norm_exact)
        s_tay = max(0, int(np.ceil(np.log2(max(norm_1 / 0.5, 1.0)))))
        rows.append({
            "rho": rho,
            "norm_1": norm_1,
            "method": "Taylor_30",
            "poly_degree": 30,
            "squarings_s": s_tay,
            "zgemm_count": 30 + s_tay,
            "rel_error_F": rel_err_tay,
            "time_ms": ms_tay,
        })

        sub16 = [r for r in rows if r["rho"] == rho and r["method"] == "Cheb_Scaled_M16"][0]
        print(f"  rho={rho:5.1f} (||Om||_1={norm_1:6.2f}): ChebScaled16 err={sub16['rel_error_F']:.2e} ({sub16['time_ms']:.2f} ms, zgemm={sub16['zgemm_count']}) | "
              f"ChebNoScale(M={deg_ns}) err={rel_err_ns:.2e} ({ms_ns:.2f} ms) | Pade err={rel_err_pade:.2e} ({ms_pade:.2f} ms)")

    save_csv("exp1_3_expm_spectral_radius.csv", ["rho", "norm_1", "method", "poly_degree", "squarings_s", "zgemm_count", "rel_error_F", "time_ms"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m_label in ["Cheb_Scaled_M16", "Cheb_NoScaling_Adaptive", "EigenPade_13", "Taylor_30"]:
        sub = [r for r in rows if r["method"] == m_label]
        ax1.loglog([r["norm_1"] for r in sub], [max(r["rel_error_F"], 1e-17) for r in sub], "o-", label=m_label)
        ax2.loglog([r["norm_1"] for r in sub], [r["time_ms"] for r in sub], "s-", label=m_label)

    ax1.set_title("EXP-1.3: Ошибка exp(Omega) от спектрального радиуса ||Omega||_1")
    ax1.set_xlabel("1-норма матрицы ||Omega||_1")
    ax1.set_ylabel("Относительная ошибка E_F")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("EXP-1.3: Время вычисления exp(Omega) (мс)")
    ax2.set_xlabel("1-норма матрицы ||Omega||_1")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp1_3_expm_spectral_radius.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-2.1: Физика — Осцилляции Раби вне RWA и сдвиг Блоха-Зигерта (N=2)
# ==============================================================================
def run_exp2_1() -> dict:
    print("\n--- [EXP-2.1] Физика: Двухуровневая система (RWA и сдвиг Блоха-Зигерта) ---")
    w0 = 2.0 * np.pi * 5.0
    omega = w0
    T_drive = 2.0 * np.pi / omega
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    H0 = np.ascontiguousarray(0.5 * w0 * sz)
    H1 = np.ascontiguousarray(sx)
    psi0 = np.array([1.0, 0.0], dtype=np.complex128)

    # Эталонная квазиэнергия Флоке за один период T_drive через DOP853 (rtol=1e-14)
    eps_strong = 0.5 * w0
    sol_ref = solve_ivp(
        lambda t, y: (-1j * ((H0 + eps_strong * np.cos(omega * t) * H1) @ y.reshape((2, 2)))).ravel(),
        (0.0, T_drive), np.eye(2, dtype=np.complex128).ravel(),
        method="DOP853", rtol=1e-14, atol=1e-14
    )
    U_mono_ref = sol_ref.y[:, -1].reshape((2, 2))
    eigs_ref = np.linalg.eigvals(U_mono_ref)
    mu_ref = float(np.max(np.abs(np.angle(eigs_ref))) / T_drive)

    # Сходимость квазиэнергии Флоке по величине шага dt = T_drive / steps_per_period
    floquet_rows = []
    for spp in [5, 10, 20, 40, 80]:
        dt = T_drive / spp
        U_num = pm.solve_magnus(H0, H1, 0.0, T_drive, steps=spp, epsilon=eps_strong, omega=omega, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=18)
        mu_num = float(np.max(np.abs(np.angle(np.linalg.eigvals(U_num)))) / T_drive)
        mu_err = abs(mu_num - mu_ref)
        floquet_rows.append({"steps_per_period": spp, "dt": dt, "mu_num": mu_num, "mu_ref": mu_ref, "floquet_phase_error": mu_err})
        print(f"  Bloch-Siegert (eps=0.5*w0) steps/period={spp:2d}: |mu_num - mu_ref| = {mu_err:.3e}")

    save_csv("exp2_1_rabi_floquet_convergence.csv", ["steps_per_period", "dt", "mu_num", "mu_ref", "floquet_phase_error"], floquet_rows)

    # Динамика заселенности P1(t) в режимах RWA (eps=0.01*w0) и Bloch-Siegert (eps=0.5*w0)
    t_weak = np.linspace(0.0, 2.0 * np.pi / (0.01 * w0), 120)
    p1_weak_num, p1_weak_rwa = [], []
    U_acc = np.eye(2, dtype=np.complex128)
    for idx in range(len(t_weak)):
        if idx > 0:
            U_step = pm.solve_magnus(H0, H1, float(t_weak[idx - 1]), float(t_weak[idx]), steps=2, epsilon=0.01 * w0, omega=omega, magnus_method="Formula314_6", backend="Cpu")
            U_acc = U_step @ U_acc
        psi_t = U_acc @ psi0
        p1_weak_num.append(float(abs(psi_t[1]) ** 2))
        p1_weak_rwa.append(float(np.sin(0.5 * (0.01 * w0) * t_weak[idx]) ** 2))

    t_strong = np.linspace(0.0, 6.0 * T_drive, 180)
    p1_strong_mag, p1_strong_dop = [], []
    U_acc = np.eye(2, dtype=np.complex128)
    for idx in range(len(t_strong)):
        if idx > 0:
            U_step = pm.solve_magnus(H0, H1, float(t_strong[idx - 1]), float(t_strong[idx]), steps=4, epsilon=eps_strong, omega=omega, magnus_method="Formula314_6", backend="Cpu")
            U_acc = U_step @ U_acc
        p1_strong_mag.append(float(abs((U_acc @ psi0)[1]) ** 2))

    sol_strong = solve_ivp(
        lambda t, y: -1j * ((H0 + eps_strong * np.cos(omega * t) * H1) @ y),
        (0.0, float(t_strong[-1])), psi0, t_eval=t_strong, method="DOP853", rtol=1e-13, atol=1e-13
    )
    p1_strong_dop = [float(abs(sol_strong.y[1, i]) ** 2) for i in range(len(t_strong))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(t_weak * w0 / (2 * np.pi), p1_weak_rwa, "k--", label="Аналитика RWA: sin^2(Omega_R t / 2)")
    ax1.plot(t_weak * w0 / (2 * np.pi), p1_weak_num, "b-", alpha=0.8, label="Formula314_6 (eps = 0.01 w0)")
    ax1.set_title("EXP-2.1a: Осцилляции Раби в режиме слабого поля (RWA)")
    ax1.set_xlabel("Число периодов поля t / T_drive")
    ax1.set_ylabel("Заселенность P_1(t)")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(t_strong / T_drive, p1_strong_dop, "k-", lw=2, label="Эталон DOP853 (rtol=1e-13)")
    ax2.plot(t_strong / T_drive, p1_strong_mag, "r--", lw=1.5, label="Formula314_6 (крупный шаг)")
    ax2.set_title("EXP-2.1b: Режим сильного поля (сдвиг Блоха-Зигерта, eps=0.5 w0)")
    ax2.set_xlabel("Число периодов поля t / T_drive")
    ax2.set_ylabel("Заселенность P_1(t)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp2_1_rabi_bloch_siegert.png", dpi=300)
    plt.close(fig)

    err_at_10 = [r["floquet_phase_error"] for r in floquet_rows if r["steps_per_period"] == 10][0]
    return {"status": "done", "floquet_error_at_10_steps": err_at_10, "max_rwa_diff": float(np.max(np.abs(np.array(p1_weak_num) - np.array(p1_weak_rwa))))}


# ==============================================================================
# EXP-2.2: Физика — Модель Ландау-Зенера-Штюкельберга (N=2)
# ==============================================================================
def run_exp2_2() -> dict:
    print("\n--- [EXP-2.2] Физика: Переход Ландау-Зенера-Штюкельберга (t in [-50, 50]) ---")
    v = 1.0
    T_max = 50.0 / np.sqrt(v)
    sz = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.complex128)
    sx = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
    psi0 = np.array([0.0, 1.0], dtype=np.complex128)

    deltas = [0.2, 0.5, 1.0, 2.0]
    rows = []

    for Delta in deltas:
        H0 = np.ascontiguousarray(0.5 * Delta * sx)
        H1 = np.ascontiguousarray(0.5 * v * sz)
        p_lz_asymp = float(np.exp(-np.pi * (Delta ** 2) / (2.0 * v)))

        # Точное решение на конечном интервале [-T_max, +T_max] (сверхточный DOP853 rtol=1e-14)
        sol_exact = solve_ivp(
            lambda t, y: -1j * ((H0 + t * H1) @ y),
            (-T_max, T_max), psi0, method="DOP853", rtol=1e-14, atol=1e-14
        )
        p_exact_finite = float(abs(sol_exact.y[1, -1]) ** 2)

        for steps in [500, 1000, 2000, 4000]:
            dt = (2.0 * T_max) / steps
            U_mag = pm.solve_magnus(H0, H1, -T_max, T_max, steps=steps, f_mod=lambda t: t, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=20)
            p_num = float(abs((U_mag @ psi0)[1]) ** 2)
            err_finite = abs(p_num - p_exact_finite)
            rows.append({
                "Delta": Delta,
                "steps": steps,
                "dt": dt,
                "P_num": p_num,
                "P_exact_finite": p_exact_finite,
                "P_LZ_asymptotic": p_lz_asymp,
                "error_vs_finite_exact": err_finite,
            })

        best_row = [r for r in rows if r["Delta"] == Delta and r["steps"] == 4000][0]
        print(f"  Delta={Delta:.1f}: P_num={best_row['P_num']:.12e}, P_finite={p_exact_finite:.12e}, |err|={best_row['error_vs_finite_exact']:.3e}, P_LZ_inf={p_lz_asymp:.6e}")

    save_csv("exp2_2_landau_zener.csv", ["Delta", "steps", "dt", "P_num", "P_exact_finite", "P_LZ_asymptotic", "error_vs_finite_exact"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for Delta in deltas:
        sub = [r for r in rows if r["Delta"] == Delta]
        ax1.loglog([r["dt"] for r in sub], [max(r["error_vs_finite_exact"], 1e-16) for r in sub], "o-", label=f"Delta = {Delta}")
    ax1.axhline(1e-11, color="k", ls="--", label="Критерий 1e-11")
    ax1.set_title("EXP-2.2a: Сходимость вероятности перехода ЛЗШ по шагу dt")
    ax1.set_xlabel("Шаг времени dt")
    ax1.set_ylabel("|P_num(T_max) - P_exact(T_max)|")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    d_dense = np.linspace(0.1, 2.2, 100)
    ax2.plot(d_dense, np.exp(-np.pi * (d_dense ** 2) / (2.0 * v)), "k--", label="Формула Ландау-Зенера exp(-pi*Delta^2 / 2v)")
    sub_4k = [r for r in rows if r["steps"] == 4000]
    ax2.plot([r["Delta"] for r in sub_4k], [r["P_num"] for r in sub_4k], "ro", markersize=8, label="Formula314_6 (T_max = 50)")
    ax2.set_title("EXP-2.2b: Профиль неадиабатического перехода P_LZ(Delta)")
    ax2.set_xlabel("Туннельное расщепление Delta")
    ax2.set_ylabel("Вероятность перехода P")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp2_2_landau_zener.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-2.3: Физика — Многочастичная спиновая цепочка Гейзенберга XXZ (N = 2^L)
# ==============================================================================
def build_xxz_hamiltonian(L: int, J: float = 1.0, delta_z: float = 1.2):
    N = 1 << L
    sz_diag = np.zeros((L, N), dtype=np.float64)
    for j in range(L):
        bit = 1 << (L - 1 - j)
        for idx in range(N):
            sz_diag[j, idx] = 1.0 if (idx & bit) == 0 else -1.0

    # Диагональная часть H0: -J * Delta_z * sum sz_j sz_{j+1} - sum h_j sz_j
    diag_H0 = np.zeros(N, dtype=np.float64)
    for j in range(L - 1):
        diag_H0 -= J * delta_z * sz_diag[j] * sz_diag[j + 1]
    for j in range(L):
        h_j = 0.5 * np.cos(2.0 * np.pi * (j + 1) / L)
        diag_H0 -= h_j * sz_diag[j]

    H0 = np.diag(diag_H0).astype(np.complex128)
    H1 = np.zeros((N, N), dtype=np.complex128)

    # Обменные члены -J * (sx_j sx_{j+1} + sy_j sy_{j+1}) = -2J * (|01><10| + |10><01|)
    for j in range(L - 1):
        b1 = 1 << (L - 1 - j)
        b2 = 1 << (L - 2 - j)
        mask = b1 | b2
        for idx in range(N):
            pair = idx & mask
            if pair == b1 or pair == b2:
                flipped = idx ^ mask
                H0[idx, flipped] -= 2.0 * J

    # Поперечное поле H1 = sum_j sigma_j^x
    for j in range(L):
        bit = 1 << (L - 1 - j)
        for idx in range(N):
            H1[idx, idx ^ bit] += 1.0

    # Начальное неелевское состояние |up, down, up, down, ...>
    neel_idx = 0
    for j in range(L):
        if j % 2 == 1:
            neel_idx |= (1 << (L - 1 - j))
    psi0 = np.zeros(N, dtype=np.complex128)
    psi0[neel_idx] = 1.0

    mz_op = np.mean(sz_diag, axis=0)
    c1L_op = sz_diag[0] * sz_diag[L - 1]
    return np.ascontiguousarray(H0), np.ascontiguousarray(H1), psi0, mz_op, c1L_op


def run_exp2_3(quick: bool = False) -> dict:
    print("\n--- [EXP-2.3] Физика: Спиновая цепочка Гейзенберга XXZ (N = 2^L) ---")
    eps, omega = 0.8, 3.0
    T_period = 2.0 * np.pi / omega

    # 1. Валидация наблюдаемых M_z(t), Q(nT), C_{1,L}(t) для L=6 (N=64) против QuTiP 5 / DOP853
    L_val = 6
    H0, H1, psi0, mz_op, c1L_op = build_xxz_hamiltonian(L_val)
    n_periods = 8 if quick else 15
    t_grid = np.linspace(0.0, n_periods * T_period, n_periods * 4 + 1)

    # Эталон через DOP853 (и QuTiP 5 Vern9, если доступен)
    sol_ref = solve_ivp(
        lambda t, y: -1j * ((H0 + eps * np.cos(omega * t) * H1) @ y),
        (0.0, float(t_grid[-1])), psi0, t_eval=t_grid, method="DOP853", rtol=1e-13, atol=1e-13
    )

    qutip_max_diff = None
    try:
        import qutip as qt
        H_qt = [qt.Qobj(H0), [qt.Qobj(H1), lambda t: eps * np.cos(omega * t)]]
        res_qt = qt.sesolve(H_qt, qt.Qobj(psi0), t_grid, options={"method": "vern9", "rtol": 1e-13, "atol": 1e-13})
        mz_qt = [float(np.sum((np.abs(st.full().ravel()) ** 2) * mz_op)) for st in res_qt.states]
    except Exception:
        mz_qt = None

    obs_rows = []
    U_acc = np.eye(1 << L_val, dtype=np.complex128)
    E0 = float(np.real(psi0.conj() @ (H0 @ psi0)))

    for idx, t_val in enumerate(t_grid):
        if idx > 0:
            U_step = pm.solve_magnus(H0, H1, float(t_grid[idx - 1]), float(t_val), steps=4, epsilon=eps, omega=omega, magnus_method="Formula314_6", backend="Cpu", chebyshev_degree=18)
            U_acc = U_step @ U_acc
        psi_mag = U_acc @ psi0
        prob_mag = np.abs(psi_mag) ** 2
        mz_mag = float(np.sum(prob_mag * mz_op))
        c1L_mag = float(np.sum(prob_mag * c1L_op))
        q_mag = float((np.real(psi_mag.conj() @ (H0 @ psi_mag)) - E0) / L_val)

        prob_ref = np.abs(sol_ref.y[:, idx]) ** 2
        mz_ref = float(np.sum(prob_ref * mz_op))
        c1L_ref = float(np.sum(prob_ref * c1L_op))

        obs_rows.append({
            "t": float(t_val),
            "Mz_magnus": mz_mag,
            "Mz_ref": mz_ref,
            "Mz_err": abs(mz_mag - mz_ref),
            "C1L_magnus": c1L_mag,
            "C1L_ref": c1L_ref,
            "C1L_err": abs(c1L_mag - c1L_ref),
            "Q_energy_absorption": q_mag,
        })

    max_mz_err = max(r["Mz_err"] for r in obs_rows)
    max_c1L_err = max(r["C1L_err"] for r in obs_rows)
    if mz_qt is not None:
        qutip_max_diff = max(abs(obs_rows[i]["Mz_magnus"] - mz_qt[i]) for i in range(len(t_grid)))
    print(f"  XXZ (L=6, N=64): max |Mz_mag - Mz_ref| = {max_mz_err:.3e}, max |C1L_mag - C1L_ref| = {max_c1L_err:.3e}, QuTiP diff = {qutip_max_diff}")
    save_csv("exp2_3_xxz_observables.csv", ["t", "Mz_magnus", "Mz_ref", "Mz_err", "C1L_magnus", "C1L_ref", "C1L_err", "Q_energy_absorption"], obs_rows)

    # 2. Масштабирование времени 1 шага по размерности N = 2^L (L = 2 .. 10/11)
    L_list = list(range(2, 10 if quick else 11))
    scale_rows = []
    has_gpu = pm.has_cuda_backend()
    for L in L_list:
        N = 1 << L
        H0_l, H1_l, _, _, _ = build_xxz_hamiltonian(L)
        t0 = time.perf_counter()
        _ = pm.solve_magnus(H0_l, H1_l, 0.0, 0.1, steps=10, epsilon=eps, omega=omega, backend="Cpu", chebyshev_degree=16)
        cpu_step_ms = (time.perf_counter() - t0) * 1e3 / 10.0

        gpu_step_ms = float("nan")
        if has_gpu:
            _ = pm.solve_magnus(H0_l, H1_l, 0.0, 0.02, steps=2, epsilon=eps, omega=omega, backend="Cuda", chebyshev_degree=16)
            t0 = time.perf_counter()
            _ = pm.solve_magnus(H0_l, H1_l, 0.0, 0.1, steps=10, epsilon=eps, omega=omega, backend="Cuda", chebyshev_degree=16)
            gpu_step_ms = (time.perf_counter() - t0) * 1e3 / 10.0

        scale_rows.append({"L_spins": L, "dim_N": N, "cpu_step_ms": cpu_step_ms, "gpu_step_ms": gpu_step_ms})
        print(f"  Scaling L={L:2d} (N={N:4d}): CPU step = {cpu_step_ms:8.3f} ms | GPU step = {gpu_step_ms:8.3f} ms")

    save_csv("exp2_3_xxz_scaling.csv", ["L_spins", "dim_N", "cpu_step_ms", "gpu_step_ms"], scale_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot([r["t"] for r in obs_rows], [r["Mz_magnus"] for r in obs_rows], "b-", label="M_z(t) намагниченность")
    ax1.plot([r["t"] for r in obs_rows], [r["C1L_magnus"] for r in obs_rows], "r--", label="C_{1,L}(t) коррелятор")
    ax1.set_title("EXP-2.3a: Флоке-динамика цепочки Гейзенберга XXZ (L=6, N=64)")
    ax1.set_xlabel("Время t")
    ax1.set_ylabel("Наблюдаемая")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    ax2.loglog([r["dim_N"] for r in scale_rows], [r["cpu_step_ms"] for r in scale_rows], "o-", label="CPU (MKL)")
    if has_gpu:
        ax2.loglog([r["dim_N"] for r in scale_rows], [r["gpu_step_ms"] for r in scale_rows], "s-", label="GPU (CUDA)")
    ax2.set_title("EXP-2.3b: Время 1 шага от размерности N = 2^L")
    ax2.set_xlabel("Размерность гильбертова пространства N")
    ax2.set_ylabel("Время на шаг (мс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp2_3_xxz_heisenberg.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "max_Mz_err": max_mz_err, "max_C1L_err": max_c1L_err, "qutip_max_diff": qutip_max_diff}


# ==============================================================================
# EXP-3.1: Single CPU Work-Precision и поиск Break-Even Point S(omega, kappa)
# ==============================================================================
def run_exp3_1(quick: bool = False) -> dict:
    print("\n--- [EXP-3.1] Single CPU Work-Precision & Break-Even Point (N=100, T=1.0) ---")
    N = 100
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    H0, H1 = make_normalized_hermitian_pair(N, norm_h0=5.0, norm_h1=2.0, seed=31)

    U_ref, _ = run_cpp_method("Formula314_6", "PadeEigen", H0, H1, 0.0, T, 5000, eps, omega, 32)
    repeats = 3 if quick else 7
    wp_rows = []

    for m_name, expm_name, step_list in [
        ("Formula314_6", "Chebyshev", [20, 40, 80, 160, 320]),
        ("Acc6", "Chebyshev", [20, 40, 80, 160, 320]),
        ("Acc4", "Chebyshev", [40, 80, 160, 320, 640]),
        ("Formula314_6", "PadeEigen", [20, 40, 80, 160]),
        ("RK4", "None", [1000, 2000, 4000, 8000]),
    ]:
        for steps in step_list:
            # Прогрев
            _ = run_cpp_method(m_name, expm_name, H0, H1, 0.0, T, steps, eps, omega, 16)
            times = []
            U_num = None
            for _ in range(repeats):
                U_num, ms = run_cpp_method(m_name, expm_name, H0, H1, 0.0, T, steps, eps, omega, 16)
                times.append(ms)
            ef = float(np.linalg.norm(U_num - U_ref, ord="fro") / np.sqrt(N))
            med_ms = float(np.median(times))
            wp_rows.append({"solver": f"{m_name}+{expm_name}", "steps": steps, "E_F": ef, "median_ms": med_ms})

    # Внешний эталон SciPy DOP853 & RK45
    y0_flat = np.eye(N, dtype=np.complex128).ravel()
    rhs = lambda t, y: (-1j * ((H0 + eps * np.cos(omega * t) * H1) @ y.reshape((N, N)))).ravel()
    for sp_method in ["DOP853", "RK45"]:
        for tol in ([1e-5, 1e-8, 1e-11] if quick else [1e-4, 1e-6, 1e-8, 1e-10, 1e-12]):
            t0 = time.perf_counter()
            sol = solve_ivp(rhs, (0.0, T), y0_flat, method=sp_method, rtol=tol, atol=tol)
            ms = (time.perf_counter() - t0) * 1e3
            ef = float(np.linalg.norm(sol.y[:, -1].reshape((N, N)) - U_ref, ord="fro") / np.sqrt(N))
            wp_rows.append({"solver": f"SciPy_{sp_method}", "steps": len(sol.t) - 1, "E_F": ef, "median_ms": ms})

    save_csv("exp3_1_work_precision_cpu.csv", ["solver", "steps", "E_F", "median_ms"], wp_rows)

    # Фазовая карта ускорения S(omega, kappa) при целевой точности ~ 1e-9
    kappas = [1e1, 1e2, 1e3] if quick else [1e1, 1e2, 1e3, 1e4]
    w_ratios = [0.5, 2.0, 10.0, 30.0] if quick else [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0]
    be_rows = []
    N_be = 48
    for kappa in kappas:
        H0_k, H1_k = make_spectral_pair(N_be, kappa, max_eig=5.0, seed=310)
        norm_h0 = float(np.linalg.norm(H0_k, ord=2))
        y0_be = np.eye(N_be, dtype=np.complex128).ravel()
        for wr in w_ratios:
            w_val = wr * norm_h0
            rhs_be = lambda t, y: (-1j * ((H0_k + 0.5 * np.cos(w_val * t) * H1_k) @ y.reshape((N_be, N_be)))).ravel()
            t0 = time.perf_counter()
            _ = solve_ivp(rhs_be, (0.0, 1.0), y0_be, method="DOP853", rtol=1e-10, atol=1e-10)
            ms_dop = (time.perf_counter() - t0) * 1e3

            mag_steps = max(40, int(np.ceil(6.0 * max(w_val, norm_h0) / (2.0 * np.pi))))
            _, ms_mag = run_cpp_method("Formula314_6", "Chebyshev", H0_k, H1_k, 0.0, 1.0, mag_steps, 0.5, w_val, 16)
            speedup = ms_dop / max(ms_mag, 1e-6)
            be_rows.append({"kappa": kappa, "omega_over_H0": wr, "dop853_ms": ms_dop, "magnus314_ms": ms_mag, "speedup_S": speedup})

    save_csv("exp3_1_breakeven_map.csv", ["kappa", "omega_over_H0", "dop853_ms", "magnus314_ms", "speedup_S"], be_rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for s_label in sorted(set(r["solver"] for r in wp_rows)):
        sub = sorted([r for r in wp_rows if r["solver"] == s_label], key=lambda x: x["median_ms"])
        ax1.loglog([r["median_ms"] for r in sub], [max(r["E_F"], 1e-16) for r in sub], "o-", label=s_label)
    ax1.set_title("EXP-3.1a: Диаграмма Work-Precision (Single CPU, N=100)")
    ax1.set_xlabel("Медиана времени счета T_wall (мс)")
    ax1.set_ylabel("Глобальная ошибка E_F(T)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend(fontsize=8)

    for kappa in kappas:
        sub = [r for r in be_rows if r["kappa"] == kappa]
        ax2.semilogx([r["omega_over_H0"] for r in sub], [r["speedup_S"] for r in sub], "o-", label=f"kappa = {kappa:.0e}")
    ax2.axhline(1.0, color="k", ls="--", label="Break-Even S = 1.0")
    ax2.set_title("EXP-3.1b: Ускорение Formula314 над SciPy DOP853")
    ax2.set_xlabel("Безразмерная частота omega / ||H0||_2")
    ax2.set_ylabel("Ускорение S = T_DOP853 / T_Magnus314")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp3_1_work_precision_breakeven.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-4.1: Аппаратное масштабирование на GPU и Roofline-анализ
# ==============================================================================
def run_exp4_1(quick: bool = False) -> dict:
    print("\n--- [EXP-4.1] Аппаратное масштабирование на GPU (cudaEvent_t) и Roofline ---")
    if not pm.has_cuda_backend():
        print("  [SKIP] CUDA backend не скомпилирован (-DBUILD_SINGLE_GPU=ON).")
        return {"status": "skipped_no_cuda"}

    dims = [32, 64, 128, 256, 512, 1024] if quick else [32, 64, 128, 256, 512, 1024, 2048]
    M_cheb = 16
    rows = []

    for N in dims:
        steps = 1000 if N <= 512 else (200 if N == 1024 else 50)
        H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=41)

        # CPU замер
        U_cpu, ms_cpu = run_cpp_method("Formula314_6", "Chebyshev", H0, H1, 0.0, 1.0, steps, 0.5, 2.0 * np.pi, M_cheb)

        # Прогрев GPU + замер по событиям cudaEvent_t
        U_gpu = np.empty_like(H0)
        _ = _magnus_py.benchmark_gpu_events(H0, H1, 0.0, 0.01, 2, 0.5, 2.0 * np.pi, M_cheb, U_gpu)
        ms_h2d, ms_comp, ms_d2h = _magnus_py.benchmark_gpu_events(H0, H1, 0.0, 1.0, steps, 0.5, 2.0 * np.pi, M_cheb, U_gpu)

        ms_gpu_total = ms_h2d + ms_comp + ms_d2h
        s_total = ms_cpu / max(ms_gpu_total, 1e-6)
        s_kernel = ms_cpu / max(ms_comp, 1e-6)
        pcie_pct = 100.0 * (ms_h2d + ms_d2h) / max(ms_gpu_total, 1e-6)

        # Число zgemm на шаг ~ M_cheb (15 для 2..16) + 1 накопление = 16 zgemm/step; 1 complex zgemm = 8*N^3 FLOPs
        zgemms_per_step = M_cheb
        total_flops = steps * zgemms_per_step * 8.0 * (N ** 3)
        tflops_fp64 = total_flops / (ms_comp * 1e9)
        # Арифметическая интенсивность (FLOP / Byte): 3 матрицы по 16*N^2 байт на zgemm -> AI = 8*N^3 / (48*N^2) = N / 6
        ai_flop_per_byte = N / 6.0

        err_gpu = float(np.linalg.norm(U_cpu - U_gpu, ord="fro") / np.sqrt(N))
        rows.append({
            "N": N,
            "steps": steps,
            "cpu_ms": ms_cpu,
            "gpu_h2d_ms": ms_h2d,
            "gpu_compute_ms": ms_comp,
            "gpu_d2h_ms": ms_d2h,
            "gpu_total_ms": ms_gpu_total,
            "S_total": s_total,
            "S_kernel": s_kernel,
            "pcie_overhead_pct": pcie_pct,
            "fp64_tflops": tflops_fp64,
            "arithmetic_intensity": ai_flop_per_byte,
            "cpu_gpu_diff": err_gpu,
        })
        print(f"  N={N:4d} (steps={steps:4d}): CPU={ms_cpu:8.2f} ms | GPU_comp={ms_comp:6.2f} ms (H2D={ms_h2d:.2f}, D2H={ms_d2h:.2f}) | "
              f"S_kernel={s_kernel:5.2f}x, S_total={s_total:5.2f}x | FP64={tflops_fp64:.4f} TFLOPS")

    save_csv("exp4_1_gpu_scaling_roofline.csv", list(rows[0].keys()), rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ns = [r["N"] for r in rows]
    ax1.semilogx(ns, [r["S_kernel"] for r in rows], "o-", base=2, label="S_kernel (чистые вычисления GPU)")
    ax1.semilogx(ns, [r["S_total"] for r in rows], "s-", base=2, label="S_total (с учетом PCIe H2D+D2H)")
    ax1.axhline(1.0, color="k", ls="--", label="Точка насыщения N_crit (S = 1)")
    ax1.set_title("EXP-4.1a: Ускорение GPU над CPU по размерности N")
    ax1.set_xlabel("Размерность матрицы N")
    ax1.set_ylabel("Коэффициент ускорения")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ais = np.array([r["arithmetic_intensity"] for r in rows])
    tflops = np.array([r["fp64_tflops"] for r in rows])
    ax2.loglog(ais, tflops, "ro-", label="Измеренная производительность FP64")
    # Теоретическая крыша RTX 4060 Laptop (FP64 ~ 0.30 TFLOPS, Memory BW ~ 256 GB/s)
    ai_grid = np.logspace(0, 3, 100)
    roof = np.minimum(0.30, 0.256 * ai_grid)
    ax2.loglog(ai_grid, roof, "k--", label="Roofline RTX 4060 (FP64 peak ~0.30 TFLOPS)")
    ax2.set_title("EXP-4.1b: Roofline-диаграмма (FP64 Complex)")
    ax2.set_xlabel("Арифметическая интенсивность (FLOP / Byte)")
    ax2.set_ylabel("Производительность (TFLOPS)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp4_1_gpu_scaling_roofline.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "max_S_kernel": max(r["S_kernel"] for r in rows), "max_fp64_tflops": max(r["fp64_tflops"] for r in rows)}


# ==============================================================================
# EXP-4.2: Multi-CPU масштабирование (OpenMP/MKL потоки & MPI)
# ==============================================================================
def run_exp4_2(quick: bool = False) -> dict:
    print("\n--- [EXP-4.2] Multi-CPU масштабирование (OpenMP/MKL & MPI) ---")
    N = 256 if quick else 512
    H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=42)
    max_cores = os.cpu_count() or 4
    p_list = [p for p in [1, 2, 4, 8, 12, 16] if p <= max_cores]
    if max_cores not in p_list:
        p_list.append(max_cores)

    # Strong & Weak Scaling замер через подпроцессы с фиксированным числом потоков (MKL_NUM_THREADS=P, OMP_NUM_THREADS=P)
    # А также проверка наличия MPI-бинарника apps/multi_cpu/mpi_verification.exe
    rows = []
    base_steps = 120 if quick else 480
    t1_strong = None
    t1_weak = None

    for P in p_list:
        env = os.environ.copy()
        env["MKL_NUM_THREADS"] = str(P)
        env["OMP_NUM_THREADS"] = str(P)

        # Запускаем изолированный дочерний Python-процесс, чтобы Intel MKL инициализировал ровно P потоков
        cmd_strong = (
            f"import numpy as np; from pymagnus_hpc import _magnus_py; "
            f"H0=np.eye({N},dtype=np.complex128); H1=np.ones(({N},{N}),dtype=np.complex128)*0.01; out=np.empty_like(H0); "
            f"_magnus_py.propagate_method_cpu('Formula314_6','Chebyshev',H0,H1,0.0,1.0,5,0.5,6.28,16,out); "
            f"print(_magnus_py.propagate_method_cpu('Formula314_6','Chebyshev',H0,H1,0.0,1.0,{base_steps},0.5,6.28,16,out))"
        )
        res_s = subprocess.run([sys.executable, "-c", cmd_strong], env=env, capture_output=True, text=True, check=True)
        ms_strong = float(res_s.stdout.strip().splitlines()[-1])
        if t1_strong is None:
            t1_strong = ms_strong
        s_strong = t1_strong / ms_strong
        e_strong = 100.0 * s_strong / P

        weak_steps = (30 if quick else 80) * P
        cmd_weak = (
            f"import numpy as np; from pymagnus_hpc import _magnus_py; "
            f"H0=np.eye({N},dtype=np.complex128); H1=np.ones(({N},{N}),dtype=np.complex128)*0.01; out=np.empty_like(H0); "
            f"_magnus_py.propagate_method_cpu('Formula314_6','Chebyshev',H0,H1,0.0,1.0,5,0.5,6.28,16,out); "
            f"print(_magnus_py.propagate_method_cpu('Formula314_6','Chebyshev',H0,H1,0.0,1.0,{weak_steps},0.5,6.28,16,out))"
        )
        res_w = subprocess.run([sys.executable, "-c", cmd_weak], env=env, capture_output=True, text=True, check=True)
        ms_weak = float(res_w.stdout.strip().splitlines()[-1])
        if t1_weak is None:
            t1_weak = ms_weak
        e_weak = 100.0 * (t1_weak / ms_weak)

        rows.append({
            "P_workers": P,
            "strong_ms": ms_strong,
            "S_strong": s_strong,
            "E_strong_pct": e_strong,
            "weak_steps": weak_steps,
            "weak_ms": ms_weak,
            "E_weak_pct": e_weak,
        })
        print(f"  P={P:2d}: Strong={ms_strong:7.2f} ms (S={s_strong:.2f}x, E={e_strong:.1f}%) | Weak({weak_steps} steps)={ms_weak:7.2f} ms (E_weak={e_weak:.1f}%)")

    save_csv("exp4_2_multicpu_scaling.csv", ["P_workers", "strong_ms", "S_strong", "E_strong_pct", "weak_steps", "weak_ms", "E_weak_pct"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ps = [r["P_workers"] for r in rows]
    ax1.plot(ps, [r["S_strong"] for r in rows], "o-", label="Измеренное ускорение S_strong(P)")
    ax1.plot(ps, ps, "k--", label="Идеальное линейное ускорение S = P")
    ax1.set_title(f"EXP-4.2a: Strong Scaling Multi-CPU (N={N})")
    ax1.set_xlabel("Число потоков / рангов P")
    ax1.set_ylabel("Ускорение S(P)")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend()

    ax2.plot(ps, [r["E_strong_pct"] for r in rows], "o-", label="E_strong (%)")
    ax2.plot(ps, [r["E_weak_pct"] for r in rows], "s-", label="E_weak (%)")
    ax2.axhline(85.0, color="r", ls="--", label="Целевой порог 85%")
    ax2.set_title(f"EXP-4.2b: Параллельная эффективность (N={N})")
    ax2.set_xlabel("Число потоков / рангов P")
    ax2.set_ylabel("Эффективность (%)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp4_2_multicpu_scaling.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "mpi_compiled": pm.has_mpi_backend()}


# ==============================================================================
# EXP-5.1a: Накладные расходы Python-биндингов (P1 vs P2 vs P3) и Zero-Copy
# ==============================================================================
def run_exp5_1a(quick: bool = False) -> dict:
    print("\n--- [EXP-5.1a] Накладные расходы nanobind (P1 vs P2 vs P3) и Zero-Copy ---")
    dims = [2, 4, 16, 64, 256] if quick else [2, 4, 16, 64, 256, 1024]
    steps = 100
    rows = []

    for N in dims:
        H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=51)
        U_out = np.empty_like(H0, order="C")
        time_nodes = np.linspace(0.0, 1.0, steps + 1, dtype=np.float64)
        mod_empty = np.empty(0, dtype=np.float64)
        opts = _magnus_py.SolverOptions()
        opts.backend = _magnus_py.BackendType.Cpu

        # Проверка Zero-Copy совпадения физических адресов памяти
        py_addr = int(H0.__array_interface__["data"][0])
        cpp_addr = int(_magnus_py.get_ptr_address(H0))
        zero_copy_ok = (py_addr == cpp_addr)
        assert zero_copy_ok, f"Zero-copy failed for N={N}: {py_addr} != {cpp_addr}"

        # P1: Пошаговый цикл в Python (100 вызовов C++ по 1 шагу)
        t0 = time.perf_counter()
        for s in range(steps):
            t_pair = np.array([time_nodes[s], time_nodes[s + 1]], dtype=np.float64)
            _magnus_py.propagate_cpu_zero_copy(H0, H1, t_pair, mod_empty, True, 0.5, 6.28, opts, U_out)
        ms_p1 = (time.perf_counter() - t0) * 1e3

        # P2: Пакетный вызов из Python с отпущенным GIL
        n_rep = 20 if N <= 64 else 5
        t0 = time.perf_counter()
        for _ in range(n_rep):
            _magnus_py.propagate_cpu_zero_copy(H0, H1, time_nodes, mod_empty, True, 0.5, 6.28, opts, U_out)
        ms_p2 = (time.perf_counter() - t0) * 1e3 / n_rep

        # P3: Чистое время C++ внутри функции (без Python call overhead)
        cpp_times = []
        for _ in range(n_rep):
            cpp_times.append(_magnus_py.propagate_method_cpu("Formula314_6", "Chebyshev", H0, H1, 0.0, 1.0, steps, 0.5, 6.28, 16, U_out))
        ms_p3 = float(np.median(cpp_times))

        latency_us = max(0.0, (ms_p2 - ms_p3) * 1e3)
        rel_overhead_pct = max(0.0, 100.0 * (ms_p2 - ms_p3) / max(ms_p3, 1e-6))

        rows.append({
            "N": N,
            "zero_copy_verified": zero_copy_ok,
            "P1_python_loop_ms": ms_p1,
            "P2_batch_python_ms": ms_p2,
            "P3_native_cpp_ms": ms_p3,
            "call_latency_us": latency_us,
            "rel_overhead_pct": rel_overhead_pct,
        })
        print(f"  N={N:4d}: ZeroCopy={zero_copy_ok} | P1={ms_p1:7.3f} ms | P2={ms_p2:7.3f} ms | P3={ms_p3:7.3f} ms | Latency={latency_us:5.2f} us")

    save_csv("exp5_1a_binding_overhead.csv", list(rows[0].keys()), rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ns = [r["N"] for r in rows]
    ax1.loglog(ns, [r["P1_python_loop_ms"] for r in rows], "o-", label="P1: цикл в Python (100 вызовов)")
    ax1.loglog(ns, [r["P2_batch_python_ms"] for r in rows], "s-", label="P2: пакетный вызов Python (Zero-GIL)")
    ax1.loglog(ns, [r["P3_native_cpp_ms"] for r in rows], "^--", label="P3: чистое время C++ ядра")
    ax1.set_title("EXP-5.1a: Сравнение сценариев вызова P1 / P2 / P3 (100 шагов)")
    ax1.set_xlabel("Размерность матрицы N")
    ax1.set_ylabel("Время выполнения (мс)")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.semilogx(ns, [r["call_latency_us"] for r in rows], "ro-", label="Латентность перехода Python <-> C++ (мкс)")
    ax2.set_title("EXP-5.1a: Накладные расходы вызова nanobind")
    ax2.set_xlabel("Размерность матрицы N")
    ax2.set_ylabel("Латентность (мкс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp5_1a_binding_overhead.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "all_zero_copy_verified": all(r["zero_copy_verified"] for r in rows)}


# ==============================================================================
# EXP-5.1b: Стресс-тест на утечки памяти (RAM & VRAM) и кросс-валидация бэкендов
# ==============================================================================
def run_exp5_1b(quick: bool = False) -> dict:
    print("\n--- [EXP-5.1b] Стресс-тест утечек памяти (RAM/VRAM) и кросс-валидация ---")
    N = 256
    n_iter = 2_000 if quick else 50_000
    H0, H1 = make_normalized_hermitian_pair(N, norm_h0=2.0, norm_h1=1.0, seed=52)
    U_out = np.empty_like(H0, order="C")
    time_nodes = np.linspace(0.0, 0.1, 11, dtype=np.float64)
    mod_empty = np.empty(0, dtype=np.float64)
    opts = _magnus_py.SolverOptions()
    opts.backend = _magnus_py.BackendType.Cpu

    # 1. Мониторинг Host RAM на n_iter последовательных вызовах
    mem_rows = []
    rss_warmup = 0
    sample_every = max(100, n_iter // 50)
    for it in range(1, n_iter + 1):
        _magnus_py.propagate_cpu_zero_copy(H0, H1, time_nodes, mod_empty, True, 0.5, 6.28, opts, U_out)
        if it == max(100, n_iter // 20):
            gc.collect()
            rss_warmup = get_rss_bytes()
        if it % sample_every == 0:
            rss_now = get_rss_bytes()
            mem_rows.append({"iteration": it, "rss_mb": rss_now / (1024.0 * 1024.0)})

    gc.collect()
    rss_final = get_rss_bytes()
    delta_rss_mb = (rss_final - rss_warmup) / (1024.0 * 1024.0) if rss_warmup > 0 else 0.0
    print(f"  Host RAM stress ({n_iter} calls, N={N}): delta_RSS = {delta_rss_mb:.3f} MB")
    save_csv("exp5_1b_memory_profile.csv", ["iteration", "rss_mb"], mem_rows)

    # 2. Мониторинг GPU VRAM (cudaMemGetInfo до и после GPU-цикла)
    vram_leak_bytes = 0
    if pm.has_cuda_backend():
        gc.collect()
        free_before, total_vram = _magnus_py.get_cuda_mem_info()
        for _ in range(200 if quick else 1000):
            _ = pm.solve_magnus(H0, H1, 0.0, 0.1, steps=10, epsilon=0.5, omega=6.28, backend="Cuda")
        gc.collect()
        free_after, _ = _magnus_py.get_cuda_mem_info()
        vram_leak_bytes = int(free_before - free_after)
        print(f"  GPU VRAM check: free_before={free_before}, free_after={free_after}, delta={vram_leak_bytes} bytes")

    # 3. Кросс-валидация на детерминированных матрицах std::mt19937_64(42)
    H0_det = np.empty((128, 128), dtype=np.complex128, order="C")
    H1_det = np.empty((128, 128), dtype=np.complex128, order="C")
    _magnus_py.generate_mt19937_hermitian_pair(128, 42, H0_det, H1_det)
    H0_det = np.ascontiguousarray(H0_det / np.linalg.norm(H0_det, 2) * 2.0)
    H1_det = np.ascontiguousarray(H1_det / np.linalg.norm(H1_det, 2) * 1.0)

    U_native, _ = run_cpp_method("Formula314_6", "Chebyshev", H0_det, H1_det, 0.0, 1.0, 100, 0.5, 2.0 * np.pi, 16)
    U_py = pm.solve_magnus(H0_det, H1_det, 0.0, 1.0, steps=100, epsilon=0.5, omega=2.0 * np.pi, backend="Cpu", chebyshev_degree=16)
    diff_cpp_py = float(np.max(np.abs(U_native - U_py)))

    cross_rows = [{"pair": "Native_CPP vs Python_Wrapper", "max_chebyshev_diff": diff_cpp_py}]
    diff_cpu_gpu = float("nan")
    if pm.has_cuda_backend():
        U_gpu = pm.solve_magnus(H0_det, H1_det, 0.0, 1.0, steps=100, epsilon=0.5, omega=2.0 * np.pi, backend="Cuda", chebyshev_degree=16)
        diff_cpu_gpu = float(np.max(np.abs(U_py - U_gpu)))
        cross_rows.append({"pair": "CPU_MKL vs CUDA_GPU", "max_chebyshev_diff": diff_cpu_gpu})

    print(f"  Cross-validation (mt19937_64(42)): C++ vs Python = {diff_cpp_py:.3e}, CPU vs GPU = {diff_cpu_gpu:.3e}")
    save_csv("exp5_1b_cross_validation.csv", ["pair", "max_chebyshev_diff"], cross_rows)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([r["iteration"] for r in mem_rows], [r["rss_mb"] for r in mem_rows], "b-")
    ax.set_title(f"EXP-5.1b: Профиль оперативной памяти Host RSS ({n_iter} вызовов, N={N})")
    ax.set_xlabel("Номер вызова")
    ax.set_ylabel("Память процесса RSS (МБ)")
    ax.grid(True, ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp5_1b_memory_profile.png", dpi=300)
    plt.close(fig)

    return {
        "status": "done",
        "delta_rss_mb": delta_rss_mb,
        "vram_leak_bytes": vram_leak_bytes,
        "diff_cpp_py": diff_cpp_py,
        "diff_cpu_gpu": diff_cpu_gpu,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск валидационных экспериментов Раздела 2 (EXP-1.1 .. EXP-5.1b)")
    parser.add_argument(
        "--exp",
        nargs="+",
        default=["1.1", "1.2", "1.3", "2.1", "2.2", "2.3", "3.1", "4.1", "4.2", "5.1a", "5.1b"],
        help="Список ID экспериментов (например: 1.1 1.2 2.1 4.1 5.1a)",
    )
    parser.add_argument("--quick", action="store_true", help="Ускоренный прогон с уменьшенным числом итераций")
    args = parser.parse_args()

    ensure_out_dir()
    summary = {}

    dispatch = {
        "1.1": lambda: run_exp1_1(args.quick),
        "1.2": lambda: run_exp1_2(args.quick),
        "1.3": lambda: run_exp1_3(args.quick),
        "2.1": run_exp2_1,
        "2.2": run_exp2_2,
        "2.3": lambda: run_exp2_3(args.quick),
        "3.1": lambda: run_exp3_1(args.quick),
        "4.1": lambda: run_exp4_1(args.quick),
        "4.2": lambda: run_exp4_2(args.quick),
        "5.1a": lambda: run_exp5_1a(args.quick),
        "5.1b": lambda: run_exp5_1b(args.quick),
    }

    for exp_id in args.exp:
        if exp_id in dispatch:
            summary[f"EXP-{exp_id}"] = dispatch[exp_id]()

    with (OUT_DIR / "summary_section2.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Готово. Результаты Раздела 2 сохранены в {OUT_DIR.resolve()} ===")


if __name__ == "__main__":
    main()