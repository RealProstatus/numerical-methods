from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pymagnus_hpc as pm
from pymagnus_hpc import _magnus_py


OUT_DIR = Path("results/section1")


def ensure_out_dir() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)


def save_csv(filename: str, fieldnames: list[str], rows: list[dict]) -> None:
    path = OUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_hermitian_pair(N: int = 100, seed: int = 42, norm_scale: float = 2.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    Y = rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))
    H0 = 0.5 * (X + X.conj().T)
    H1 = 0.5 * (Y + Y.conj().T)

    # Добавляем строгое диагональное преобладание |H_ii| > sum_{j != i} |H_ij|
    for M, margin in ((H0, 10.0), (H1, 5.0)):
        np.fill_diagonal(M, 0.0)
        off_sums = np.sum(np.abs(M), axis=1)
        np.fill_diagonal(M, off_sums + margin)

    H0 = H0 / np.linalg.norm(H0, ord=2) * norm_scale
    H1 = H1 / np.linalg.norm(H1, ord=2) * (0.5 * norm_scale)
    return np.ascontiguousarray(H0, dtype=np.complex128), np.ascontiguousarray(H1, dtype=np.complex128)


def make_spectral_scaled_pair(N: int, kappa: float, seed: int = 42, max_eig: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Генерирует эрмитову матрицу H0 с заданным числом обусловленности kappa = lambda_max / lambda_min."""
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


def metrics_diff(A: np.ndarray, B: np.ndarray) -> tuple[float, float]:
    diff = A - B
    max_elem = float(np.max(np.abs(diff)))
    max_eig = float(np.max(np.abs(np.linalg.eigvals(diff))))
    return max_elem, max_eig


def call_omega(method: str, order: int, H0: np.ndarray, H1: np.ndarray, t0: float, dt: float, eps: float, omega: float, sub_steps: int = 10, repeats: int = 5):
    out = np.empty_like(H0, order="C")
    times = []
    for _ in range(repeats):
        ms = _magnus_py.compute_omega_step(method, order, H0, H1, t0, dt, eps, omega, sub_steps, out)
        times.append(ms)
    return out, float(np.median(times))


def call_rk4(H0: np.ndarray, H1: np.ndarray, t0: float, tf: float, steps: int, eps: float, omega: float):
    out = np.empty_like(H0, order="C")
    ms = _magnus_py.propagate_rk4_mkl(H0, H1, t0, tf, steps, eps, omega, out)
    return out, float(ms)


def call_piecewise_acc(H0: np.ndarray, H1: np.ndarray, t0: float, tf: float, macro_steps: int, inner_steps: int, eps: float, omega: float, order: int = 4, cheb_degree: int = 16):
    out = np.empty_like(H0, order="C")
    ms = _magnus_py.propagate_piecewise_acc_inner(
        H0, H1, t0, tf, macro_steps, inner_steps, eps, omega, order, cheb_degree, out
    )
    return out, float(ms)


# ==============================================================================
# EXP-01: Точность и время вычисления матрицы Omega при гармоническом возмущении
# ==============================================================================
def run_exp01(quick: bool = False) -> dict:
    print("\n--- [EXP-01] Точность и время вычисления матрицы Omega (N=128, dt=1e-5) ---")
    N = 128
    dt = 1e-5
    eps, omega = 1.0, 2.0 * np.pi
    A, B = make_hermitian_pair(N, seed=101, norm_scale=3.0)

    Omega_ref, _ = call_omega("Recursive", 10, A, B, 0.0, dt, eps, omega, sub_steps=10, repeats=1)

    configs = [
        ("Classic", [1, 2], {1: 0, 2: 10}),
        ("ACC", [1, 2, 3, 4, 5, 6], {1: 0, 2: 1, 3: 1, 4: 1, 5: 4, 6: 4}),
        ("Recursive", list(range(1, 7 if quick else 9)), {k: k * 15 for k in range(1, 9)}),
        ("Formula314", [4, 6], {4: 7, 6: 7}),
    ]

    rows = []
    for method, orders, ops_map in configs:
        for k in orders:
            reps = 1 if (method == "Recursive" and k >= 6) else 5
            Om, ms = call_omega(method, k, A, B, 0.0, dt, eps, omega, sub_steps=10, repeats=reps)
            max_el, max_eig = metrics_diff(Om, Omega_ref)
            rows.append({
                "method": method,
                "order_k": k,
                "max_element_diff": max_el,
                "max_eig_diff": max_eig,
                "time_ms": ms,
                "zgemm_ops": ops_map[k],
            })
            print(f"  {method:10s} k={k}: MaxElem={max_el:.3e}, MaxEig={max_eig:.3e}, Time={ms:.3f} ms, zgemm={ops_map[k]}")

    save_csv("exp01_omega_accuracy_time.csv", ["method", "order_k", "max_element_diff", "max_eig_diff", "time_ms", "zgemm_ops"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for method in ["Classic", "ACC", "Recursive", "Formula314"]:
        sub = [r for r in rows if r["method"] == method]
        ks = [r["order_k"] for r in sub]
        errs = [max(r["max_element_diff"], 1e-20) for r in sub]
        tms = [r["time_ms"] for r in sub]
        ax1.semilogy(ks, errs, marker="o", label=method)
        ax2.semilogy(ks, tms, marker="s", label=method)

    ax1.set_title("EXP-01: Максимальная ошибка элементов Omega")
    ax1.set_xlabel("Порядок аппроксимации k")
    ax1.set_ylabel("Max Element Diff")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("EXP-01: Время вычисления Omega (мс)")
    ax2.set_xlabel("Порядок аппроксимации k")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp01_omega_accuracy_time.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "rows": len(rows)}


# ==============================================================================
# EXP-02: Точность и сходимость exp(Omega) (Taylor vs Chebyshev vs Pade)
# ==============================================================================
def run_exp02() -> dict:
    print("\n--- [EXP-02] Сходимость матричной экспоненты exp(Omega) (N=128, dt=1e-5) ---")
    N = 128
    dt = 1e-5
    A, B = make_hermitian_pair(N, seed=102, norm_scale=3.0)
    Omega, _ = call_omega("ACC", 4, A, B, 0.0, dt, 1.0, 2.0 * np.pi, sub_steps=6, repeats=1)

    t0 = time.perf_counter()
    E_ref = pm.expm_magnus(Omega, method="PadeEigen")
    pade_ms = (time.perf_counter() - t0) * 1e3
    print(f"  Reference Eigen-Pade time = {pade_ms:.3f} ms")

    rows = []
    # Taylor terms 1..30
    for terms in range(1, 31):
        out = np.empty_like(Omega)
        times = []
        for _ in range(5):
            ts = time.perf_counter()
            _magnus_py.expm_cpu(Omega, out, _magnus_py.ExpmMethod.Taylor, terms)
            times.append((time.perf_counter() - ts) * 1e3)
        max_el, max_eig = metrics_diff(out, E_ref)
        rows.append({
            "method": "Taylor",
            "degree_or_terms": terms,
            "max_element_diff": max_el,
            "max_eig_diff": max_eig,
            "time_ms": float(np.median(times)),
        })

    # Chebyshev M 1..25
    for deg in range(1, 26):
        out = np.empty_like(Omega)
        times = []
        for _ in range(5):
            ts = time.perf_counter()
            _magnus_py.expm_cpu(Omega, out, _magnus_py.ExpmMethod.Chebyshev, deg)
            times.append((time.perf_counter() - ts) * 1e3)
        max_el, max_eig = metrics_diff(out, E_ref)
        rows.append({
            "method": "Chebyshev",
            "degree_or_terms": deg,
            "max_element_diff": max_el,
            "max_eig_diff": max_eig,
            "time_ms": float(np.median(times)),
        })

    save_csv("exp02_expm_convergence.csv", ["method", "degree_or_terms", "max_element_diff", "max_eig_diff", "time_ms"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for method in ["Taylor", "Chebyshev"]:
        sub = [r for r in rows if r["method"] == method]
        xs = [r["degree_or_terms"] for r in sub]
        errs = [max(r["max_element_diff"], 1e-17) for r in sub]
        tms = [r["time_ms"] for r in sub]
        ax1.semilogy(xs, errs, marker="o", markersize=4, label=method)
        ax2.plot(xs, tms, marker="s", markersize=4, label=method)

    ax2.axhline(pade_ms, color="k", ls="--", label=f"Eigen-Pade ({pade_ms:.2f} ms)")
    ax1.set_title("EXP-02: Сходимость exp(Omega) к Eigen-Pade")
    ax1.set_xlabel("Число членов K / Степень полинома M")
    ax1.set_ylabel("Max Element Diff")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("EXP-02: Время вычисления exp(Omega) (мс)")
    ax2.set_xlabel("Число членов K / Степень полинома M")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp02_expm_convergence.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "pade_ms": pade_ms}


# ==============================================================================
# EXP-03: Устойчивость и Work-Precision на полной траектории (RK4 vs Magnus ACC)
# ==============================================================================
def run_exp03(quick: bool = False) -> dict:
    print("\n--- [EXP-03] Устойчивость и Work-Precision (RK4 vs Piecewise Magnus ACC, N=128, T=1.0) ---")
    N = 128
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    # Спектральный радиус ~ 350, чтобы при dt > 1e-3 явный RK4 терял устойчивость
    H0, H1 = make_hermitian_pair(N, seed=103, norm_scale=350.0)

    ref_steps = 100_000 if quick else 1_000_000
    U_ref, ref_ms = call_rk4(H0, H1, 0.0, T, ref_steps, eps, omega)
    print(f"  Reference RK4 ({ref_steps} steps) computed in {ref_ms:.1f} ms")

    rk4_dts = [1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4, 5e-5]
    rows = []
    for dt in rk4_dts:
        steps = int(round(T / dt))
        U_rk, ms = call_rk4(H0, H1, 0.0, T, steps, eps, omega)
        max_el = float(np.max(np.abs(U_rk - U_ref)))
        rows.append({"solver": "RK4", "dt": dt, "steps": steps, "max_element_diff": max_el, "time_ms": ms})
        print(f"  RK4 dt={dt:.1e}: MaxElem={max_el:.3e}, Time={ms:.2f} ms")

    for macro_dt in [2e-2, 1e-2, 5e-3]:
        m_steps = int(round(T / macro_dt))
        U_mag, ms = call_piecewise_acc(H0, H1, 0.0, T, m_steps, inner_steps=6, eps=eps, omega=omega, order=6, cheb_degree=20)
        max_el = float(np.max(np.abs(U_mag - U_ref)))
        rows.append({"solver": "Piecewise_Magnus_ACC", "dt": macro_dt, "steps": m_steps, "max_element_diff": max_el, "time_ms": ms})
        print(f"  Magnus ACC macro_dt={macro_dt:.1e}: MaxElem={max_el:.3e}, Time={ms:.2f} ms")

    save_csv("exp03_work_precision_stability.csv", ["solver", "dt", "steps", "max_element_diff", "time_ms"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    rk_sub = [r for r in rows if r["solver"] == "RK4"]
    mg_sub = [r for r in rows if r["solver"] == "Piecewise_Magnus_ACC"]

    ax1.loglog([r["dt"] for r in rk_sub], [min(r["max_element_diff"], 1e150) for r in rk_sub], "o-", label="RK4 (явный)")
    ax1.loglog([r["dt"] for r in mg_sub], [r["max_element_diff"] for r in mg_sub], "s-", label="Piecewise Magnus ACC")
    ax1.set_title("EXP-03: Граница устойчивости и глобальная ошибка U(T)")
    ax1.set_xlabel("Шаг интегрирования dt / macro_dt")
    ax1.set_ylabel("Max Element Diff")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    rk_stable = [r for r in rk_sub if r["max_element_diff"] < 1.0]
    ax2.loglog([r["time_ms"] for r in rk_stable], [r["max_element_diff"] for r in rk_stable], "o-", label="RK4 (устойчивая ветвь)")
    ax2.loglog([r["time_ms"] for r in mg_sub], [r["max_element_diff"] for r in mg_sub], "s-", label="Piecewise Magnus ACC")
    ax2.set_title("EXP-03: Диаграмма Work-Precision (Точность - Время)")
    ax2.set_xlabel("Время расчета траектории (мс)")
    ax2.set_ylabel("Max Element Diff")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp03_work_precision_stability.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-04: Устойчивость вычисления Omega при спектральном масштабировании kappa
# ==============================================================================
def run_exp04() -> dict:
    print("\n--- [EXP-04] Устойчивость вычисления Omega от жесткости kappa in {10..10^4} ---")
    N = 128
    dt = 1e-5
    eps, omega = 1.0, 2.0 * np.pi
    kappas = [1e1, 1e2, 1e3, 1e4]
    methods = [("Classic", 2), ("ACC", 4), ("Recursive", 4), ("Formula314", 4)]

    rows = []
    for kappa in kappas:
        A, B = make_spectral_scaled_pair(N, kappa, seed=104, max_eig=5.0)
        Om_ref, _ = call_omega("Recursive", 10, A, B, 0.0, dt, eps, omega, sub_steps=10, repeats=1)
        for m_name, order in methods:
            reps = 1 if m_name == "Recursive" else 5
            Om, ms = call_omega(m_name, order, A, B, 0.0, dt, eps, omega, sub_steps=10, repeats=reps)
            max_el, max_eig = metrics_diff(Om, Om_ref)
            rows.append({
                "kappa": kappa,
                "method": m_name,
                "order": order,
                "max_element_diff": max_el,
                "max_eig_diff": max_eig,
                "time_ms": ms,
            })
            print(f"  kappa={kappa:7.0f} | {m_name:10s}: MaxElem={max_el:.3e}, MaxEig={max_eig:.3e}, Time={ms:.2f} ms")

    save_csv("exp04_omega_spectral_scaling.csv", ["kappa", "method", "order", "max_element_diff", "max_eig_diff", "time_ms"], rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m_name, _ in methods:
        sub = [r for r in rows if r["method"] == m_name]
        ax.loglog([r["kappa"] for r in sub], [max(r["max_eig_diff"], 1e-20) for r in sub], "o-", label=m_name)

    ax.set_title("EXP-04: Стабильность Omega при росте числа обусловленности kappa")
    ax.set_xlabel("Число обусловленности kappa")
    ax.set_ylabel("Max Eig Diff")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp04_omega_spectral_scaling.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-05: Устойчивость exp(Omega) при спектральном масштабировании kappa
# ==============================================================================
def run_exp05() -> dict:
    print("\n--- [EXP-05] Устойчивость exp(Omega) при спектральном масштабировании kappa ---")
    N = 128
    dt = 1e-5
    kappas = [1e1, 1e2, 1e3, 1e4]
    rows = []

    for kappa in kappas:
        A, B = make_spectral_scaled_pair(N, kappa, seed=105, max_eig=5.0)
        Omega, _ = call_omega("ACC", 4, A, B, 0.0, dt, 1.0, 2.0 * np.pi, sub_steps=6, repeats=1)
        E_ref = pm.expm_magnus(Omega, method="PadeEigen")

        for m_name, enum_val, deg in [
            ("Chebyshev", _magnus_py.ExpmMethod.Chebyshev, 16),
            ("Taylor", _magnus_py.ExpmMethod.Taylor, 30),
        ]:
            out = np.empty_like(Omega)
            times = []
            for _ in range(5):
                ts = time.perf_counter()
                _magnus_py.expm_cpu(Omega, out, enum_val, deg)
                times.append((time.perf_counter() - ts) * 1e3)
            max_el, max_eig = metrics_diff(out, E_ref)
            rows.append({
                "kappa": kappa,
                "method": m_name,
                "max_element_diff": max_el,
                "max_eig_diff": max_eig,
                "time_ms": float(np.median(times)),
            })
            print(f"  kappa={kappa:7.0f} | {m_name:9s}: MaxElem={max_el:.3e}, MaxEig={max_eig:.3e}, Time={np.median(times):.3f} ms")

    save_csv("exp05_expm_spectral_scaling.csv", ["kappa", "method", "max_element_diff", "max_eig_diff", "time_ms"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for m_name in ["Chebyshev", "Taylor"]:
        sub = [r for r in rows if r["method"] == m_name]
        ax1.loglog([r["kappa"] for r in sub], [max(r["max_element_diff"], 1e-17) for r in sub], "o-", label=m_name)
        ax2.semilogx([r["kappa"] for r in sub], [r["time_ms"] for r in sub], "s-", label=m_name)

    ax1.set_title("EXP-05: Ошибка exp(Omega) при росте жесткости kappa")
    ax1.set_xlabel("Число обусловленности kappa")
    ax1.set_ylabel("Max Element Diff")
    ax1.grid(True, which="both", ls="--", alpha=0.5)
    ax1.legend()

    ax2.set_title("EXP-05: Время вычисления exp(Omega) (мс)")
    ax2.set_xlabel("Число обусловленности kappa")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp05_expm_spectral_scaling.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-06: Устойчивость решения ОДУ на всей траектории при вариации kappa
# ==============================================================================
def run_exp06(quick: bool = False) -> dict:
    print("\n--- [EXP-06] Устойчивость траектории U(T) при спектральном масштабировании kappa ---")
    N = 128
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    kappas = [1e1, 1e2, 1e3, 1e4]
    ref_steps = 100_000 if quick else 1_000_000

    rows = []
    for kappa in kappas:
        A, B = make_spectral_scaled_pair(N, kappa, seed=106, max_eig=5.0)
        U_ref, _ = call_rk4(A, B, 0.0, T, ref_steps, eps, omega)

        U_rk4, ms_rk4 = call_rk4(A, B, 0.0, T, 10_000, eps, omega)  # dt = 1e-4
        el_rk4, eig_rk4 = metrics_diff(U_rk4, U_ref)

        U_mag, ms_mag = call_piecewise_acc(A, B, 0.0, T, 100, inner_steps=6, eps=eps, omega=omega, order=6, cheb_degree=18)
        el_mag, eig_mag = metrics_diff(U_mag, U_ref)

        speedup = ms_rk4 / max(ms_mag, 1e-9)
        rows.append({"kappa": kappa, "method": "RK4_dt1e-4", "max_element_diff": el_rk4, "max_eig_diff": eig_rk4, "time_ms": ms_rk4, "speedup": 1.0})
        rows.append({"kappa": kappa, "method": "Piecewise_ACC_dt0.01", "max_element_diff": el_mag, "max_eig_diff": eig_mag, "time_ms": ms_mag, "speedup": speedup})
        print(f"  kappa={kappa:7.0f} | RK4={el_rk4:.2e} ({ms_rk4:.1f} ms) | Magnus={el_mag:.2e} ({ms_mag:.1f} ms, {speedup:.1f}x)")

    save_csv("exp06_ode_spectral_scaling.csv", ["kappa", "method", "max_element_diff", "max_eig_diff", "time_ms", "speedup"], rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    for m_name in ["RK4_dt1e-4", "Piecewise_ACC_dt0.01"]:
        sub = [r for r in rows if r["method"] == m_name]
        ax.loglog([r["kappa"] for r in sub], [r["max_element_diff"] for r in sub], "o-", label=m_name)

    ax.set_title("EXP-06: Глобальная ошибка U(T) при вариации жесткости kappa")
    ax.set_xlabel("Число обусловленности kappa")
    ax.set_ylabel("Max Element Diff")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp06_ode_spectral_scaling.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-07: Сходимость RK4 при последовательном делении шага dt пополам
# ==============================================================================
def run_exp07(quick: bool = False) -> dict:
    print("\n--- [EXP-07] Сходимость метода RK4 от шага dt (N=128, T=1.0) ---")
    N = 128
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    H0, H1 = make_hermitian_pair(N, seed=107, norm_scale=3.0)

    # Эталон: Piecewise Magnus Solver (macro_dt = 1e-3, order = 6 для эталонной точности)
    U_ref, _ = call_piecewise_acc(H0, H1, 0.0, T, 1000, inner_steps=6, eps=eps, omega=omega, order=6, cheb_degree=20)

    max_pow = 8 if quick else 11  # в полном режиме до 1000 * 2^11 = 2,048,000 шагов (dt = 4.88e-7)
    rows = []
    for p in range(max_pow + 1):
        steps = 1000 * (2 ** p)
        dt = T / steps
        U_rk, ms = call_rk4(H0, H1, 0.0, T, steps, eps, omega)
        max_el = float(np.max(np.abs(U_rk - U_ref)))
        rows.append({"steps": steps, "dt": dt, "max_element_diff": max_el, "time_ms": ms})
        print(f"  steps={steps:7d}, dt={dt:.3e}: MaxElem={max_el:.3e}, Time={ms:.1f} ms")

    save_csv("exp07_rk4_convergence.csv", ["steps", "dt", "max_element_diff", "time_ms"], rows)

    fig, ax = plt.subplots(figsize=(8, 5))
    dts = np.array([r["dt"] for r in rows])
    errs = np.array([r["max_element_diff"] for r in rows])
    ax.loglog(dts, errs, "o-", label="RK4 (эмпирическая ошибка)")
    ref_line = errs[0] * (dts / dts[0]) ** 4
    ax.loglog(dts, ref_line, "k--", label="Теоретическая асимптотика O(dt^4)")

    ax.set_title("EXP-07: Сходимость RK4 и выход на плато машинного округления")
    ax.set_xlabel("Шаг интегрирования dt")
    ax.set_ylabel("Max Element Diff")
    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp07_rk4_convergence.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-08: Самосходимость (правило Рунге) кусочного метода Магнуса по macro_dt
# ==============================================================================
def run_exp08(quick: bool = False) -> dict:
    print("\n--- [EXP-08] Самосходимость Piecewise Magnus по величине macro_dt ---")
    N = 128
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    H0, H1 = make_hermitian_pair(N, seed=108, norm_scale=3.0)
    dt_inner_target = 1e-5 if quick else 1e-6

    macro_counts = [10, 20, 40, 80, 160, 320, 640]
    rows = []
    prev_U = None
    for m_steps in macro_counts:
        macro_dt = T / m_steps
        inner_steps = max(2, int(round(macro_dt / dt_inner_target)))
        U_curr, ms = call_piecewise_acc(H0, H1, 0.0, T, m_steps, inner_steps=inner_steps, eps=eps, omega=omega, order=4, cheb_degree=16)
        diff_prev = float(np.max(np.abs(U_curr - prev_U))) if prev_U is not None else float("nan")
        prev_U = U_curr
        rows.append({
            "macro_segments": m_steps,
            "macro_dt": macro_dt,
            "inner_steps_per_macro": inner_steps,
            "diff_with_prev_step": diff_prev,
            "time_ms": ms,
        })
        print(f"  segments={m_steps:3d}, macro_dt={macro_dt:.6f}: DiffPrev={diff_prev:.3e}, Time={ms:.1f} ms")

    save_csv("exp08_piecewise_self_convergence.csv", ["macro_segments", "macro_dt", "inner_steps_per_macro", "diff_with_prev_step", "time_ms"], rows)

    valid = [r for r in rows if not np.isnan(r["diff_with_prev_step"])]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.loglog([r["macro_dt"] for r in valid], [r["diff_with_prev_step"] for r in valid], "o-")
    ax1.set_title("EXP-08: Внутренняя самосходимость Рунге (Diff with prev)")
    ax1.set_xlabel("Макрошаг macro_dt")
    ax1.set_ylabel("Max Element Diff с предыдущей сеткой")
    ax1.grid(True, which="both", ls="--", alpha=0.5)

    ax2.loglog([r["macro_dt"] for r in rows], [r["time_ms"] for r in rows], "s-", color="tab:orange")
    ax2.set_title("EXP-08: Вычислительные затраты (мс)")
    ax2.set_xlabel("Макрошаг macro_dt")
    ax2.set_ylabel("Время (мс)")
    ax2.grid(True, which="both", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp08_piecewise_self_convergence.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-09: Адаптивный выбор внутреннего шага inner_dt по 20 макросегментам
# ==============================================================================
def run_exp09() -> dict:
    print("\n--- [EXP-09] Адаптивный подбор внутреннего шага inner_dt (20 макросегментов, Tol=1e-12) ---")
    N = 128
    T = 1.0
    n_segments = 20
    macro_dt = T / n_segments
    tol = 1e-12
    eps, omega = 1.5, 4.0 * np.pi
    H0, H1 = make_hermitian_pair(N, seed=109, norm_scale=3.0)

    rows = []
    for seg in range(n_segments):
        ts = seg * macro_dt
        te = ts + macro_dt
        inner_steps = 2
        halvings = 0
        t_seg_start = time.perf_counter()

        U_prev, _ = call_piecewise_acc(H0, H1, ts, te, 1, inner_steps, eps, omega, order=4, cheb_degree=18)
        while halvings < 14:
            inner_steps *= 2
            halvings += 1
            U_curr, _ = call_piecewise_acc(H0, H1, ts, te, 1, inner_steps, eps, omega, order=4, cheb_degree=18)
            err = float(np.max(np.abs(U_curr - U_prev)))
            U_prev = U_curr
            if err < tol:
                break

        seg_ms = (time.perf_counter() - t_seg_start) * 1e3
        inner_dt = macro_dt / inner_steps
        rows.append({
            "segment": seg + 1,
            "t_start": round(ts, 4),
            "t_end": round(te, 4),
            "inner_dt": inner_dt,
            "halvings": halvings,
            "time_ms": seg_ms,
        })
        print(f"  Seg {seg+1:2d} [{ts:.2f}, {te:.2f}]: halvings={halvings}, inner_dt={inner_dt:.3e}, time={seg_ms:.2f} ms")

    save_csv("exp09_adaptive_inner_step.csv", ["segment", "t_start", "t_end", "inner_dt", "halvings", "time_ms"], rows)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar([r["segment"] for r in rows], [r["halvings"] for r in rows], color="steelblue", edgecolor="black")
    ax.set_title("EXP-09: Количество делений внутреннего шага по сегментам времени")
    ax.set_xlabel("Номер макросегмента (macro_dt = 0.05)")
    ax.set_ylabel("Число делений шага пополам")
    ax.set_xticks(range(1, n_segments + 1))
    ax.grid(True, axis="y", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp09_adaptive_inner_step.png", dpi=300)
    plt.close(fig)
    return {"status": "done"}


# ==============================================================================
# EXP-10: Сравнение RK4 (CPU) и MagnusChebyshevGPU (CUDA) при N=100, T=1.0
# ==============================================================================
def run_exp10(quick: bool = False) -> dict:
    print("\n--- [EXP-10] Производительность и точность: RK4 (CPU) vs MagnusChebyshevGPU (CUDA) ---")
    N = 100
    T = 1.0
    eps, omega = 1.0, 2.0 * np.pi
    H0, H1 = make_hermitian_pair(N, seed=110, norm_scale=3.0)

    ref_steps = 100_000 if quick else 1_000_000
    U_ref, ref_ms = call_rk4(H0, H1, 0.0, T, ref_steps, eps, omega)
    print(f"  Reference RK4 ({ref_steps} steps): {ref_ms:.1f} ms")

    # 1. RK4 на CPU (dt = 1e-4 -> 10 000 шагов)
    U_rk4, ms_rk4 = call_rk4(H0, H1, 0.0, T, 10_000, eps, omega)
    err_rk4 = float(np.max(np.abs(U_rk4 - U_ref)))

    # 2. GPU CUDA (dt = 1e-3 -> 1000 шагов) или fallback на CPU Formula314_6
    backend = "Cuda" if pm.has_cuda_backend() else "Cpu"
    # Прогрев GPU контекста
    _ = pm.solve_magnus(H0, H1, 0.0, 0.01, steps=10, epsilon=eps, omega=omega, backend=backend, chebyshev_degree=16)

    t0 = time.perf_counter()
    U_gpu = pm.solve_magnus(H0, H1, 0.0, T, steps=1000, epsilon=eps, omega=omega, backend=backend, chebyshev_degree=16)
    ms_gpu = (time.perf_counter() - t0) * 1e3
    err_gpu = float(np.max(np.abs(U_gpu - U_ref)))
    speedup = ms_rk4 / max(ms_gpu, 1e-9)

    print(f"  RK4 CPU (dt=1e-4):         MaxElem={err_rk4:.3e}, Time={ms_rk4:.3f} ms")
    print(f"  Magnus ({backend}, dt=1e-3): MaxElem={err_gpu:.3e}, Time={ms_gpu:.3f} ms, Speedup={speedup:.2f}x")

    rows = [
        {"solver": "RK4_CPU_dt1e-4", "max_element_diff": err_rk4, "time_ms": ms_rk4, "speedup": 1.0},
        {"solver": f"MagnusChebyshev_{backend}_dt1e-3", "max_element_diff": err_gpu, "time_ms": ms_gpu, "speedup": speedup},
    ]
    save_csv("exp10_cpu_rk4_vs_gpu_magnus.csv", ["solver", "max_element_diff", "time_ms", "speedup"], rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = ["RK4 CPU\n(dt=1e-4)", f"Magnus {backend}\n(dt=1e-3)"]
    ax1.bar(labels, [ms_rk4, ms_gpu], color=["#d95f02", "#1b9e77"], edgecolor="black")
    ax1.set_title(f"EXP-10: Время расчета (Ускорение {speedup:.2f}x)")
    ax1.set_ylabel("Время (мс)")
    ax1.grid(True, axis="y", ls="--", alpha=0.5)

    ax2.bar(labels, [err_rk4, err_gpu], color=["#d95f02", "#1b9e77"], edgecolor="black")
    ax2.set_yscale("log")
    ax2.set_title("EXP-10: Максимальная ошибка элементов U(T)")
    ax2.set_ylabel("Max Element Diff (log)")
    ax2.grid(True, which="both", axis="y", ls="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUT_DIR / "exp10_cpu_rk4_vs_gpu_magnus.png", dpi=300)
    plt.close(fig)
    return {"status": "done", "backend": backend, "speedup": speedup, "err_gpu": err_gpu}


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск валидационных экспериментов Раздела 1 (EXP-01 .. EXP-10)")
    parser.add_argument("--exp", nargs="+", type=int, default=list(range(1, 11)), help="Номера экспериментов (1..10)")
    parser.add_argument("--quick", action="store_true", help="Быстрый режим с уменьшенным числом шагов эталона RK4")
    args = parser.parse_args()

    ensure_out_dir()
    summary = {}

    dispatch = {
        1: lambda: run_exp01(args.quick),
        2: run_exp02,
        3: lambda: run_exp03(args.quick),
        4: run_exp04,
        5: run_exp05,
        6: lambda: run_exp06(args.quick),
        7: lambda: run_exp07(args.quick),
        8: lambda: run_exp08(args.quick),
        9: run_exp09,
        10: lambda: run_exp10(args.quick),
    }

    for idx in args.exp:
        if idx in dispatch:
            summary[f"EXP-{idx:02d}"] = dispatch[idx]()

    with (OUT_DIR / "summary_section1.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n=== Готово. Таблицы и графики сохранены в {OUT_DIR.resolve()} ===")


if __name__ == "__main__":
    main()