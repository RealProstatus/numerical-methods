from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union
import numpy as np
from scipy.interpolate import CubicSpline

from . import _magnus_py

# Узлы квадратуры Гаусса-Лежандра 6-го порядка на [0, 1]
_GL3_C = np.array(
    [0.5 - np.sqrt(15.0) / 10.0, 0.5, 0.5 + np.sqrt(15.0) / 10.0],
    dtype=np.float64,
)

ScalarModType = Union[Callable[[Union[float, np.ndarray]], Union[float, np.ndarray]], Sequence[float], np.ndarray]
ChannelType = Tuple[np.ndarray, ScalarModType]


def has_cuda_backend() -> bool:
    return bool(_magnus_py.has_cuda_backend())


def has_mpi_backend() -> bool:
    return bool(_magnus_py.has_mpi_backend())


def _build_options(
    magnus_method: str = "Formula314_6",
    expm_method: str = "Chebyshev",
    backend: str = "Cpu",
    chebyshev_degree: int = 16,
    taylor_terms: int = 30,
    recursive_order: int = 6,
    hbar: float = 1.0,
) -> _magnus_py.SolverOptions:
    opts = _magnus_py.SolverOptions()
    opts.magnus_method = getattr(_magnus_py.MagnusMethod, magnus_method)
    opts.expm_method = getattr(_magnus_py.ExpmMethod, expm_method)
    opts.backend = getattr(_magnus_py.BackendType, backend)
    opts.chebyshev_degree = int(chebyshev_degree)
    opts.taylor_terms = int(taylor_terms)
    opts.recursive_order = int(recursive_order)
    opts.hbar = float(hbar)
    return opts


def evaluate_gl3_scalar(f_mod: ScalarModType, time_nodes: np.ndarray) -> np.ndarray:
    """Вычисляет скалярную функцию f(t) в 3 узлах Гаусса-Лежандра каждого шага [t_n, t_{n+1}]."""
    t0 = time_nodes[:-1, None]
    h = (time_nodes[1:] - time_nodes[:-1])[:, None]
    t_gl = t0 + h * _GL3_C[None, :]  # (steps, 3)

    if callable(f_mod):
        try:
            vals = np.asarray(f_mod(t_gl), dtype=np.float64)
            if vals.shape == t_gl.shape:
                return np.ascontiguousarray(vals)
        except Exception:
            pass
        vals = np.array([[float(f_mod(float(t))) for t in row] for row in t_gl], dtype=np.float64)
        return np.ascontiguousarray(vals)

    arr = np.asarray(f_mod, dtype=np.float64).ravel()
    if arr.size < 2:
        raise ValueError("Tabulated modulation array must contain at least 2 points.")
    t_grid = np.linspace(float(time_nodes[0]), float(time_nodes[-1]), arr.size)
    cs = CubicSpline(t_grid, arr, bc_type="natural")
    return np.ascontiguousarray(cs(t_gl), dtype=np.float64)


def expm_magnus(
    A: np.ndarray,
    method: str = "Chebyshev",
    degree: int = 16,
    terms: int = 30,
) -> np.ndarray:
    A_c = np.ascontiguousarray(A, dtype=np.complex128)
    out = np.empty_like(A_c, order="C")
    enum_m = getattr(_magnus_py.ExpmMethod, method)
    param = terms if method == "Taylor" else degree
    _magnus_py.expm_cpu(A_c, out, enum_m, int(param))
    return out


def solve_magnus(
    H0: Optional[np.ndarray] = None,
    H1: Optional[np.ndarray] = None,
    t_start: float = 0.0,
    t_final: float = 1.0,
    steps: int = 100,
    epsilon: float = 0.0,
    omega: float = 0.0,
    f_mod: Optional[ScalarModType] = None,
    channels: Optional[Sequence[ChannelType]] = None,
    H_func: Optional[Callable[[float], np.ndarray]] = None,
    time_nodes: Optional[np.ndarray] = None,
    magnus_method: str = "Formula314_6",
    expm_method: str = "Chebyshev",
    backend: str = "Cpu",
    chebyshev_degree: int = 16,
    taylor_terms: int = 30,
    recursive_order: int = 6,
    hbar: float = 1.0,
) -> np.ndarray:
    """
    Универсальный решатель уравнения Шрёдингера i * hbar * dU/dt = H(t) U(t):
      1. Гармонический режим: H(t) = H0 + epsilon * cos(omega * t) * H1 (CPU / CUDA / MPI)
      2. Произвольная скалярная модуляция: H(t) = H0 + f_mod(t) * H1 (CPU / CUDA)
      3. Многоканальный режим: H(t) = H0 + sum_k f_k(t) * H_k (через channels=[(H1, f1), ...])
      4. Произвольный матричный оракул: H(t) = H_func(t)
    """
    if time_nodes is None:
        t_nodes = np.linspace(float(t_start), float(t_final), int(steps) + 1, dtype=np.float64)
    else:
        t_nodes = np.ascontiguousarray(time_nodes, dtype=np.float64)
        steps = int(t_nodes.size - 1)

    opts = _build_options(
        magnus_method=magnus_method,
        expm_method=expm_method,
        backend=backend,
        chebyshev_degree=chebyshev_degree,
        taylor_terms=taylor_terms,
        recursive_order=recursive_order,
        hbar=hbar,
    )

    # === УРОВЕНЬ 3: Полностью произвольный матричный оракул H(t) ===
    if H_func is not None:
        t0_step = t_nodes[:-1, None]
        h_step = (t_nodes[1:] - t_nodes[:-1])[:, None]
        t_gl = t0_step + h_step * _GL3_C[None, :]  # (steps, 3)

        H_sample = np.asarray(H_func(float(t_gl[0, 0])), dtype=np.complex128)
        N = H_sample.shape[0]
        A_gl = np.empty((steps, 3, N, N), dtype=np.complex128, order="C")
        factor = -1j / float(hbar)
        for s in range(steps):
            for q in range(3):
                A_gl[s, q] = factor * np.asarray(H_func(float(t_gl[s, q])), dtype=np.complex128)

        U_out = np.empty((N, N), dtype=np.complex128, order="C")
        _magnus_py.propagate_general_gl_cpu(t_nodes, A_gl, opts, U_out)
        return U_out

    if H0 is None:
        raise ValueError("H0 must be provided unless H_func is specified.")

    # Поддержка GPU-массивов (CuPy) через DLPack без выхода на CPU
    if hasattr(H0, "__cuda_array_interface__"):
        import cupy as cp
        d_H0 = cp.ascontiguousarray(H0, dtype=cp.complex128)
        d_H1 = cp.zeros_like(d_H0) if H1 is None else cp.ascontiguousarray(H1, dtype=cp.complex128)
        d_U = cp.empty_like(d_H0)
        _magnus_py.propagate_cuda_dlpack(
            d_H0, d_H1, float(t_nodes[0]), float(t_nodes[-1]), steps, float(epsilon), float(omega), opts, d_U
        )
        return d_U

    H0_c = np.ascontiguousarray(H0, dtype=np.complex128)
    N = H0_c.shape[0]
    U_out = np.empty((N, N), dtype=np.complex128, order="C")

    # === УРОВЕНЬ 2: Многоканальный гамильтониан H0 + sum_k f_k(t) H_k ===
    if channels is not None:
        factor = -1j / float(hbar)
        A_gl = np.broadcast_to((factor * H0_c)[None, None, :, :], (steps, 3, N, N)).copy()
        for H_k, f_k in channels:
            H_k_c = np.ascontiguousarray(H_k, dtype=np.complex128)
            f_vals = evaluate_gl3_scalar(f_k, t_nodes)  # (steps, 3)
            A_gl += (factor * f_vals[:, :, None, None]) * H_k_c[None, None, :, :]

        A_gl = np.ascontiguousarray(A_gl, dtype=np.complex128)
        _magnus_py.propagate_general_gl_cpu(t_nodes, A_gl, opts, U_out)
        return U_out

    H1_c = np.zeros_like(H0_c) if H1 is None else np.ascontiguousarray(H1, dtype=np.complex128)

    # === УРОВЕНЬ 1Б: Одноканальная произвольная модуляция f_mod(t) (CPU & CUDA) ===
    if f_mod is not None:
        gl_vals = evaluate_gl3_scalar(f_mod, t_nodes).ravel()
        if backend == "Cuda":
            _magnus_py.propagate_cuda_gl3(H0_c, H1_c, t_nodes, gl_vals, opts, U_out)
        else:
            _magnus_py.propagate_cpu_zero_copy(H0_c, H1_c, t_nodes, gl_vals, False, 0.0, 0.0, opts, U_out)
        return U_out

    # === УРОВЕНЬ 1А: Быстрый гармонический путь H0 + epsilon * cos(omega * t) * H1 ===
    # Выбор Cpu / Cuda / Auto выполняется внутри propagate_cpu_zero_copy по флагу opts.backend
    empty_mod = np.empty(0, dtype=np.float64)
    _magnus_py.propagate_cpu_zero_copy(
        H0_c, H1_c, t_nodes, empty_mod, True, float(epsilon), float(omega), opts, U_out
    )
    return U_out