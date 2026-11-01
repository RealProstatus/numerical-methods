from __future__ import annotations
from typing import Callable, Optional, Union
import numpy as np
from . import _magnus_py

def expm_magnus(
    A: np.ndarray,
    method: str = "Chebyshev",
    degree: int = 16,
) -> np.ndarray:
    """Вычисляет матричную экспоненту exp(A) через C++/MKL ядро проекта."""
    A_c = np.ascontiguousarray(A, dtype=np.complex128)
    out = np.empty_like(A_c, order="C")
    expm_enum = getattr(_magnus_py.ExpmMethod, method)
    _magnus_py.expm_cpu(A_c, out, expm_enum, int(degree))
    return out

def solve_magnus(
    H0: np.ndarray,
    H1: np.ndarray,
    t_start: float,
    t_final: float,
    steps: int,
    *,
    f_mod: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    epsilon: float = 1.0,
    omega: float = 1.0,
    magnus_method: str = "Formula314_6",
    expm_method: str = "Chebyshev",
    backend: str = "Auto",
    chebyshev_degree: int = 16,
    hbar: float = 1.0,
    mpi_comm=None,
) -> np.ndarray:
    """
    Вычисляет унитарный пропагатор U(t_final, t_start) для H(t) = H0 + f(t)*H1.
    Если f_mod не задан, используется гармонический закон f(t) = epsilon * cos(omega * t).
    """
    opts = _magnus_py.SolverOptions()
    opts.magnus_method = getattr(_magnus_py.MagnusMethod, magnus_method)
    opts.expm_method = getattr(_magnus_py.ExpmMethod, expm_method)
    opts.backend = getattr(_magnus_py.BackendType, backend)
    opts.chebyshev_degree = int(chebyshev_degree)
    opts.hbar = float(hbar)

    if mpi_comm is not None and hasattr(mpi_comm, "py2f"):
        opts.mpi_comm_fortran = int(mpi_comm.py2f())

    # Проверяем, переданы ли GPU-массивы (CuPy)
    if hasattr(H0, "__dlpack__") and getattr(H0, "device", None) is not None and str(H0.device) != "cpu":
        import cupy as cp
        d_H0 = cp.ascontiguousarray(H0, dtype=cp.complex128)
        d_H1 = cp.ascontiguousarray(H1, dtype=cp.complex128)
        d_U = cp.empty_like(d_H0, order="C")
        _magnus_py.propagate_cuda_dlpack(
            d_H0, d_H1, float(t_start), float(t_final), int(steps),
            float(epsilon), float(omega), opts, d_U
        )
        return d_U

    H0_c = np.ascontiguousarray(H0, dtype=np.complex128)
    H1_c = np.ascontiguousarray(H1, dtype=np.complex128)
    U_out = np.empty_like(H0_c, order="C")

    time_nodes = np.linspace(t_start, t_final, steps + 1, dtype=np.float64)
    is_harmonic = f_mod is None

    if is_harmonic:
        mod_grid = np.empty(0, dtype=np.float64)
    else:
        # Узлы Симпсона (t_n, t_n + dt/2, t_{n+1}) для каждого шага
        micro_nodes = np.linspace(t_start, t_final, 2 * steps + 1, dtype=np.float64)
        mod_vals = np.asarray(f_mod(micro_nodes), dtype=np.float64)
        if mod_vals.ndim == 0 or mod_vals.shape[0] != micro_nodes.shape[0]:
            mod_vals = np.array([float(np.real(f_mod(float(t)))) for t in micro_nodes], dtype=np.float64)
        mod_grid = np.ascontiguousarray(mod_vals, dtype=np.float64)

    _magnus_py.propagate_cpu_zero_copy(
        H0_c, H1_c, time_nodes, mod_grid,
        is_harmonic, float(epsilon), float(omega), opts, U_out
    )
    return U_out