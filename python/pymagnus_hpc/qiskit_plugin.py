from __future__ import annotations
import numpy as np
from .solver import solve_magnus

def solve_lmde_magnus_hpc(
    generator,
    t_span,
    y0,
    max_dt=1e-2,
    t_eval=None,
    magnus_method="Formula314_6",
    expm_method="Chebyshev",
    backend="Auto",
    chebyshev_degree=16,
    **kwargs,
):
    """
    Внешний C++/MKL/CUDA солвер для qiskit_dynamics.solvers.solve_lmde(..., method=solve_lmde_magnus_hpc).
    Принимает генератор G(t) = -i * (H0 + s(t)*H1).
    """
    t0, tf = float(t_span[0]), float(t_span[1])
    H0 = np.ascontiguousarray(1j * np.asarray(generator.drift), dtype=np.complex128)
    if getattr(generator, "operators", None) is not None and len(generator.operators) > 0:
        H1 = np.ascontiguousarray(1j * np.asarray(generator.operators[0]), dtype=np.complex128)
        sig = generator.signals[0]
        f_mod = lambda t_arr: np.real(sig(t_arr))
    else:
        H1 = np.zeros_like(H0)
        f_mod = lambda t_arr: np.zeros_like(t_arr)

    time_points = np.asarray(t_eval if t_eval is not None else [t0, tf], dtype=np.float64)
    y_curr = np.ascontiguousarray(y0, dtype=np.complex128)
    results = [y_curr.copy()]
    t_curr = t0

    for t_target in time_points[1:]:
        if t_target > t_curr:
            steps = max(1, int(np.ceil((t_target - t_curr) / max_dt)))
            U_step = solve_magnus(
                H0, H1, t_curr, float(t_target), steps,
                f_mod=f_mod,
                magnus_method=magnus_method,
                expm_method=expm_method,
                backend=backend,
                chebyshev_degree=chebyshev_degree,
            )
            y_curr = U_step @ y_curr
            t_curr = float(t_target)
        results.append(y_curr.copy())

    return np.stack(results, axis=0)