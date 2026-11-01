from __future__ import annotations
import numpy as np
from scipy.integrate import OdeSolver
from scipy.integrate._ivp.common import OdeResult
from .solver import solve_magnus

class MagnusOdeSolver(OdeSolver):
    """
    Кастомный подкласс scipy.integrate.OdeSolver для запуска C++/MKL/CUDA солвера Магнуса
    через стандартный вызов scipy.integrate.solve_ivp(..., method=MagnusOdeSolver).
    Передача матриц H0, H1 выполняется через kwargs solve_ivp.
    """

    def __init__(
        self,
        fun,
        t0,
        y0,
        t_bound,
        vectorized=False,
        support_complex=True,
        H0=None,
        H1=None,
        f_mod=None,
        epsilon=1.0,
        omega=1.0,
        dt=1e-2,
        magnus_method="Formula314_6",
        expm_method="Chebyshev",
        backend="Auto",
        chebyshev_degree=16,
        **extraneous,
    ):
        super().__init__(fun, t0, y0, t_bound, vectorized, support_complex=True)
        if H0 is None or H1 is None:
            raise ValueError("MagnusOdeSolver requires H0 and H1 matrices passed to solve_ivp.")
        self.H0 = np.ascontiguousarray(H0, dtype=np.complex128)
        self.H1 = np.ascontiguousarray(H1, dtype=np.complex128)
        self.N = self.H0.shape[0]
        self.f_mod = f_mod
        self.epsilon = epsilon
        self.omega = omega
        self.dt = float(dt)
        self.magnus_method = magnus_method
        self.expm_method = expm_method
        self.backend = backend
        self.chebyshev_degree = chebyshev_degree

    def _step_impl(self):
        t_next = min(self.t + self.dt, self.t_bound) if self.direction > 0 else max(self.t - self.dt, self.t_bound)
        U_step = solve_magnus(
            self.H0,
            self.H1,
            self.t,
            t_next,
            steps=1,
            f_mod=self.f_mod,
            epsilon=self.epsilon,
            omega=self.omega,
            magnus_method=self.magnus_method,
            expm_method=self.expm_method,
            backend=self.backend,
            chebyshev_degree=self.chebyshev_degree,
        )
        if self.y.size == self.N * self.N:
            Y_mat = self.y.reshape((self.N, self.N))
            self.y = (U_step @ Y_mat).ravel()
        else:
            self.y = U_step @ self.y
        self.t = t_next
        return True, None

    def _dense_output_impl(self):
        return None