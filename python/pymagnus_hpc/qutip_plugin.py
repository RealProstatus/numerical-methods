from __future__ import annotations

import numpy as np
import qutip as qt
from qutip.solver.integrator import Integrator
from qutip.solver.sesolve import SESolver

from .solver import solve_magnus


class MagnusChebyshevIntegrator(Integrator):
    """Интегратор Магнуса-Чебышева для QuTiP 5 с поддержкой произвольных QobjEvo."""

    supports_blackbox = True

    integrator_options = {
        "dt": None,
        "nsteps": 10,
        "magnus_method": "Formula314_6",
        "expm_method": "Chebyshev",
        "backend": "Cpu",
        "chebyshev_degree": 16,
    }

    def __init__(self, system: qt.QobjEvo, options: dict):
        self.H = system
        self._t = 0.0
        self._y = None
        self._H0 = None
        self._channels = []
        self._use_oracle = False
        super().__init__(system, options)

    def _prepare(self) -> None:
        """Разбирает правую часть SESolver (rhs = -i * H) на каналы H0 + sum_k f_k(t) H_k."""
        self._H0 = None
        self._channels = []
        self._use_oracle = False

        try:
            elements = self.system.to_list()
            H0_accum = None
            for item in elements:
                if isinstance(item, qt.Qobj):
                    # SESolver передаёт rhs = -i * H => H = i * rhs
                    mat = np.ascontiguousarray(1j * item.full(), dtype=np.complex128)
                    H0_accum = mat if H0_accum is None else (H0_accum + mat)
                elif isinstance(item, (list, tuple)) and len(item) == 2 and isinstance(item[0], qt.Qobj):
                    raw_op = np.ascontiguousarray(item[0].full(), dtype=np.complex128)
                    coeff_k = item[1]
                    h_cand = 1j * raw_op
                    if np.allclose(h_cand, h_cand.conj().T, atol=1e-12):
                        mat_k = np.ascontiguousarray(h_cand, dtype=np.complex128)
                        fn_k = (
                            (lambda t, c=coeff_k: float(np.real(c(t))))
                            if callable(coeff_k)
                            else (lambda t, c=coeff_k: float(np.real(c(t))))
                        )
                    elif np.allclose(raw_op, raw_op.conj().T, atol=1e-12):
                        mat_k = raw_op
                        fn_k = lambda t, c=coeff_k: float(np.real(1j * c(t)))
                    else:
                        self._use_oracle = True
                        break
                    self._channels.append((mat_k, fn_k))
                else:
                    self._use_oracle = True
                    break

            if H0_accum is None and not self._use_oracle and self._channels:
                N = self._channels[0][0].shape[0]
                H0_accum = np.zeros((N, N), dtype=np.complex128)
            self._H0 = H0_accum
        except Exception:
            self._use_oracle = True

    @property
    def options(self):
        """
        Supported options for MagnusChebyshevIntegrator:

        dt : float or None, default=None
            Substep size for Magnus integration. If None, `nsteps` is used.
        nsteps : int, default=10
            Number of substeps per integration interval when `dt` is None.
        magnus_method : str, default='Formula314_6'
            Magnus expansion scheme ('Classic2', 'Classic4', 'Acc4', 'Acc6', 'Formula314_6', 'Recursive').
        expm_method : str, default='Chebyshev'
            Matrix exponential algorithm ('Chebyshev', 'PadeEigen', 'Taylor').
        backend : str, default='Cpu'
            Execution backend ('Cpu', 'Cuda', 'Auto').
        chebyshev_degree : int, default=16
            Degree of Chebyshev polynomial approximation.
        """
        return self._options

    @options.setter
    def options(self, new_options):
        Integrator.options.fset(self, new_options)

    def set_state(self, t: float, state0: qt.Data) -> None:
        self._t = float(t)
        self._y = state0
        self._is_set = True

    def get_state(self, copy: bool = False):
        return self._t, self._y

    def _compute_steps(self, t_target: float) -> int:
        span = abs(t_target - self._t)
        dt = self._options.get("dt", None)
        if dt is not None and float(dt) > 0.0:
            return max(1, int(np.ceil(span / float(dt))))
        return max(1, int(self._options.get("nsteps", 10)))

    def integrate(self, t: float, copy: bool = False):
        t_target = float(t)
        if np.isclose(t_target, self._t):
            return self._t, self._y

        steps = self._compute_steps(t_target)
        magnus_method = str(self._options.get("magnus_method", "Formula314_6"))
        expm_method = str(self._options.get("expm_method", "Chebyshev"))
        backend = str(self._options.get("backend", "Cpu"))
        chebyshev_degree = int(self._options.get("chebyshev_degree", 16))

        if self._use_oracle:
            U_step = solve_magnus(
                t_start=self._t,
                t_final=t_target,
                steps=steps,
                H_func=lambda tau: 1j * self.system(tau).full(),
                magnus_method=magnus_method,
                expm_method=expm_method,
                backend=backend,
                chebyshev_degree=chebyshev_degree,
            )
        elif len(self._channels) == 0:
            U_step = solve_magnus(
                H0=self._H0,
                t_start=self._t,
                t_final=t_target,
                steps=steps,
                epsilon=0.0,
                omega=0.0,
                magnus_method=magnus_method,
                expm_method=expm_method,
                backend=backend,
                chebyshev_degree=chebyshev_degree,
            )
        elif len(self._channels) == 1:
            H1_mat, f1_fn = self._channels[0]
            U_step = solve_magnus(
                H0=self._H0,
                H1=H1_mat,
                t_start=self._t,
                t_final=t_target,
                steps=steps,
                f_mod=f1_fn,
                magnus_method=magnus_method,
                expm_method=expm_method,
                backend=backend,
                chebyshev_degree=chebyshev_degree,
            )
        else:
            U_step = solve_magnus(
                H0=self._H0,
                t_start=self._t,
                t_final=t_target,
                steps=steps,
                channels=self._channels,
                magnus_method=magnus_method,
                expm_method=expm_method,
                backend=backend,
                chebyshev_degree=chebyshev_degree,
            )

        psi_vec = self._y.to_array()
        psi_next = U_step @ psi_vec
        self._y = qt.core.data.create(psi_next)
        self._t = t_target
        return self._t, self._y


SESolver.add_integrator(MagnusChebyshevIntegrator, "magnus_chebyshev")