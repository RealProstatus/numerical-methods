from __future__ import annotations
import numpy as np
import qutip
from qutip.core import data as _data
from qutip.solver.integrator import Integrator
from . import _magnus_py


class _IntegratorOptions(dict):
    """
    Integrator options
    ------------------
    magnus_method : str, default: "Formula314_6"
        Order and formula of Magnus expansion:
        'Classic2', 'Classic4', 'Acc4', 'Acc6', 'Formula314_6', 'Recursive'.
    expm_method : str, default: "Chebyshev"
        Matrix exponential algorithm:
        'Chebyshev', 'PadeEigen', 'Taylor'.
    backend : str, default: "Auto"
        Hardware execution backend:
        'Auto', 'Cpu', 'Cuda', 'MpiCpu'.
    chebyshev_degree : int, default: 16
        Polynomial degree for Chebyshev expansion.
    taylor_terms : int, default: 30
        Number of terms for Taylor expansion.
    recursive_order : int, default: 8
        Maximum order for recursive Magnus expansion.
    dt : float, default: 1e-2
        Integration step size for Magnus propagation.
    is_harmonic : bool, default: False
        Fast-path for harmonic driving: f(t) = epsilon * cos(omega * t).
    epsilon : float, default: 1.0
        Modulation amplitude.
    omega : float, default: 1.0
        Angular modulation frequency.
    mpi_comm : object, default: None
        mpi4py MPI.Comm communicator object.
    """
    pass


class MagnusChebyshevIntegrator(Integrator):
    """
    High-Performance Magnus-Chebyshev Integrator (C++20 / Intel MKL / CUDA / MPI)
    for QuTiP 5 SESolver and Propagator.
    """

    options = _IntegratorOptions({
        "magnus_method": "Formula314_6",
        "expm_method": "Chebyshev",
        "backend": "Auto",
        "chebyshev_degree": 16,
        "taylor_terms": 30,
        "recursive_order": 8,
        "dt": 1e-2,
        "is_harmonic": False,
        "epsilon": 1.0,
        "omega": 1.0,
        "mpi_comm": None,
    })
    integrator_options = options
    support_time_dependant = True
    supports_blackbox = True
    method = "magnus_chebyshev"

    def _prepare(self):
        self._N = self.system.shape[0]
        self._opts = _magnus_py.SolverOptions()
        self._opts.magnus_method = getattr(_magnus_py.MagnusMethod, self.options["magnus_method"])
        self._opts.expm_method = getattr(_magnus_py.ExpmMethod, self.options["expm_method"])
        self._opts.backend = getattr(_magnus_py.BackendType, self.options["backend"])
        self._opts.chebyshev_degree = int(self.options["chebyshev_degree"])
        self._opts.taylor_terms = int(self.options["taylor_terms"])
        self._opts.recursive_order = int(self.options["recursive_order"])
        self._opts.hbar = 1.0

        if self.options.get("mpi_comm") is not None and hasattr(self.options["mpi_comm"], "py2f"):
            self._opts.mpi_comm_fortran = int(self.options["mpi_comm"].py2f())

        self._extract_qobjevo_components()
        self._t_current = 0.0
        self._state_dense = None
        self._name = f"magnus_chebyshev ({self.options['magnus_method']})"

    def _extract_qobjevo_components(self):
        self._H0 = np.zeros((self._N, self._N), dtype=np.complex128, order="C")
        self._H1 = np.zeros((self._N, self._N), dtype=np.complex128, order="C")
        self._drive_terms = []

        if hasattr(self.system, "to_list"):
            ops = self.system.to_list()
        else:
            ops = [self.system]

        for item in ops:
            if isinstance(item, qutip.Qobj):
                # В генераторе уравнения Шрёдингера QuTiP: G = -1j * H  =>  H = 1j * G
                self._H0 += np.ascontiguousarray(1j * item.full(), dtype=np.complex128)
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                H_k = np.ascontiguousarray(1j * item[0].full(), dtype=np.complex128)
                coeff_obj = item[1]
                self._drive_terms.append((H_k, coeff_obj))

        if len(self._drive_terms) >= 1:
            self._H1 = self._drive_terms[0][0]

    def _eval_coeff(self, t: float) -> float:
        if not self._drive_terms:
            return 0.0
        _, coeff = self._drive_terms[0]
        if callable(coeff):
            try:
                return float(np.real(coeff(t)))
            except TypeError:
                return float(np.real(coeff(t, {})))
        return float(np.real(coeff))

    def set_state(self, t: float, state0):
        self._t_current = float(t)
        if hasattr(state0, "to_array"):
            arr = state0.to_array()
        elif hasattr(state0, "full"):
            arr = state0.full()
        else:
            arr = np.asarray(state0)
        self._state_dense = np.ascontiguousarray(arr, dtype=np.complex128)

    def get_state(self, copy: bool = True):
        state_arr = self._state_dense.copy() if copy else self._state_dense
        return self._t_current, _data.Dense(state_arr)

    def integrate(self, t: float, copy: bool = False):
        t = float(t)
        if t <= self._t_current:
            return self.get_state(copy=copy)

        dt = float(self.options["dt"])
        steps = max(1, int(np.ceil((t - self._t_current) / dt)))
        time_nodes = np.linspace(self._t_current, t, steps + 1, dtype=np.float64)

        is_harmonic = bool(self.options["is_harmonic"])
        if is_harmonic:
            mod_grid = np.empty(0, dtype=np.float64)
        else:
            micro_nodes = np.linspace(self._t_current, t, 2 * steps + 1, dtype=np.float64)
            mod_grid = np.array([self._eval_coeff(float(tau)) for tau in micro_nodes], dtype=np.float64)

        U_step = np.empty((self._N, self._N), dtype=np.complex128, order="C")
        _magnus_py.propagate_cpu_zero_copy(
            self._H0,
            self._H1,
            time_nodes,
            mod_grid,
            is_harmonic,
            float(self.options["epsilon"]),
            float(self.options["omega"]),
            self._opts,
            U_step,
        )

        self._state_dense = U_step @ self._state_dense
        self._t_current = t
        return self.get_state(copy=copy)

    def reset(self, hard: bool = False):
        self._t_current = 0.0
        self._state_dense = None


if hasattr(qutip, "SESolver") and hasattr(qutip.SESolver, "add_integrator"):
    qutip.SESolver.add_integrator(MagnusChebyshevIntegrator, "magnus_chebyshev")
elif hasattr(qutip.solver, "SESolver") and hasattr(qutip.solver.SESolver, "add_integrator"):
    qutip.solver.SESolver.add_integrator(MagnusChebyshevIntegrator, "magnus_chebyshev")