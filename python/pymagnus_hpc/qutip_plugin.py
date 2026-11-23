from __future__ import annotations

import math
import numpy as np
import qutip as qt
from qutip.solver.integrator import Integrator, IntegratorException
from qutip.solver.sesolve import SESolver

from . import _magnus_py
from .solver import _build_options, expm_magnus, solve_magnus

_SQRT15 = math.sqrt(15.0)
_GL3_C1 = 0.5 - _SQRT15 / 10.0
_GL3_C2 = 0.5
_GL3_C3 = 0.5 + _SQRT15 / 10.0
_GL3_C_ARR = np.array([_GL3_C1, _GL3_C2, _GL3_C3], dtype=np.float64)


def _detect_harmonic_cos(fn) -> tuple[float, float] | None:
    """
    Проверяет, совпадает ли fn(t) с eps * cos(omega * t) (omega > 0).
    При малейшем отклонении возвращает None (безопасный откат на общий GL3-путь).
    """
    try:
        eps = float(fn(0.0))
    except Exception:
        return None
    if not math.isfinite(eps) or abs(eps) < 1e-14:
        return None

    omega = None
    for dt in (1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        try:
            r = float(fn(dt)) / eps
        except Exception:
            return None
        if not math.isfinite(r):
            return None
        if 0.0 < r < 0.999:
            omega = math.acos(r) / dt
            break

    if omega is None or not math.isfinite(omega) or omega <= 0.0:
        return None

    # Уточняем omega в точке максимальной чувствительности (omega * dt ~ 1 рад)
    dt_opt = 1.0 / omega
    try:
        r_opt = float(fn(dt_opt)) / eps
    except Exception:
        return None
    if not (-1.0 <= r_opt <= 1.0):
        return None
    omega = math.acos(r_opt) / dt_opt
    if not math.isfinite(omega) or omega <= 0.0:
        return None

    tol = 1e-11 * max(1.0, abs(eps))
    for scale in (-2.37, -0.71, -0.19, 0.37, 1.23, 2.71, 4.13, 7.89, 13.41):
        t_test = scale / omega
        try:
            val = float(fn(t_test))
        except Exception:
            return None
        if not math.isfinite(val) or abs(val - eps * math.cos(omega * t_test)) > tol:
            return None

    return eps, omega

_METHOD_ORDER = {
    "Classic2": 2,
    "Classic4": 4,
    "Acc4": 4,
    "Formula314_4": 4,
    "Acc6": 6,
    "Formula314_6": 6,
    "Recursive": 6,
}


def _op_norm1(mat: np.ndarray) -> float:
    """Операторная 1-норма (для антиэрмитовых матриц строго ограничивает спектральную 2-норму)."""
    return float(np.linalg.norm(mat, 1))


class MagnusChebyshevIntegrator(Integrator):
    """Адаптивный интегратор Магнуса-Чебышева для QuTiP 5 с поддержкой atol/rtol."""

    supports_blackbox = True

    integrator_options = {
        "atol": 1e-8,
        "rtol": 1e-6,
        "nsteps": 2500,          # Лимит внутренних шагов на интервал tlist (в режиме adaptive=False — фиксированное число)
        "first_step": 0.0,       # 0.0 -> автоматический начальный шаг
        "min_step": 0.0,
        "max_step": 0.0,         # 0.0 -> без ограничения сверху
        "dt": None,              # Если задан (> 0), включает режим фиксированного шага
        "adaptive": True,
        "magnus_method": "Formula314_6",
        "expm_method": "Chebyshev",
        "backend": "Cpu",
        "chebyshev_degree": 16,
        "use_harmonic_core": True
    }

    def __init__(self, system: qt.QobjEvo, options: dict):
        self.H = system
        self._t = 0.0
        self._y = None
        self._h_curr = 0.0
        self._H0 = None
        self._channels = []
        self._use_oracle = False

        self._norms_ready = False
        self._norms_order = 0
        self._norm_scale = 1.0
        self._sc_norms: dict[str, float] = {}
        self._mc_norm_A0 = 0.0
        self._mc_norm_Ak: list[float] = []

        # Счётчики шагов для профилирования и work-precision диаграмм
        self.total_steps = 0
        self.rejected_steps = 0

        self._harmonic_params: tuple[float, float] | None = None

        super().__init__(system, options)

    def _prepare(self) -> None:
        """Разбирает правую часть SESolver (rhs = -i * H) на каналы H0 + sum_k f_k(t) H_k."""
        self._H0 = None
        self._channels = []
        self._use_oracle = False
        self._norms_ready = False
        self._norms_order = 0
        self._sc_norms.clear()
        self._mc_norm_Ak.clear()
        self._harmonic_params = None

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
                        fn_k = lambda t, c=coeff_k: float(np.real(c(t)))
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
            
            if not self._use_oracle and len(self._channels) == 1:
                self._harmonic_params = _detect_harmonic_cos(self._channels[0][1])
        except Exception:
            self._use_oracle = True

    def _ensure_error_norms(self, order: int) -> None:
        """Один раз предвычисляет операторные 1-нормы коммутаторов/каналов для скалярной оценки ошибки."""
        if self._norms_ready and self._norms_order >= order:
            return
        if self._use_oracle or len(self._channels) == 0:
            self._norms_ready = True
            self._norms_order = order
            return

        if len(self._channels) == 1:
            A0 = -1j * self._H0
            A1 = -1j * self._channels[0][0]
            norm_A0 = _op_norm1(A0)
            norm_A1 = _op_norm1(A1)
            self._norm_scale = max(norm_A0 + norm_A1, 1e-3)

            C1 = A0 @ A1 - A1 @ A0
            self._sc_norms["A1"] = norm_A1
            self._sc_norms["C1"] = _op_norm1(C1)

            if order >= 4:
                C2 = A0 @ C1 - C1 @ A0
                C3 = A1 @ C1 - C1 @ A1
                C4 = A0 @ C2 - C2 @ A0
                # По тождеству Якоби C6 = [A0, C3] == [A1, C2] = C5
                C5 = A1 @ C2 - C2 @ A1
                C7 = A1 @ C3 - C3 @ A1

                self._sc_norms["C2"] = _op_norm1(C2)
                self._sc_norms["C3"] = _op_norm1(C3)
                self._sc_norms["C4"] = _op_norm1(C4)
                self._sc_norms["C5"] = _op_norm1(C5)
                self._sc_norms["C7"] = _op_norm1(C7)
        else:
            self._mc_norm_A0 = _op_norm1(self._H0)
            self._mc_norm_Ak = [_op_norm1(Hk) for Hk, _ in self._channels]
            self._norm_scale = max(self._mc_norm_A0 + sum(self._mc_norm_Ak), 1e-3)

        self._norms_ready = True
        self._norms_order = order

    @property
    def options(self):
        """
        Supported options for MagnusChebyshevIntegrator:

        atol : float, default=1e-8
            Absolute local error tolerance for adaptive step-size control.
        rtol : float, default=1e-6
            Relative local error tolerance for adaptive step-size control.
        nsteps : int, default=2500
            Maximum internal steps per interval in adaptive mode, or fixed step
            count when `adaptive=False` and `dt=None`.
        first_step : float, default=0.0
            Initial step size (0.0 for automatic selection).
        min_step : float, default=0.0
            Minimum allowed adaptive step size.
        max_step : float, default=0.0
            Maximum allowed adaptive step size (0.0 for no upper cap).
        dt : float or None, default=None
            Fixed substep size. If set (> 0), disables adaptive stepping.
        adaptive : bool, default=True
            Enable zero-overhead a priori adaptive grid selection via atol/rtol.
        magnus_method : str, default='Formula314_6'
            Magnus expansion scheme ('Classic2', 'Classic4', 'Acc4', 'Formula314_4',
            'Acc6', 'Formula314_6', 'Recursive').
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
        self._h_curr = float(self._options.get("first_step", 0.0))
        self.total_steps = 0
        self.rejected_steps = 0
        self._is_set = True

    def get_state(self, copy: bool = False):
        return self._t, self._y

    def _estimate_single_channel_err(
        self, h: float, f_left: float, fv1: float, fv2: float, fv3: float, f_right: float, order: int
    ) -> float:
        d1 = (_SQRT15 * h / 3.0) * (fv3 - fv1)
        d2 = (10.0 * h / 3.0) * (fv3 - 2.0 * fv2 + fv1)
        n = self._sc_norms

        if order == 2:
            # Главный член локальной ошибки 2-го порядка: Omega^[4] - Omega^[2] = O(h^3)
            return abs((1.0 / 12.0) * d2) * n["A1"] + abs((1.0 / 12.0) * h * d1) * n["C1"]

        # Вложенная пара 6(4): ||Omega^[6] - Omega^[4]||_1 + |I_GL3 - I_Simp| * ||A1||_1 = O(h^5)
        h2 = h * h
        h3 = h2 * h
        c2 = abs((1.0 / 360.0) * h2 * d2)
        c3 = abs(-(1.0 / 240.0) * h * (d1 * d1) + (1.0 / 360.0) * h2 * fv2 * d2)
        c4 = abs((1.0 / 720.0) * h3 * d1)
        c56 = abs((1.0 / 360.0) * h3 * fv2 * d1)
        c7 = abs((1.0 / 720.0) * h3 * (fv2 * fv2) * d1)

        i_gl3 = h * ((5.0 / 18.0) * fv1 + (4.0 / 9.0) * fv2 + (5.0 / 18.0) * fv3)
        i_simp = (h / 6.0) * (f_left + 4.0 * fv2 + f_right)
        quad_err = abs(i_gl3 - i_simp) * n["A1"]

        return (
            c2 * n["C2"]
            + c3 * n["C3"]
            + c4 * n["C4"]
            + c56 * n["C5"]
            + c7 * n["C7"]
            + quad_err
        )

    def _estimate_multichannel_err(
        self,
        h: float,
        f_lefts: list[float],
        f_gl: list[tuple[float, float, float]],
        f_rights: list[float],
        order: int,
    ) -> float:
        h_abs = abs(h)
        a1 = h_abs * self._mc_norm_A0
        a2 = 0.0
        a3 = 0.0
        quad_err = 0.0

        for k, norm_k in enumerate(self._mc_norm_Ak):
            fv1, fv2, fv3 = f_gl[k]
            a1 += h_abs * abs(fv2) * norm_k
            a2 += (_SQRT15 * h_abs / 3.0) * abs(fv3 - fv1) * norm_k
            a3 += (10.0 * h_abs / 3.0) * abs(fv3 - 2.0 * fv2 + fv1) * norm_k
            if order >= 4:
                i_gl3 = h * ((5.0 / 18.0) * fv1 + (4.0 / 9.0) * fv2 + (5.0 / 18.0) * fv3)
                i_simp = (h / 6.0) * (f_lefts[k] + 4.0 * fv2 + f_rights[k])
                quad_err += abs(i_gl3 - i_simp) * norm_k

        if order == 2:
            # ||alpha3/12|| + ||[alpha1, alpha2]/12|| <= a3/12 + a1*a2/6
            return (1.0 / 12.0) * a3 + (1.0 / 6.0) * a1 * a2

        # Оценка ||C2/240|| + ||C3/720|| через субмультипликативность ||[X, Y]||_1 <= 2 ||X||_1 ||Y||_1
        c1_bound = 2.0 * a1 * a2
        c2_bound = 2.0 * a2 * (a3 + c1_bound)
        c3_bound = 4.0 * (a1 * a1) * (2.0 * a3 + c1_bound)
        return (1.0 / 240.0) * c2_bound + (1.0 / 720.0) * c3_bound + quad_err

    def _build_adaptive_grid(self, t_start: float, t_target: float, order: int):
        """
        Строит неравномерную сетку time_nodes на [t_start, t_target] за O(1) скалярных операций
        на шаг (без матричных умножений и expm) и возвращает готовые сэмплы в узлах GL3.
        """
        atol = float(self._options.get("atol", 1e-8))
        rtol = float(self._options.get("rtol", 1e-6))
        tol = max(atol + rtol, 1e-15)
        max_nsteps = max(1, int(self._options.get("nsteps", 2500)))
        min_step = float(self._options.get("min_step", 0.0))
        max_step = float(self._options.get("max_step", 0.0))

        err_exp = (1.0 / 3.0) if order == 2 else (1.0 / 5.0)
        direction = 1.0 if t_target >= t_start else -1.0
        span = abs(t_target - t_start)

        if self._h_curr > 0.0:
            h = self._h_curr
        else:
            h = min(span, 0.5 / max(self._norm_scale, 1e-3))

        if max_step > 0.0:
            h = min(h, max_step)
        if min_step > 0.0:
            h = max(h, min_step)

        t_nodes = [t_start]
        t_curr = t_start
        steps_taken = 0
        just_rejected = False

        mode = "oracle" if self._use_oracle else ("single" if len(self._channels) == 1 else "multi")

        if mode == "single":
            f1 = self._channels[0][1]
            f_left = f1(t_curr)
            gl_rows: list[tuple[float, float, float]] = []
        elif mode == "multi":
            f_lefts = [fk(t_curr) for _, fk in self._channels]
            mc_rows: list[list[tuple[float, float, float]]] = [[] for _ in self._channels]
        else:
            A_left = np.ascontiguousarray(self.system(t_curr).full(), dtype=np.complex128)
            if self._h_curr <= 0.0:
                self._norm_scale = max(_op_norm1(A_left), 1e-3)
                h = min(span, 0.5 / self._norm_scale)
            oracle_rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        eps_t = 1e-14 * max(1.0, abs(t_target))
        while direction * (t_target - t_curr) > eps_t:
            if steps_taken >= max_nsteps:
                raise IntegratorException(
                    f"MagnusChebyshevIntegrator: exceeded nsteps={max_nsteps} "
                    f"at t={t_curr:.6g} (target={t_target:.6g})."
                )

            rem = abs(t_target - t_curr)
            h_try = min(h, rem)
            if max_step > 0.0:
                h_try = min(h_try, max_step)
            h_signed = direction * h_try
            t_next = t_target if abs(h_try - rem) <= eps_t else (t_curr + h_signed)

            t1 = t_curr + h_signed * _GL3_C1
            t2 = t_curr + h_signed * _GL3_C2
            t3 = t_curr + h_signed * _GL3_C3

            if mode == "single":
                fv1, fv2, fv3 = f1(t1), f1(t2), f1(t3)
                f_right = f1(t_next)
                err_abs = self._estimate_single_channel_err(h_signed, f_left, fv1, fv2, fv3, f_right, order)
            elif mode == "multi":
                f_gl = [(fk(t1), fk(t2), fk(t3)) for _, fk in self._channels]
                f_rights = [fk(t_next) for _, fk in self._channels]
                err_abs = self._estimate_multichannel_err(h_signed, f_lefts, f_gl, f_rights, order)
            else:
                A1 = np.ascontiguousarray(self.system(t1).full(), dtype=np.complex128)
                A2 = np.ascontiguousarray(self.system(t2).full(), dtype=np.complex128)
                A3 = np.ascontiguousarray(self.system(t3).full(), dtype=np.complex128)
                A_right = np.ascontiguousarray(self.system(t_next).full(), dtype=np.complex128)
                a1 = h_try * _op_norm1(A2)
                a2 = (_SQRT15 * h_try / 3.0) * _op_norm1(A3 - A1)
                a3 = (10.0 * h_try / 3.0) * _op_norm1(A3 - 2.0 * A2 + A1)
                if order == 2:
                    err_abs = (1.0 / 12.0) * a3 + (1.0 / 6.0) * a1 * a2
                else:
                    c1_b = 2.0 * a1 * a2
                    c2_b = 2.0 * a2 * (a3 + c1_b)
                    c3_b = 4.0 * (a1 * a1) * (2.0 * a3 + c1_b)
                    i_diff = (5.0 / 18.0) * A1 + (4.0 / 9.0) * A2 + (5.0 / 18.0) * A3 - (
                        (1.0 / 6.0) * A_left + (2.0 / 3.0) * A2 + (1.0 / 6.0) * A_right
                    )
                    err_abs = (1.0 / 240.0) * c2_b + (1.0 / 720.0) * c3_b + h_try * _op_norm1(i_diff)

            err_ratio = err_abs / tol

            if err_ratio <= 1.0 or (min_step > 0.0 and h_try <= min_step * (1.0 + 1e-12)):
                t_curr = t_next
                t_nodes.append(t_curr)
                steps_taken += 1
                self.total_steps += 1

                if mode == "single":
                    gl_rows.append((fv1, fv2, fv3))
                    f_left = f_right
                elif mode == "multi":
                    for k in range(len(self._channels)):
                        mc_rows[k].append(f_gl[k])
                    f_lefts = f_rights
                else:
                    oracle_rows.append((A1, A2, A3))
                    A_left = A_right

                if err_ratio == 0.0:
                    factor = 1.0 if just_rejected else 5.0
                else:
                    fac_max = 1.0 if just_rejected else 5.0
                    factor = min(fac_max, max(0.2, 0.9 * (err_ratio ** (-err_exp))))

                # Не занижаем шаг для следующего интервала, если упёрлись в остаток rem
                if h_try >= h * (1.0 - 1e-12) or factor < 1.0:
                    h = h_try * factor
                    if max_step > 0.0:
                        h = min(h, max_step)
                    if min_step > 0.0:
                        h = max(h, min_step)
                just_rejected = False
            else:
                self.rejected_steps += 1
                factor = max(0.2, min(0.9, 0.9 * (err_ratio ** (-err_exp))))
                h = h_try * factor
                if min_step > 0.0 and h < min_step:
                    h = min_step
                if h <= 1e-15 * max(1.0, abs(t_curr)):
                    raise IntegratorException(
                        f"MagnusChebyshevIntegrator: step size underflow at t={t_curr:.6g}."
                    )
                just_rejected = True

        self._h_curr = h
        t_arr = np.asarray(t_nodes, dtype=np.float64)
        if mode == "single":
            return t_arr, np.asarray(gl_rows, dtype=np.float64)
        if mode == "multi":
            return t_arr, [np.asarray(rows, dtype=np.float64) for rows in mc_rows]
        return t_arr, np.asarray(oracle_rows, dtype=np.complex128)

    def _compute_fixed_steps(self, t_target: float) -> int:
        span = abs(t_target - self._t)
        dt = self._options.get("dt", None)
        if dt is not None and float(dt) > 0.0:
            ratio = span / float(dt)
            return max(1, int(np.ceil(ratio - 1e-12 * max(1.0, ratio))))
        return max(1, int(self._options.get("nsteps", 10)))

    def integrate(self, t: float, copy: bool = False):
        t_target = float(t)
        if np.isclose(t_target, self._t):
            return self._t, self._y

        magnus_method = str(self._options.get("magnus_method", "Formula314_6"))
        expm_method = str(self._options.get("expm_method", "Chebyshev"))
        backend = str(self._options.get("backend", "Cpu"))
        chebyshev_degree = int(self._options.get("chebyshev_degree", 16))

        dt_fixed = self._options.get("dt", None)
        use_adaptive = bool(self._options.get("adaptive", True)) and not (
            dt_fixed is not None and float(dt_fixed) > 0.0
        )

        # Стационарный гамильтониан всегда точен за 1 шаг
# Стационарный гамильтониан всегда точен за 1 матричную экспоненту exp(-i * dt * H0)
        if not self._use_oracle and len(self._channels) == 0:
            if backend == "Cuda":
                U_step = solve_magnus(
                    H0=self._H0,
                    t_start=self._t,
                    t_final=t_target,
                    steps=1,
                    epsilon=0.0,
                    omega=1.0,
                    magnus_method=magnus_method,
                    expm_method=expm_method,
                    backend=backend,
                    chebyshev_degree=chebyshev_degree,
                )
            else:
                U_step = expm_magnus(
                    (-1j * (t_target - self._t)) * self._H0,
                    method=expm_method,
                    degree=chebyshev_degree,
                )
            self.total_steps += 1
        elif not use_adaptive:
            steps = self._compute_fixed_steps(t_target)
            self.total_steps += steps
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
            elif len(self._channels) == 1:
                H1_mat, f1_fn = self._channels[0]
                use_harm = bool(self._options.get("use_harmonic_core", True))
                if use_harm and backend != "Cuda" and self._harmonic_params is not None:
                    h_eps, h_omega = self._harmonic_params
                    t_mid = 0.5 * (self._t + t_target)
                    tol_h = 1e-11 * max(1.0, abs(h_eps))
                    if all(
                        abs(f1_fn(tv) - h_eps * math.cos(h_omega * tv)) <= tol_h
                        for tv in (self._t, t_mid, t_target)
                    ):
                        U_step = solve_magnus(
                            H0=self._H0,
                            H1=H1_mat,
                            t_start=self._t,
                            t_final=t_target,
                            steps=steps,
                            epsilon=h_eps,
                            omega=h_omega,
                            magnus_method=magnus_method,
                            expm_method=expm_method,
                            backend=backend,
                            chebyshev_degree=chebyshev_degree,
                        )
                    else:
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
        else:
            order = _METHOD_ORDER.get(magnus_method, 6)
            self._ensure_error_norms(order)
            t_nodes, payload = self._build_adaptive_grid(self._t, t_target, order)

            if self._use_oracle:
                opts = _build_options(
                    magnus_method=magnus_method,
                    expm_method=expm_method,
                    backend=backend,
                    chebyshev_degree=chebyshev_degree,
                )
                A_gl = np.ascontiguousarray(payload, dtype=np.complex128)
                N = A_gl.shape[2]
                U_step = np.empty((N, N), dtype=np.complex128, order="C")
                _magnus_py.propagate_general_gl_cpu(t_nodes, A_gl, opts, U_step)
            elif len(self._channels) == 1:
                H1_mat, _ = self._channels[0]
                use_harm = bool(self._options.get("use_harmonic_core", True))
                matched_harmonic = False
                if use_harm and backend != "Cuda" and self._harmonic_params is not None:
                    h_eps, h_omega = self._harmonic_params
                    t_gl = (
                        t_nodes[:-1, None]
                        + (t_nodes[1:] - t_nodes[:-1])[:, None] * _GL3_C_ARR[None, :]
                    )
                    tol_h = 1e-11 * max(1.0, abs(h_eps))
                    matched_harmonic = bool(
                        np.all(np.abs(payload - h_eps * np.cos(h_omega * t_gl)) <= tol_h)
                    )

                if matched_harmonic:
                    U_step = solve_magnus(
                        H0=self._H0,
                        H1=H1_mat,
                        time_nodes=t_nodes,
                        epsilon=h_eps,
                        omega=h_omega,
                        magnus_method=magnus_method,
                        expm_method=expm_method,
                        backend=backend,
                        chebyshev_degree=chebyshev_degree,
                    )
                else:
                    U_step = solve_magnus(
                        H0=self._H0,
                        H1=H1_mat,
                        time_nodes=t_nodes,
                        gl_vals=payload,
                        magnus_method=magnus_method,
                        expm_method=expm_method,
                        backend=backend,
                        chebyshev_degree=chebyshev_degree,
                    )
            else:
                opts = _build_options(
                    magnus_method=magnus_method,
                    expm_method=expm_method,
                    backend=backend,
                    chebyshev_degree=chebyshev_degree,
                )
                N = self._H0.shape[0]
                steps = t_nodes.size - 1
                A_gl = np.broadcast_to((-1j * self._H0)[None, None, :, :], (steps, 3, N, N)).copy()
                for (H_k, _), f_vals in zip(self._channels, payload):
                    A_gl += (-1j * f_vals[:, :, None, None]) * H_k[None, None, :, :]
                A_gl = np.ascontiguousarray(A_gl, dtype=np.complex128)
                U_step = np.empty((N, N), dtype=np.complex128, order="C")
                _magnus_py.propagate_general_gl_cpu(t_nodes, A_gl, opts, U_step)

        psi_vec = self._y.to_array()
        psi_next = U_step @ psi_vec
        self._y = qt.core.data.create(psi_next)
        self._t = t_target
        return self._t, self._y


SESolver.add_integrator(MagnusChebyshevIntegrator, "magnus_chebyshev")