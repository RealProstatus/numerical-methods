#include "ClassicMagnus.h"

using namespace std;
using namespace matrix_ops;

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = static_cast<int>(samples.size());

    if (n_samples < 2) return result;

    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = (samples[0][k] + samples[n_samples - 1][k]) * 0.5;
    }

    for (int i = 1; i < n_samples - 1; ++i) {
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k];
        }
    }

    mat_scale_inplace(result, N, complexd(dt, 0.0));
    return result;
}

// ------ Simpson's rule integration (4th order accuracy) ------
// Нечётное число точек: чистый составной метод Симпсона 1/3.
// Чётное число точек (>= 4): Симпсон 1/3 на [0 .. n-4] + Симпсон 3/8 на [n-4 .. n-1].
CMatrix simpson_integral(const vector<CMatrix>& samples, double dt, int N) {
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
    CMatrix result(elems, complexd(0.0, 0.0));
    const int n_samples = static_cast<int>(samples.size());

    if (n_samples < 2) return result;
    if (n_samples == 2) return trapezoidal_integral(samples, dt, N);

    const bool use_38 = (n_samples % 2 == 0);
    const int limit_13 = use_38 ? (n_samples - 3) : n_samples;

    if (limit_13 >= 3) {
        for (size_t k = 0; k < elems; ++k) {
            result[k] = samples[0][k] + samples[limit_13 - 1][k];
        }

        for (int i = 1; i < limit_13 - 1; ++i) {
            const double w = (i % 2 == 1) ? 4.0 : 2.0;
            for (size_t k = 0; k < elems; ++k) {
                result[k] += samples[i][k] * w;
            }
        }

        mat_scale_inplace(result, N, complexd(dt / 3.0, 0.0));
    }

    if (use_38) {
        const int base = n_samples - 4;
        const double c38 = 3.0 * dt / 8.0;
        for (size_t k = 0; k < elems; ++k) {
            result[k] += (samples[base][k] + 3.0 * samples[base + 1][k]
                + 3.0 * samples[base + 2][k] + samples[base + 3][k]) * c38;
        }
    }

    return result;
}

// ------ Cumulative Simpson integration (4th order accuracy at every node) ------
vector<CMatrix> cumulative_simpson_integral(const vector<CMatrix>& samples, double dt, int N) {
    const int M = static_cast<int>(samples.size());
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
    vector<CMatrix> F(M, CMatrix(elems, complexd(0.0, 0.0)));

    if (M < 2) return F;

    if (M == 2) {
        const double half_dt = 0.5 * dt;
        for (size_t k = 0; k < elems; ++k) {
            F[1][k] = (samples[0][k] + samples[1][k]) * half_dt;
        }
        return F;
    }

    if (M == 3) {
        const double c12 = dt / 12.0;
        const double c3 = dt / 3.0;
        for (size_t k = 0; k < elems; ++k) {
            F[1][k] = (5.0 * samples[0][k] + 8.0 * samples[1][k] - samples[2][k]) * c12;
            F[2][k] = (samples[0][k] + 4.0 * samples[1][k] + samples[2][k]) * c3;
        }
        return F;
    }

    const double c3 = dt / 3.0;
    const double c24 = dt / 24.0;
    const double c38 = 3.0 * dt / 8.0;

    // Узел i = 1: левый кубический шаг 4-го порядка по узлам {0, 1, 2, 3}
    for (size_t k = 0; k < elems; ++k) {
        F[1][k] = (9.0 * samples[0][k] + 19.0 * samples[1][k] - 5.0 * samples[2][k] + samples[3][k]) * c24;
    }

    // Чётные узлы i = 2, 4, ... — точный составной метод Симпсона 1/3
    for (int i = 2; i < M; i += 2) {
        for (size_t k = 0; k < elems; ++k) {
            F[i][k] = F[i - 2][k] + (samples[i - 2][k] + 4.0 * samples[i - 1][k] + samples[i][k]) * c3;
        }
    }

    // Внутренние нечётные узлы i = 3, 5, ..., < M - 1 — центральный кубический шаг по {i-2, i-1, i, i+1}
    for (int i = 3; i < M - 1; i += 2) {
        for (size_t k = 0; k < elems; ++k) {
            F[i][k] = F[i - 1][k]
                + (-samples[i - 2][k] + 13.0 * samples[i - 1][k] + 13.0 * samples[i][k] - samples[i + 1][k]) * c24;
        }
    }

    // Если M чётное (последний узел нечётный): правило Симпсона 3/8 на последних 3 интервалах
    if (M % 2 == 0) {
        const int last = M - 1;
        for (size_t k = 0; k < elems; ++k) {
            F[last][k] = F[last - 3][k]
                + (samples[last - 3][k] + 3.0 * samples[last - 2][k]
                    + 3.0 * samples[last - 1][k] + samples[last][k]) * c38;
        }
    }

    return F;
}

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================

CMatrix magnus_classic(double t0, double t1, double dt, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W, int max_order)
{
    if (max_order < 1) max_order = 1;
    if (max_order > 3) max_order = 3;

    vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    const int M = static_cast<int>(A.size());
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);

    if (max_order == 1) {
        return simpson_integral(A, dt, N);
    }

    // Omega_1(t) в каждом узле методом Симпсона
    vector<CMatrix> Omega1_hist = cumulative_simpson_integral(A, dt, N);
    CMatrix Omega = mat_copy(Omega1_hist.back());

    // Omega_2(t) = 0.5 * int [A(tau), Omega_1(tau)] dtau
    vector<CMatrix> Integrand2(M, CMatrix(elems, complexd(0.0, 0.0)));
    for (int i = 1; i < M; ++i) {
        Integrand2[i] = commutator(A[i], Omega1_hist[i], N);
    }

    if (max_order == 2) {
        CMatrix Omega2 = simpson_integral(Integrand2, dt, N);
        mat_scale_inplace(Omega2, N, complexd(0.5, 0.0));
        mat_add(Omega, Omega2, Omega, N);
        return Omega;
    }

    // max_order >= 3: Omega_3 = int (0.5 * [A, Omega_2] - (1/12) * [Omega_1, [A, Omega_1]]) dtau
    vector<CMatrix> Omega2_hist = cumulative_simpson_integral(Integrand2, dt, N);
    for (int i = 1; i < M; ++i) {
        mat_scale_inplace(Omega2_hist[i], N, complexd(0.5, 0.0));
    }
    mat_add(Omega, Omega2_hist.back(), Omega, N);

    vector<CMatrix> Integrand3(M, CMatrix(elems, complexd(0.0, 0.0)));
    for (int i = 1; i < M; ++i) {
        CMatrix term1 = commutator(A[i], Omega2_hist[i], N);
        CMatrix term2 = commutator(Omega1_hist[i], Integrand2[i], N);
        mat_axpy(term1, Integrand3[i], N, complexd(0.5, 0.0));
        mat_axpy(term2, Integrand3[i], N, complexd(-1.0 / 12.0, 0.0));
    }

    CMatrix Omega3 = simpson_integral(Integrand3, dt, N);
    mat_add(Omega, Omega3, Omega, N);

    return Omega;
}

vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W) {
    vector<CMatrix> samples;
    const int n_steps = (int)std::round((t1 - t0) / dt) + 1;
    samples.reserve(n_steps);

    for (int i = 0; i < n_steps; ++i) {
        const double t = (i + 1 == n_steps) ? t1 : (t0 + i * dt);
        const double f_t = eps0 * cos(W * t);

        // A(t) = -i * (H0 + f(t) * H_mod)
        CMatrix A_t((size_t)N * N, complexd(0.0, 0.0));
        for (size_t k = 0; k < A_t.size(); ++k) {
            A_t[k] = complexd(0.0, -1.0) * (H0[k] + f_t * H_mod[k]);
        }
        samples.push_back(std::move(A_t));
    }
    return samples;
}