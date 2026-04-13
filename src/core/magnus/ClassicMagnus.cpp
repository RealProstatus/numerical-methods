#include "ClassicMagnus.h"

using namespace std;
using namespace matrix_ops;

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();

    if (n_samples < 2) return result;

    // First and last samples with weight 0.5
    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = (samples[0][k] + samples[n_samples - 1][k]) * 0.5;
    }

    // Middle samples with weight 1.0
    for (int i = 1; i < n_samples - 1; ++i) {
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k];
        }
    }

    // Multiply by dt
    mat_scale_inplace(result, N, complexd(dt, 0.0));

    return result;
}

// ------ Simpson's rule integration (4th order accuracy) ------
// Requires an odd number of points (an even number of intervals).
CMatrix simpson_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();

    if (n_samples < 3) return trapezoidal_integral(samples, dt, N);

    // Simpson's rule requires an even number of intervals (odd number of points).
    // If the number of points is even, handle the last interval with trapezoids.
    int limit = (n_samples % 2 == 1) ? n_samples : n_samples - 1;

    // Formula: (h/3) * (f0 + 4f1 + 2f2 + 4f3 + ... + fn)

    // Endpoints (weight 1)
    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = samples[0][k] + samples[limit - 1][k];
    }

    // Interior points
    for (int i = 1; i < limit - 1; ++i) {
        double w = (i % 2 == 1) ? 4.0 : 2.0; // alternating weights: 4, 2, 4, 2...
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k] * w;
        }
    }

    // Scale by dt/3
    mat_scale_inplace(result, N, complexd(dt / 3.0, 0.0));

    // If there is one leftover interval at the end (even number of samples)
    if (n_samples % 2 == 0) {
        // Add the area of the last interval using trapezoids
        for (size_t k = 0; k < result.size(); ++k) {
            complexd trap_area = (samples[n_samples - 2][k] + samples[n_samples - 1][k]) * 0.5 * dt;
            result[k] += trap_area;
        }
    }

    return result;
}

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================

CMatrix magnus_classic(double t0, double t1, double dt, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W, int max_order)
{
    if (max_order < 1) max_order = 1;
    if (max_order > 3) max_order = 3;

    vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    int M = A.size();

    CMatrix Omega((size_t)N * N, complexd(0.0, 0.0));
    CMatrix Omega1 = trapezoidal_integral(A, dt, N);
    mat_add(Omega, Omega1, Omega, N);

    if (max_order == 1) return Omega;

    CMatrix Omega2((size_t)N * N, complexd(0.0, 0.0));
    CMatrix Accumulator((size_t)N * N, complexd(0.0, 0.0));

    for (int i = 0; i < M; ++i) {
        if (i > 0) {
            CMatrix comm = commutator(A[i], Accumulator, N);
            mat_add(Omega2, comm, Omega2, N);
        }
        mat_add(Accumulator, A[i], Accumulator, N);
    }

    mat_scale_inplace(Omega2, N, complexd(dt * dt / 2.0, 0.0));
    mat_add(Omega, Omega2, Omega, N);

    if (max_order == 2) return Omega;

    if (max_order >= 3) {
        CMatrix Omega3((size_t)N * N, complexd(0.0, 0.0));
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < i; ++j)
            {
                CMatrix Cij = commutator(A[i], A[j], N);
                for (int k = 0; k < j; ++k)
                {
                    CMatrix term1 = commutator(A[i], commutator(A[j], A[k], N), N);
                    CMatrix term2 = commutator(Cij, A[k], N);
                    CMatrix sum = mat_copy(term1);
                    mat_add(sum, term2, sum, N);
                    mat_add(Omega3, sum, Omega3, N);
                }
            }
        mat_scale_inplace(Omega3, N, complexd(dt * dt * dt / 6.0, 0.0));
        mat_add(Omega, Omega3, Omega, N);
    }

    return Omega;
}


// ------ Trapezoidal integration for matrix functions ------
vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W) {
    vector<CMatrix> samples;
    int n_steps = (int)std::round((t1 - t0) / dt) + 1;

    for (int i = 0; i < n_steps; ++i) {
        double t = t0 + i * dt;
        double f_t = eps0 * cos(W * t);

        // A(t) = -i * (H0 + f(t) * H_mod)
        CMatrix A_t((size_t)N * N, complexd(0.0, 0.0));
        for (size_t k = 0; k < A_t.size(); ++k) {
            A_t[k] = complexd(0.0, -1.0) * (H0[k] + f_t * H_mod[k]);
        }
        samples.push_back(A_t);
    }
    return samples;
}