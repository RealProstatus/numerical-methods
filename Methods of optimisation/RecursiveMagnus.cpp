#include "RecursiveMagnus.h"
#include "ClassicMagnus.h"

#include <functional>

using namespace std;
using namespace matrix_ops;

// ================================================================
// =================== FULL ACCURACY Recursive Magnus =============
// ================================================================

CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n)
{
    // 1. Generate A(t) samples
    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);
    int steps = A_samples.size();

    // Trapezoids do not require an even number of steps,
    // but if steps is 0 or 1, the integration loop will not run.

    // 2. History storage (initialized with zeros)
    vector<vector<CMatrix>> Omega_history(max_n + 1, vector<CMatrix>(steps));
    CMatrix zero_mat((size_t)N * N, complexd(0.0, 0.0));

    for (int k = 0; k <= max_n; ++k) {
        for (int t = 0; t < steps; ++t) {
            Omega_history[k][t] = zero_mat;
        }
    }

    // ============================================================
    // === Omega_1(t): trapezoidal integral of A(t) ===
    // ============================================================
    // Omega_1(t_i) = Omega_1(t_{i-1}) + (dt/2) * [A(t_{i-1}) + A(t_i)]

    double dt_half = integration_dt * 0.5;

    for (int i = 1; i < steps; ++i) {
        // step_val = A[i-1] + A[i]
        CMatrix step_val = mat_copy(A_samples[i - 1]);
        mat_add(step_val, A_samples[i], step_val, N);

        // step_val *= dt / 2
        mat_scale_inplace(step_val, N, complexd(dt_half, 0.0));

        // Omega[i] = Omega[i-1] + step_val
        mat_add(Omega_history[1][i - 1], step_val, Omega_history[1][i], N);
    }

    // ============================================================
    // === Omega_n(t): recursion ===
    // ============================================================
    for (int n = 2; n <= max_n; ++n) {

        vector<CMatrix> Integrand(steps, zero_mat);

        // 1. Build integrand S_n (commutators) for all time points
        for (int t = 0; t < steps; ++t) {
            // Cache lives inside the time loop because each time point has its own matrices
            using SKey = std::pair<int, int>;
            std::map<SKey, CMatrix> S_cache;

            std::function<CMatrix(int, int)> get_S =
                [&](int k, int j) -> CMatrix {

                    SKey key = { k, j };
                    if (S_cache.count(key)) return S_cache.at(key);

                    CMatrix res((size_t)N * N, complexd(0, 0));

                    if (j == 1) {
                        res = commutator(Omega_history[k - 1][t], A_samples[t], N);
                    }
                    else {
                        for (int m = 1; m <= k - j; ++m) {
                            CMatrix term = commutator(Omega_history[m][t], get_S(k - m, j - 1), N);
                            mat_add(res, term, res, N);
                        }
                    }
                    S_cache[key] = res;
                    return res;
                };

            for (int j = 1; j <= n - 1; ++j) {
                double B_j = matrix_ops::bernoulli_number(j);
                if (std::abs(B_j) < 1e-12) continue;

                double factorial = 1.0;
                for (int f = 1; f <= j; ++f) factorial *= f;

                CMatrix S_val = get_S(n, j);
                mat_scale_inplace(S_val, N, complexd(B_j / factorial, 0.0));
                mat_add(Integrand[t], S_val, Integrand[t], N);
            }
        }

        // 2. Integrate S_n with trapezoids to get Omega_n
        for (int i = 1; i < steps; ++i) {
            // step_val = Integrand[i-1] + Integrand[i]
            CMatrix step_val = mat_copy(Integrand[i - 1]);
            mat_add(step_val, Integrand[i], step_val, N);

            // step_val *= dt / 2
            mat_scale_inplace(step_val, N, complexd(dt_half, 0.0));

            // Omega[i] = Omega[i-1] + step_val
            mat_add(Omega_history[n][i - 1], step_val, Omega_history[n][i], N);
        }
    }

    // Sum all Omega_n at the final time point (last index)
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));
    int last_idx = steps - 1;

    for (int n = 1; n <= max_n; ++n) {
        mat_add(Omega_total, Omega_history[n][last_idx], Omega_total, N);
    }
    return Omega_total;
}


