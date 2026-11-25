#include "RecursiveMagnus.h"
#include "ClassicMagnus.h"

#include <cmath>
#include <cstdint>
#include <functional>
#include <utility>
#include <vector>

using namespace std;
using namespace matrix_ops;

// ================================================================
// =================== FULL ACCURACY Recursive Magnus =============
// ================================================================

CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n)
{
    const size_t elems = static_cast<size_t>(N) * static_cast<size_t>(N);
    if (max_n < 1) return CMatrix(elems, complexd(0.0, 0.0));

    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);
    const int steps = static_cast<int>(A_samples.size());
    if (steps < 2) return CMatrix(elems, complexd(0.0, 0.0));

    // Фактический точный шаг сгенерированной сетки (защита от IEEE-754 некратности)
    const double actual_dt = (t1 - t0) / static_cast<double>(steps - 1);

    vector<vector<CMatrix>> Omega_history(max_n + 1);
    const CMatrix zero_mat(elems, complexd(0.0, 0.0));

    // Omega_1(t) накопленным методом Симпсона (4-й порядок в каждом узле)
    Omega_history[1] = cumulative_simpson_integral(A_samples, actual_dt, N);

    // Предрасчет B_j / j!
    vector<double> b_over_fact(max_n, 0.0);
    double factorial = 1.0;
    for (int j = 1; j < max_n; ++j) {
        factorial *= static_cast<double>(j);
        const double B_j = matrix_ops::bernoulli_number(j);
        if (std::abs(B_j) >= 1e-12) {
            b_over_fact[j] = B_j / factorial;
        }
    }

    const int stride = max_n + 1;

    for (int n = 2; n <= max_n; ++n) {
        vector<CMatrix> Integrand(steps, zero_mat);

        // При t = 0 все Omega_k(t0) = 0, поэтому Integrand[0] тождественно нулевой
        for (int t = 1; t < steps; ++t) {
            vector<CMatrix> S_cache(static_cast<size_t>(stride) * static_cast<size_t>(stride));
            vector<uint8_t> S_ready(static_cast<size_t>(stride) * static_cast<size_t>(stride), 0);

            std::function<const CMatrix&(int, int)> get_S = [&](int k, int j) -> const CMatrix& {
                const size_t pos = static_cast<size_t>(k) * stride + static_cast<size_t>(j);
                if (S_ready[pos]) return S_cache[pos];

                CMatrix res(elems, complexd(0.0, 0.0));
                if (j == 1) {
                    res = commutator(Omega_history[k - 1][t], A_samples[t], N);
                } else {
                    for (int m = 1; m <= k - j; ++m) {
                        CMatrix term = commutator(Omega_history[m][t], get_S(k - m, j - 1), N);
                        mat_add(res, term, res, N);
                    }
                }
                S_cache[pos] = std::move(res);
                S_ready[pos] = 1;
                return S_cache[pos];
            };

            for (int j = 1; j <= n - 1; ++j) {
                if (b_over_fact[j] == 0.0) continue;
                const CMatrix& S_val = get_S(n, j);
                mat_axpy(S_val, Integrand[t], N, complexd(b_over_fact[j], 0.0));
            }
        }

        // Omega_n(t) накопленным методом Симпсона (4-й порядок в каждом узле)
        Omega_history[n] = cumulative_simpson_integral(Integrand, actual_dt, N);
    }

    CMatrix Omega_total(elems, complexd(0.0, 0.0));
    const int last_idx = steps - 1;
    for (int n = 1; n <= max_n; ++n) {
        mat_add(Omega_total, Omega_history[n][last_idx], Omega_total, N);
    }
    return Omega_total;
}
