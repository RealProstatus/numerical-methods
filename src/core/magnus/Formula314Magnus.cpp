#include "Formula314Magnus.h"
#include "ClassicMagnus.h"

using namespace std;
using namespace matrix_ops;

// =============================================================
// ===================== Magnus via formula 3.14 ===============
// =============================================================

CMatrix magnus_3_14(double t0, double t1, double dt, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W)
{
    //vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    double h = t1 - t0;
    double t_half = 0.5 * (t0 + t1);

    auto f_fun = [&](double t) { return eps0 * std::cos(W * t); };
    auto fp_fun = [&](double t) { return -eps0 * W * std::sin(W * t); };
    auto f2_fun = [&](double t) { return -eps0 * W * W * std::cos(W * t); };

    double f_t1 = f_fun(t1);
    double f_half = f_fun(t_half);
    double fp_half = fp_fun(t_half);
    double f2_half = f2_fun(t_half);
    double f2_t1 = f2_fun(t1);

    CMatrix Omega = compute_Omega_3_14(H0, H_mod, h, f_t1, f_half, fp_half, f2_half, f2_t1, N);
    return Omega;
}

namespace matrix_ops {
    CMatrix compute_Omega_3_14(
        const CMatrix& H0,
        const CMatrix& Hmod,
        double h,
        double f_t1,
        double f_half,
        double fp_half,
        double f2_half,
        double f2_t1,
        int N)
    {
        auto zero = [&]() { return CMatrix((size_t)N * N, complexd(0.0, 0.0)); };

        // ===== Compute commutators according to (3.15) =====

        // Pre-allocate
        CMatrix C1(N * N), C2(N * N), C3(N * N), C4(N * N);
        CMatrix C5(N * N), C6(N * N), C7(N * N), C8(N * N), C9(N * N);

        // C1 = [H0, Hmod]
        commutator_inplace(H0, Hmod, C1, N);

        // C2 = [H0, C1], C3 = [Hmod, C1]
        commutator_inplace(H0, C1, C2, N);
        commutator_inplace(Hmod, C1, C3, N);

        // C4 = [H0, C2], C5 = [Hmod, C2]
        commutator_inplace(H0, C2, C4, N);
        commutator_inplace(Hmod, C2, C5, N);

        // C6 = [H0, C3], C7 = [Hmod, C3]
        commutator_inplace(H0, C3, C6, N);
        commutator_inplace(Hmod, C3, C7, N);

        // C8 = [C1, C2], C9 = [C1, C3]
        commutator_inplace(C1, C2, C8, N);
        commutator_inplace(C1, C3, C9, N);

        // ===== Begin constructing Omega =====
        CMatrix Omega = zero();

        auto add_scaled = [&](CMatrix& M, double scalar) {
            mat_axpy(M, Omega, N, complexd(scalar, 0.0)); // Используем быстрый axpy вместо scale + add
        };

        // 1) h H0
        add_scaled(const_cast<CMatrix&>(H0), h);

        // 2) h * ( f(t1) + (h^2/24) f''(t1/2) ) * Hmod
        add_scaled(const_cast<CMatrix&>(Hmod), h * (f_t1 + (h * h / 24.0) * f2_half));

        // 3) - (1/12) h^3 f'(t1/2) C1
        add_scaled(C1, -(h * h * h / 12.0) * fp_half);

        // 4) + (1/7200) h^5 f''(t1/2) * C2
        double h5 = h * h * h * h * h;
        add_scaled(C2, (h5 / 7200.0) * f2_half);

        // 5) + (1/240) h^5 * ( - (f'(t1/2))^2 ) * Hmod
        add_scaled(const_cast<CMatrix&>(Hmod), (h5 / 240.0) * (-fp_half * fp_half));

        // 6) + (1/30) h^5 f''(t1/2)*( f(t1/2)+1/4 h^2 f''(t1/2) ) * C3
        add_scaled(C3, (h5 / 30.0) * f2_half * (f_half + 0.25 * h * h * f2_half));

        // 7) + (1/7200) h^5 f'(t1/2) * C4
        add_scaled(C4, (h5 / 7200.0) * fp_half);

        // 8) + (1/7200) h^5 ( f(t1/2) f'(t1/2) + 1/4 h^2 f''(t1/2) ) * C5
        add_scaled(C5, (h5 / 7200.0) * (f_half * fp_half + 0.25 * h * h * f2_half));

        // 9) + (1/7200) h^5 f(t1/2) f'(t1/2) * C6
        add_scaled(C6, (h5 / 7200.0) * f_half * fp_half);

        // 10) + (1/7200) h^5 f(t1/2) f'(t1/2) ( f(t1) + 1/4 h^2 f''(t1/2) ) * C7
        add_scaled(C7, (h5 / 7200.0) * (f_half * fp_half * (f_t1 + 0.25 * h * h * f2_half)));

        // 11) - (1/14400) h^7 (f'(t1/2))^2 * C8
        double h7 = h5 * h * h;
        add_scaled(C8, -(h7 / 14400.0) * (fp_half * fp_half));

        // 12) - (1/14400) h^7 (f'(t1/2))^2 f(t1/2) * C9
        add_scaled(C9, -(h7 / 14400.0) * (fp_half * fp_half * f_half));

        return Omega;
    }
}


