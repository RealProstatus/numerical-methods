#include "Formula314Magnus.h"
#include "ClassicMagnus.h"

using namespace std;
using namespace matrix_ops;

CMatrix magnus_3_14(double t0, double t1, double dt, int N, const CMatrix& H0, const CMatrix& H_mod, double eps0, double W)
{
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
        // C1 = [H0, Hmod]
        CMatrix C1 = commutator(H0, Hmod, N);

        // C2 = [H0, C1], C3 = [Hmod, C1]
        CMatrix C2 = commutator(H0, C1, N);
        CMatrix C3 = commutator(Hmod, C1, N);

        // C4 = [H0, C2], C5 = [Hmod, C2]
        CMatrix C4 = commutator(H0, C2, N);
        CMatrix C5 = commutator(Hmod, C2, N);

        // C6 = [H0, C3], C7 = [Hmod, C3]
        CMatrix C6 = commutator(H0, C3, N);
        CMatrix C7 = commutator(Hmod, C3, N);

        // C8 = [C1, C2], C9 = [C1, C3]
        CMatrix C8 = commutator(C1, C2, N);
        CMatrix C9 = commutator(C1, C3, N);

        // ===== Begin constructing Omega =====
        CMatrix Omega = zero();

        auto add = [&](const CMatrix& M) { mat_add(Omega, M, Omega, N); };

        // 1) h H0
        {
            CMatrix term = H0;
            mat_scale_inplace(term, N, complexd(h, 0.0));
            add(term);
        }

        // 2) h * ( f(t1) + (h^2/24) f''(t1/2) ) * Hmod
        {
            double c = f_t1 + (h * h / 24.0) * f2_half;
            CMatrix term = Hmod;
            mat_scale_inplace(term, N, complexd(h * c, 0.0));
            add(term);
        }

        // 3) - (1/12) h^3 f'(t1/2) C1
        {
            CMatrix term = C1;
            double c = -(h * h * h / 12.0) * fp_half;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 4) + (1/7200) h^5 f''(t1/2) * C2
        {
            CMatrix term = C2;
            double c = (h * h * h * h * h / 7200.0) * f2_half;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 5) + (1/240) h^5 * ( - (f'(t1/2))^2 ) * Hmod
        {
            CMatrix term = Hmod;
            double c = (h * h * h * h * h / 240.0) * (-fp_half * fp_half);
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 6) + (1/30) h^5 f''(t1/2)*( f(t1/2)+1/4 h^2 f''(t1/2) ) * C3
        {
            double s = f_half + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 30.0) * (f2_half * s);
            CMatrix term = C3;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 7) + (1/7200) h^5 f'(t1/2) * C4
        {
            double c = (h * h * h * h * h / 7200.0) * fp_half;
            CMatrix term = C4;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 8) + (1/7200) h^5 ( f(t1/2) f'(t1/2) + 1/4 h^2 f''(t1/2) ) * C5
        {
            double s = f_half * fp_half + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 7200.0) * s;
            CMatrix term = C5;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 9) + (1/7200) h^5 f(t1/2) f'(t1/2) * C6
        {
            double c = (h * h * h * h * h / 7200.0) * (f_half * fp_half);
            CMatrix term = C6;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 10) + (1/7200) h^5 f(t1/2) f'(t1/2) ( f(t1) + 1/4 h^2 f''(t1/2) ) * C7
        {
            double s = f_t1 + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 7200.0) * (f_half * fp_half * s);
            CMatrix term = C7;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 11) - (1/14400) h^7 (f'(t1/2))^2 * C8
        {
            double c = -(h * h * h * h * h * h * h / 14400.0) * (fp_half * fp_half);
            CMatrix term = C8;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 12) - (1/14400) h^7 (f'(t1/2))^2 f(t1/2) * C9
        {
            double c = -(h * h * h * h * h * h * h / 14400.0) * (fp_half * fp_half * f_half);
            CMatrix term = C9;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        return Omega;
    }
}


