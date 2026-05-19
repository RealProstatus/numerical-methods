#include "Formula314Magnus.h"
#include "ClassicMagnus.h"
#include <mkl.h>

using namespace std;
using namespace matrix_ops;

namespace
{
    void accumulate_commutator(const CMatrix& A, const CMatrix& B, CMatrix& Result, int N, complexd alpha)
    {
        complexd minus_alpha = complexd(-alpha.real(), -alpha.imag());
        complexd one = complexd(1.0, 0.0);

        // Result = alpha * A * B + 1.0 * Result
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, &alpha, A.data(), N, B.data(), N, &one, Result.data(), N);

        // Result = (-alpha) * B * A + 1.0 * Result
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    N, N, N, &minus_alpha, B.data(), N, A.data(), N, &one, Result.data(), N);
    }
}

// =============================================================
// ===================== Magnus via formula 3.14 ===============
// =============================================================

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

    return compute_Omega_3_14(H0, H_mod, h, f_t1, f_half, fp_half, f2_half, f2_t1, N);
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
        // Coeffs computing in advance
        double h5 = h * h * h * h * h;
        double h7 = h5 * h * h;

        double c_Hmod  = h * (f_t1 + (h * h / 24.0) * f2_half) + (h5 / 240.0) * (-fp_half * fp_half); // Объединили 2-й и 5-й члены
        double c_C1    = -(h * h * h / 12.0) * fp_half;
        double c_C2    = (h5 / 7200.0) * f2_half;
        double c_C3    = (h5 / 30.0) * f2_half * (f_half + 0.25 * h * h * f2_half);
        double c_C4    = (h5 / 7200.0) * fp_half;
        double c_C5    = (h5 / 7200.0) * (f_half * fp_half + 0.25 * h * h * f2_half);
        double c_C6    = (h5 / 7200.0) * f_half * fp_half;
        double c_C7    = (h5 / 7200.0) * (f_half * fp_half * (f_t1 + 0.25 * h * h * f2_half));
        double c_C8    = -(h7 / 14400.0) * (fp_half * fp_half);
        double c_C9    = -(h7 / 14400.0) * (fp_half * fp_half * f_half);

        // Pre-allocate
        CMatrix Omega((size_t)N * N);
        CMatrix C1((size_t)N * N);
        CMatrix C2((size_t)N * N);
        CMatrix C3((size_t)N * N);

        // NUMA First-Touch Policy: необходимо, чтобы каждое ядро "ходило" в свою NUMA Node
        // сыграет роль, только если ядра принадлежат разным NUMA Nod`ам
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N * N; ++i) {
            Omega[i] = complexd(0.0, 0.0);
            C1[i] = complexd(0.0, 0.0);
            C2[i] = complexd(0.0, 0.0);
            C3[i] = complexd(0.0, 0.0);
        }

        mat_axpy(H0, Omega, N, complexd(h, 0.0));
        mat_axpy(Hmod, Omega, N, complexd(c_Hmod, 0.0));

        // C1 = [H0, Hmod]
        commutator_inplace(H0, Hmod, C1, N);
        // adding to omega
        mat_axpy(C1, Omega, N, complexd(c_C1, 0.0));

        // C2 = [H0, C1]
        commutator_inplace(H0, C1, C2, N);
        // adding to omega
        mat_axpy(C2, Omega, N, complexd(c_C2, 0.0));

        // C4 = [H0, C2], C5 = [Hmod, C2], C8 = [C1, C2]
        accumulate_commutator(H0, C2, Omega, N, complexd(c_C4, 0.0));
        accumulate_commutator(Hmod, C2, Omega, N, complexd(c_C5, 0.0));
        accumulate_commutator(C1, C2, Omega, N, complexd(c_C8, 0.0));

        // C3 = [Hmod, C1]
        commutator_inplace(Hmod, C1, C3, N);
        // adding to omega
        mat_axpy(C3, Omega, N, complexd(c_C3, 0.0));

        // C6 = [H0, C3], C7 = [Hmod, C3], C9 = [C1, C3]
        accumulate_commutator(H0, C3, Omega, N, complexd(c_C6, 0.0));
        accumulate_commutator(Hmod, C3, Omega, N, complexd(c_C7, 0.0));
        accumulate_commutator(C1, C3, Omega, N, complexd(c_C9, 0.0));

        return Omega;
    }
}


