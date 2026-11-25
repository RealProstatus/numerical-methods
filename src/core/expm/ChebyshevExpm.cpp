#include "ChebyshevExpm.h"
#include "Utils.h"

#include <cmath>
#include <utility>

namespace matrix_ops {
    CMatrix expm_cheb(const CMatrix& Omega, int N, int M)
    {
        using namespace std;

        double norm = mat_one_norm(Omega, N);
        if (norm < 1e-14)
            return utils::eye(N);

        int s = 0;
        double scaled_norm = norm;
        while (scaled_norm > 1.0) {
            scaled_norm *= 0.5;
            s++;
        }
        double alpha = scaled_norm;

        // X = Omega * (1 / 2^s) * (i / alpha)
        const double denom = std::ldexp(alpha, s);
        complexd combined_scale = (denom > 1e-16) ? complexd(0.0, 1.0 / denom) : complexd(0.0, 0.0);

        CMatrix X = Omega;
        mat_scale_inplace(X, N, combined_scale);

        CMatrix Tk_prev = utils::eye(N); // T_0
        CMatrix Tk_curr = X;             // T_1
        CMatrix Tk_next((size_t)N * N);
        CMatrix res_scaled((size_t)N * N, complexd(0.0, 0.0));

        // k = 0: J_0(alpha) * I
        double J0 = std::cyl_bessel_j(0, alpha);
        mat_axpy(Tk_prev, res_scaled, N, complexd(J0, 0.0));

        // k = 1: -2i * J_1(alpha) * T_1
        if (M >= 1) {
            double J1 = std::cyl_bessel_j(1, alpha);
            complexd coeff1(0.0, -2.0 * J1);
            mat_axpy(Tk_curr, res_scaled, N, coeff1);
        }

        // k = 2..M
        for (int k = 2; k <= M; ++k) {
            double Jk = std::cyl_bessel_j(k, alpha);
            if (std::abs(Jk) < 1e-18) break;

            matmul(X, Tk_curr, Tk_next, N, complexd(2.0, 0.0));
            mat_sub(Tk_next, Tk_prev, Tk_next, N);

            complexd i_pow_k;
            int rem = k % 4;
            if (rem == 0)      i_pow_k = complexd(1.0, 0.0);
            else if (rem == 1) i_pow_k = complexd(0.0, -1.0);
            else if (rem == 2) i_pow_k = complexd(-1.0, 0.0);
            else               i_pow_k = complexd(0.0, 1.0);

            complexd coeff = complexd(2.0 * Jk, 0.0) * i_pow_k;
            mat_axpy(Tk_next, res_scaled, N, coeff);

            std::swap(Tk_prev, Tk_curr);
            std::swap(Tk_curr, Tk_next);
        }

        CMatrix final_res = std::move(res_scaled);
        CMatrix temp((size_t)N * N);

        for (int i = 0; i < s; ++i) {
            matmul(final_res, final_res, temp, N);
            std::swap(final_res, temp);
        }

        return final_res;
    }
}
