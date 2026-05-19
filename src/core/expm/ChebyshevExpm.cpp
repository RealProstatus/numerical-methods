#include "ChebyshevExpm.h"

#include "Utils.h"

namespace matrix_ops {
    CMatrix expm_cheb(const CMatrix& Omega, int N, int M)
    {
        using namespace std;

        // 1. Estimate spectral radius (via 1-norm here)
        double norm = mat_one_norm(Omega, N);

        if (norm < 1e-14)
            return utils::eye(N);

        // 2. Scaling and squaring
        int s = 0;
        double scaled_norm = norm;
        while (scaled_norm > 1.0) {
            scaled_norm /= 2.0;
            s++;
        }
        double alpha = scaled_norm;

        // X = Omega * (1 / 2^s) * (i / alpha)
        complexd combined_scale(0.0, 1.0 / (std::pow(2.0, s) * alpha));
        if (alpha <= 1e-16) combined_scale = complexd(0.0, 0.0);
        CMatrix X = Omega;
        mat_scale_inplace(X, N, combined_scale);

        // Initialize recurrence
        CMatrix Tk_prev = utils::eye(N);        // T_0
        CMatrix Tk_curr = X;                    // T_1 (копируем X)
        CMatrix Tk_next(N * N);                 // Пустой буфер для T_{k+1}
        CMatrix res_scaled(N * N, complexd(0.0, 0.0));


        // Term k=0: J_0(alpha) * I
        double J0 = std::cyl_bessel_j(0, alpha);
        mat_axpy(Tk_prev, res_scaled, N, complexd(J0, 0.0));

        // Term k=1: 2 * (-i)^1 * J_1(alpha) * T_1
        // (-i)^1 = -i. Coefficient is -2i * J1.
        double J1 = std::cyl_bessel_j(1, alpha);
        // minus sign (-2.0)
        complexd coeff1 = complexd(0.0, -2.0 * J1);
        mat_axpy(Tk_curr, res_scaled, N, coeff1);

        // Recursion k=2..M
        for (int k = 2; k <= M; ++k)
        {
            double Jk = std::cyl_bessel_j(k, alpha);
            if (std::abs(Jk) < 1e-18) continue;

            // T_{k+1} = 2 * X * T_k
            // beta=0.0, чтобы занулить Tk_next внутри zgemm
            matmul(X, Tk_curr, Tk_next, N, complexd(2.0, 0.0));
            // T_{k+1} -= T_{k-1}
            mat_sub(Tk_next, Tk_prev, Tk_next, N);

            // compute (-i)^k
            complexd i_pow_k;
            int rem = k % 4;
            if (rem == 0) i_pow_k = complexd(1.0, 0.0);       // 1
            else if (rem == 1) i_pow_k = complexd(0.0, -1.0); // -i
            else if (rem == 2) i_pow_k = complexd(-1.0, 0.0); // -1
            else i_pow_k = complexd(0.0, 1.0);                // i

            complexd coeff = complexd(2.0 * Jk, 0.0) * i_pow_k;

            mat_axpy(Tk_next, res_scaled, N, coeff);

            std::swap(Tk_prev, Tk_curr);
            std::swap(Tk_curr, Tk_next);
        }

        // 4. Squaring
        CMatrix final_res = std::move(res_scaled); // avoid copying
        CMatrix temp(N * N);

        for (int i = 0; i < s; ++i) {
            matmul(final_res, final_res, temp, N);
            std::swap(final_res, temp); 
        }

        return final_res;
    }
}


