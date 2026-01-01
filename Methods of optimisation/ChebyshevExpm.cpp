#include "ChebyshevExpm.h"

#include "Utils.h"

namespace matrix_ops {
    CMatrix expm_cheb(const CMatrix& Omega, int N, int M)
    {
        using namespace std;

        // 1. Оценка спектрального радиуса
        double norm = mat_one_norm(Omega, N);

        if (norm < 1e-14)
            return utils::eye(N);

        // 2. Scaling and Squaring
        int s = 0;
        double scaled_norm = norm;
        while (scaled_norm > 1.0) {
            scaled_norm /= 2.0;
            s++;
        }

        // A_scaled = Omega / 2^s
        CMatrix A_scaled = mat_scale(Omega, complexd(1.0 / std::pow(2.0, s), 0.0));
        double alpha = scaled_norm;

        // 3. Нормировка для рекурсии
        // X = i * A_scaled / alpha
        CMatrix X = mat_copy(A_scaled);
        if (alpha > 1e-16) {
            mat_scale_inplace(X, N, complexd(0.0, 1.0 / alpha));
        }

        // Инициализация рекурсии
        CMatrix Tk_prev = utils::eye(N); // T_0
        CMatrix Tk_curr = mat_copy(X);   // T_1

        CMatrix res_scaled(N * N, complexd(0.0, 0.0));

        // --- FIX START: Corrected Coefficients for (-i)^k ---

        // Term k=0: J_0(alpha) * I
        double J0 = std::cyl_bessel_j(0, alpha);
        for (size_t i = 0; i < res_scaled.size(); ++i)
            res_scaled[i] += complexd(J0, 0.0) * Tk_prev[i];

        // Term k=1: 2 * (-i)^1 * J_1(alpha) * T_1
        // (-i)^1 = -i. Coefficient is -2i * J1.
        double J1 = std::cyl_bessel_j(1, alpha);

        // ИСПРАВЛЕНО: Знак минус (-2.0)
        complexd coeff1 = complexd(0.0, -2.0 * J1);

        for (size_t i = 0; i < res_scaled.size(); ++i)
            res_scaled[i] += coeff1 * Tk_curr[i];

        // Recursion k=2..M
        for (int k = 2; k <= M; ++k)
        {
            CMatrix Tk_next(N * N, complexd(0.0, 0.0));

            // T_{k+1} = 2 * X * T_k - T_{k-1}
            matmul(X, Tk_curr, Tk_next, N);
            mat_scale_inplace(Tk_next, N, complexd(2.0, 0.0));
            mat_sub(Tk_next, Tk_prev, Tk_next, N);

            double Jk = std::cyl_bessel_j(k, alpha);

            if (std::abs(Jk) < 1e-18) {
                // break; 
            }

            // ИСПРАВЛЕНО: Вычисляем (-i)^k
            complexd i_pow_k;
            int rem = k % 4;
            if (rem == 0) i_pow_k = complexd(1.0, 0.0);       // 1
            else if (rem == 1) i_pow_k = complexd(0.0, -1.0); // -i (было i)
            else if (rem == 2) i_pow_k = complexd(-1.0, 0.0); // -1
            else i_pow_k = complexd(0.0, 1.0);                // i  (было -i)

            complexd coeff = complexd(2.0 * Jk, 0.0) * i_pow_k;

            for (size_t i = 0; i < res_scaled.size(); ++i)
                res_scaled[i] += coeff * Tk_next[i];

            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        // --- FIX END ---

        // 4. Squaring
        CMatrix final_res = res_scaled;
        CMatrix temp(N * N, complexd(0.0, 0.0));

        for (int i = 0; i < s; ++i) {
            matmul(final_res, final_res, temp, N);
            final_res = temp;
        }

        return final_res;
    }
}


