#include "TaylorExpm.h"

#include "Utils.h"

#include <cmath>

namespace matrix_ops {
    // 3. Матричная экспонента
    // Умная экспонента: Scaling and Squaring
    CMatrix expm_taylor(const CMatrix& A, int N, int terms) {
        // 1. Оцениваем норму
        double norm = mat_one_norm(A, N);

        // 2. Определяем коэффициент масштабирования s, чтобы норма стала < 0.5
        int s = 0;
        while (norm > 0.5) {
            norm /= 2.0;
            s++;
        }

        // 3. Масштабируем матрицу: A_scaled = A / 2^s
        CMatrix A_scaled = mat_scale(A, complexd(1.0 / std::pow(2.0, s), 0.0));

        // 4. Обычный Тейлор для маленькой матрицы
        CMatrix res = utils::eye(N);
        CMatrix term = utils::eye(N);
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));

        for (int n = 1; n <= terms; ++n) {
            matmul(A_scaled, term, temp, N);
            term = temp;
            double factorial = 1.0;
            for (int i = 1; i <= n; ++i) factorial *= i;

            for (size_t k = 0; k < res.size(); ++k) {
                res[k] += term[k] / factorial;
            }
        }

        // 5. Возводим в квадрат s раз: result = (e^{A/2^s})^(2^s)
        for (int i = 0; i < s; ++i) {
            // temp = res * res
            matmul(res, res, temp, N);
            res = temp;
        }

        return res;
    }
}


