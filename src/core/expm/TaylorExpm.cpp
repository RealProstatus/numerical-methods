#include "TaylorExpm.h"

namespace matrix_ops {
    // 3. Matrix exponential
    // Scaling and squaring
    CMatrix expm_taylor(const CMatrix& A, int N, int terms) {
        // 1. Estimate the norm
        double norm = mat_one_norm(A, N);

        // 2. Choose scaling factor s so that the norm becomes < 0.5
        int s = 0;
        while (norm > 0.5) {
            norm /= 2.0;
            s++;
        }

        // 3. Scale the matrix: A_scaled = A / 2^s
        CMatrix A_scaled = mat_scale(A, complexd(1.0 / std::pow(2.0, s), 0.0));

        // 4. Plain Taylor series for the small matrix
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

        // 5. Square s times: result = (e^{A/2^s})^(2^s)
        for (int i = 0; i < s; ++i) {
            // temp = res * res
            matmul(res, res, temp, N);
            res = temp;
        }

        return res;
    }
}


