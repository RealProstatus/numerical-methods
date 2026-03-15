#include "EigenPadeExpm.h"

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>

namespace matrix_ops {
    CMatrix expm_pade_eigen(const CMatrix& A, int N) {
        using namespace Eigen;

        // Convert CMatrix into Eigen MatrixXcd
        MatrixXcd eigen_A(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                eigen_A(i, j) = A[i * N + j];
            }
        }

        // Comuting exp(A) with Pade + scaling/squaring
        MatrixXcd exp_result = eigen_A.exp();

        // Convert result
        CMatrix result(N * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                result[i * N + j] = exp_result(i, j);
            }
        }

        return result;
    }
}


