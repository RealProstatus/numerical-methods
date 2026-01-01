#pragma once

#include "MatrixOperations.h"

namespace matrix_ops {
    // 3. Матричная экспонента
    // Умная экспонента: Scaling and Squaring
    CMatrix expm_taylor(const CMatrix& A, int N, int terms = 30);
}


