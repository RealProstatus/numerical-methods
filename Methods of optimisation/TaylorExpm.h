#pragma once

#include "MatrixOperations.h"

namespace matrix_ops {
    // Matrix exponential + scaling and squaring
    CMatrix expm_taylor(const CMatrix& A, int N, int terms = 30);
}


