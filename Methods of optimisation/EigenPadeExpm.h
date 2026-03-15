#pragma once

#include "MatrixOperations.h"

namespace matrix_ops {
    // Wrappers of library methods of e^A
    CMatrix expm_pade_eigen(const CMatrix& A, int N);
}


