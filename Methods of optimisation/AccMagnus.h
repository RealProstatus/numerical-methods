#pragma once

#include "MatrixOperations.h"

#include <vector>

using namespace std;

// =========================================================
// =================== CORRECTED Magnus ACC ================
// =========================================================
CMatrix magnus_ACC(
    const vector<CMatrix>& A_full_samples,
    double dt_grid,
    int N,
    int max_order);

namespace matrix_ops {
    CMatrix compute_Omega5_ACC(const vector<CMatrix>& A, int N);
    CMatrix compute_Omega6_ACC(const vector<CMatrix>& A, int N);
}


