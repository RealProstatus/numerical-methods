#pragma once

#include "MatrixOperations.h"

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


//  ------ Generation of permutations {2,3,...,n} ------ 
void generate_permutations(vector<int>& base, vector<vector<int>>& result, int start = 0);