#pragma once

#include "Utils.h"
#include <map>

using namespace std;
using complexd = complex<double>;
using CMatrix = vector<complexd>;

namespace matrix_ops {
    void matmul(const CMatrix& A, const CMatrix& B, CMatrix& C, int N,
        complexd alpha = complexd(1.0, 0.0), complexd beta = complexd(0.0, 0.0));

    double mat_one_norm(const CMatrix& A, int N);
    void mat_add(const CMatrix& A, const CMatrix& B, CMatrix& C, int N);
    void mat_sub(const CMatrix& A, const CMatrix& B, CMatrix& C, int N);
    void mat_scale_inplace(CMatrix& A, int N, complexd alpha);
    CMatrix mat_copy(const CMatrix& A);
    CMatrix mat_scale(const CMatrix& A, complexd alpha);

    // Commutator: [A, B] = A*B - B*A
    CMatrix commutator(const CMatrix& A, const CMatrix& B, int N);

    // Iterated commutator: ad_X^k(Y) = [X, [X, ...[X, Y]...]] (k times)
    CMatrix iterated_commutator(const CMatrix& X, const CMatrix& Y, int k, int N);



    // Magnus expansion helpers
    double bernoulli_number(int j);

    // Iterated commutator used in Magnus: ad_Omega^k(A)
    CMatrix compute_ad_Omega_k(const CMatrix& Omega, const CMatrix& A, int k, int N);

    // Compute S_n^(j) for Magnus expansion (recursive formula).
    vector<CMatrix> compute_S_n_j(int n, int j,
        const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples,
        int N,
        std::map<std::pair<int, int>, vector<CMatrix>>& cache);

    CMatrix matrix_exp_special(const CMatrix& A, int N, double dt);

    CMatrix dagger(const CMatrix& A, int N);
    bool is_unitary(const CMatrix& U, int N, double tol = 1e-10);
    double max_element_diff(const CMatrix& A, const CMatrix& B, int N);
    double max_eigenvalue_modulus_hermitian(const CMatrix& M, int N);

    // Wrappers of library methods of e^A
}