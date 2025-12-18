#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <random>
#include <fstream>

using namespace std;

using complexd = complex<double>;
using CMatrix = vector<complexd>;

namespace utils
{
    inline int idx(int i, int j, int N) {
        return i * N + j;
    }

    CMatrix generate_hermitian_matrix(int N, double dominance_factor = 10.0);
    void printMat(const CMatrix& A, int N, const string& name = "A");
    CMatrix eye(int N);
    void save_matrix(const CMatrix& mat, int N, const string& filename);
    pair<CMatrix, CMatrix> generate_hermitian_pair(int N, double scaleA, double scaleB);
    CMatrix analytic_omega(double h, const CMatrix& A, const CMatrix& B, int N);
    std::pair<CMatrix, CMatrix> generate_hermitian_pair_dd(int N, double scaleA, double scaleB, double noncomm_factor, double diag_offset);

    // Generate spectrally decomposable matrix H = V*D*V^(-1) with given condition number
    std::tuple<CMatrix, CMatrix, CMatrix> generate_spectral_matrix(int N, double condition_number, double diag_dominance = 3.0);
}