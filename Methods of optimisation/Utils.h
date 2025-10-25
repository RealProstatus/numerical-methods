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
}