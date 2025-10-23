#pragma once

#include <iostream>
#include <complex>
#include <vector>
#include <random>

using namespace std;

using complexd = complex<double>;
using CMatrix = vector<complexd>;

namespace utils  
{
    inline int idx(int i, int j, int N) {
        return i * N + j;
    }

    CMatrix generate_hermitian_matrix(int N, double dominance_factor = 10.0) {
        CMatrix matrix((size_t)N * N, { 0.0,0.0 });

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < N; i++) {
            matrix[idx(i, i, N)] = { dist(gen) * dominance_factor + 50.0, 0.0 };

            for (int j = i + 1; j < N; j++) {
                double real = dist(gen);
                double imag = dist(gen);
                matrix[idx(i, j, N)] = { real,imag };
                matrix[idx(j, i, N)] = { real,-imag };
            }
        }

        return matrix;
    }

    void printMat(const CMatrix& A, int N, const string& name = "A") {
        cout << name << " (" << N << "x" << N << "):\n";
        cout.setf(std::ios::fixed); cout.precision(8);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                complexd v = A[idx(i, j, N)];
                // print as a+bi
                cout << "(" << v.real() << (v.imag() >= 0 ? "+" : "") << v.imag() << "i) ";
            }
            cout << "\n";
        }
        cout << "\n";
    }

    CMatrix eye(int N) {
        CMatrix I((size_t)N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N; ++i) I[idx(i, i, N)] = complexd(1.0, 0.0);
        return I;
    }
}
