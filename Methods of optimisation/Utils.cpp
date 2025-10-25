#include "Utils.h"
#include <iomanip>

namespace utils
{
    CMatrix generate_hermitian_matrix(int N, double dominance_factor) {
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

    void printMat(const CMatrix& A, int N, const string& name) {
        cout << name << " (" << N << "x" << N << "):\n";
        cout.setf(std::ios::fixed); cout.precision(8);
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                complexd v = A[idx(i, j, N)];
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

    void save_matrix(const CMatrix& mat, int N, const string& filename) {
        ofstream file(filename);
        file << "Matrix " << N << "x" << N << ":\n";
        file << "========================================\n";

        for (int i = 0; i < N; ++i) {
            file << "[ ";
            for (int j = 0; j < N; ++j) {
                complexd val = mat[idx(i, j, N)];

                // Форматируем вывод
                file << "(";

                // Действительная часть
                if (abs(val.real()) < 1e-10)
                    file << " 0.0000";
                else
                    file << fixed << setprecision(6) << setw(8) << val.real();

                // Мнимая часть  
                if (val.imag() >= 0)
                    file << " +" << fixed << setprecision(6) << setw(8) << val.imag() << "i";
                else
                    file << " " << fixed << setprecision(6) << setw(8) << val.imag() << "i";

                file << ")";

                if (j < N - 1) file << "  ";
            }
            file << " ]\n";
        }
        file << "========================================\n";
        file.close();
    }
}