#include "Utils.h"
#include "MatrixOperations.h"
#include <iomanip>
#include <tuple>
#include <random>

namespace utils
{
    CMatrix generate_hermitian_matrix(int N, double dominance_factor) {
        CMatrix matrix((size_t)N * N, { 0.0,0.0 });

        mt19937 gen(0);
        uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < N; i++) {
            // Диагональное преобладание ×10
            matrix[idx(i, i, N)] = { (dist(gen) * dominance_factor + 50.0) * 10.0, 0.0 };

            for (int j = i + 1; j < N; j++) {
                double real = dist(gen);
                double imag = dist(gen);
                matrix[idx(i, j, N)] = { real, imag };
                matrix[idx(j, i, N)] = { real, -imag };
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

    // Научная запись с 12 значащими цифрами
    file << scientific << setprecision(12);

    for (int i = 0; i < N; ++i) {
        file << "[ ";
        for (int j = 0; j < N; ++j) {
            complexd val = mat[idx(i, j, N)];

            file << "(";
            file << val.real();
            if (val.imag() >= 0) file << " +";
            file << val.imag() << "i";
            file << ")";

            if (j < N - 1) file << "  ";
        }
        file << " ]\n";
    }
    file << "========================================\n";
    file.close();
}

    pair<CMatrix, CMatrix> generate_hermitian_pair(int N, double scaleA, double scaleB)
    {
        auto genH = [&](double scale) {
            CMatrix H((size_t)N * N, { 0.0,0.0 });

            random_device rd;
            mt19937 gen(rd());
            uniform_real_distribution<double> dist(-1.0, 1.0);

            for (int i = 0; i < N; i++) {
                H[idx(i, i, N)] = { dist(gen) * scale + 20, 0.0 };
                for (int j = i + 1; j < N; j++) {
                    double r = dist(gen) * scale;
                    double im = dist(gen) * scale;
                    H[idx(i, j, N)] = { r, im };
                    H[idx(j, i, N)] = { r, -im };
                }
            }
            return H;
            };

        return { genH(scaleA), genH(scaleB) };
    }

    CMatrix analytic_omega(double h, const CMatrix& A, const CMatrix& B, int N)
    {
        CMatrix Om((size_t)N * N);

        double half_h2 = 0.5 * h * h;

        for (int i = 0; i < N * N; ++i)
            Om[i] = A[i] * h + B[i] * half_h2;

        return Om;
    }

    // Генерация пары эрмитовых матриц N×N с контролем диагонального преобладания.
// scaleA, scaleB — базовый масштаб вне-диагонали (умножается на случайные числа).
// noncomm_factor — множитель, которым мы умно усиливаем B (или вне-диагональ A),
//                    чтобы увеличить [A,B] без разрушения диагонального преобладания.
// diag_offset — добавляется к каждой диагонали (делает матрицы диагонально-преобладающими).
    std::pair<CMatrix, CMatrix> generate_hermitian_pair_dd(int N,
        double scaleA,
        double scaleB,
        double noncomm_factor = 5.0,
        double diag_offset = 50.0)
    {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        auto gen_one = [&](double scale, double diag_off, bool amplify_offdiag = false) {
            CMatrix H((size_t)N * N, complexd(0.0, 0.0));

            // Сначала создаём случайные верхнетреугольные значения
            for (int i = 0; i < N; ++i) {
                for (int j = i + 1; j < N; ++j) {
                    double re = dist(gen) * scale;
                    double im = dist(gen) * scale;
                    if (amplify_offdiag) { re *= noncomm_factor; im *= noncomm_factor; }
                    H[idx(i, j, N)] = complexd(re, im);
                    H[idx(j, i, N)] = complexd(re, -im);
                }
            }

            // Теперь делаем диагональ так, чтобы соблюдать строгую диагональную преобладание:
            // diag[i] = diag_offset + sum_j |offdiag_row_j| * safety_factor
            for (int i = 0; i < N; ++i) {
                double row_sum = 0.0;
                for (int j = 0; j < N; ++j) {
                    if (i == j) continue;
                    row_sum += std::abs(H[idx(i, j, N)]);
                }
                // safety_factor > 1 гарантирует строгую преобладание
                double safety_factor = 1.2;
                double diag_val = diag_off + safety_factor * row_sum;
                // добавим небольшой случайный шум, чтобы не быть полностью детерминированным
                diag_val += dist(gen) * 0.01 * diag_off;
                H[idx(i, i, N)] = complexd(diag_val, 0.0);
            }

            return H;
            };

        // Генерируем A с "обычными" off-diag, а B — с усиленными off-diag (по noncomm_factor)
        CMatrix A = gen_one(scaleA, diag_offset, false);
        CMatrix B = gen_one(scaleB, diag_offset, true);

        return { A, B };
    }

    // Generate spectrally decomposable matrix H = V*D*V^(-1) with given condition number
    std::tuple<CMatrix, CMatrix, CMatrix> generate_spectral_matrix(int N, double condition_number, double diag_dominance) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        // Generate random unitary matrix V (simplified as orthogonal matrix)
        CMatrix V(N * N, complexd(0.0, 0.0));
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double re = dist(gen);
                double im = dist(gen);
                V[idx(i, j, N)] = complexd(re, im);
            }
        }

        // Make V orthogonal (simplified Gram-Schmidt)
        for (int i = 0; i < N; ++i) {
            // Normalize column i
            complexd norm = 0.0;
            for (int k = 0; k < N; ++k) {
                norm += V[idx(k, i, N)] * conj(V[idx(k, i, N)]);
            }
            norm = sqrt(norm);
            for (int k = 0; k < N; ++k) {
                V[idx(k, i, N)] /= norm;
            }

            // Orthogonalize remaining columns
            for (int j = i + 1; j < N; ++j) {
                complexd dot = 0.0;
                for (int k = 0; k < N; ++k) {
                    dot += conj(V[idx(k, i, N)]) * V[idx(k, j, N)];
                }
                for (int k = 0; k < N; ++k) {
                    V[idx(k, j, N)] -= dot * V[idx(k, i, N)];
                }
            }
        }

        // Generate diagonal matrix D with eigenvalues spanning the condition number range
        CMatrix D(N * N, complexd(0.0, 0.0));
        double lambda_max = diag_dominance;
        double lambda_min = lambda_max / condition_number;

        // Create logarithmically spaced eigenvalues
        for (int i = 0; i < N; ++i) {
            double ratio = static_cast<double>(i) / (N - 1);
            double lambda = lambda_min * pow(lambda_max / lambda_min, ratio);
            D[idx(i, i, N)] = complexd(lambda, 0.0);
        }

        // Compute H = V * D * V^(-1)
        // For simplicity, since V is unitary, V^(-1) = V^H (Hermitian conjugate)
        CMatrix Vh = matrix_ops::dagger(V, N);
        CMatrix VD(N * N, complexd(0.0, 0.0));
        matrix_ops::matmul(V, D, VD, N);
        CMatrix H(N * N, complexd(0.0, 0.0));
        matrix_ops::matmul(VD, Vh, H, N);

        return std::make_tuple(H, V, D);
    }
}