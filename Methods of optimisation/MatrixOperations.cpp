#include "MatrixOperations.h"
#include "Utils.h"
#include <mkl.h>
#include <iostream>
#include <map>
#include <array>
#include <cmath>

namespace matrix_ops {
    // 1. Базовые матричные операции
    void matmul(const CMatrix& A, const CMatrix& B, CMatrix& C, int N,
        complexd alpha, complexd beta) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            N, N, N, &alpha,
            (const void*)A.data(), N,
            (const void*)B.data(), N, &beta,
            (void*)C.data(), N);
    }

    double mat_one_norm(const CMatrix& A, int N) {
        double maxcol = 0.0;
        for (int j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int i = 0; i < N; ++i) sum += std::abs(A[utils::idx(i, j, N)]);
            if (sum > maxcol) maxcol = sum;
        }
        return maxcol;
    }

    void mat_add(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
        size_t n = (size_t)N * N;
        for (size_t k = 0; k < n; ++k) C[k] = A[k] + B[k];
    }

    void mat_sub(const CMatrix& A, const CMatrix& B, CMatrix& C, int N) {
        size_t n = (size_t)N * N;
        for (size_t k = 0; k < n; ++k) C[k] = A[k] - B[k];
    }

    void mat_scale_inplace(CMatrix& A, int N, complexd alpha) {
        size_t n = (size_t)N * N;
        for (size_t k = 0; k < n; ++k) A[k] *= alpha;
    }

    CMatrix mat_copy(const CMatrix& A) {
        return CMatrix(A);
    }

    CMatrix mat_scale(const CMatrix& A, complexd alpha) {
        CMatrix result = A;
        for (size_t k = 0; k < result.size(); ++k)
            result[k] *= alpha;
        return result;
    }

    // 2. Коммутаторы
    CMatrix commutator(const CMatrix& A, const CMatrix& B, int N) {
        CMatrix AB((size_t)N * N, complexd(0.0, 0.0));
        CMatrix BA((size_t)N * N, complexd(0.0, 0.0));
        CMatrix result((size_t)N * N, complexd(0.0, 0.0));

        matmul(A, B, AB, N);
        matmul(B, A, BA, N);
        mat_sub(AB, BA, result, N);

        return result;
    }

    CMatrix iterated_commutator(const CMatrix& X, const CMatrix& Y, int k, int N) {
        CMatrix result = mat_copy(Y);
        for (int i = 0; i < k; ++i) {
            result = commutator(X, result, N);
        }
        return result;
    }

    // 3. Матричная экспонента
    CMatrix expm_taylor(const CMatrix& A, int N, int terms) {
        CMatrix result = utils::eye(N);
        CMatrix term = utils::eye(N);
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));

        for (int n = 1; n <= terms; ++n) {
            matmul(A, term, temp, N);
            term = temp;
            double factorial = 1.0;
            for (int i = 1; i <= n; ++i) factorial *= i;

            for (size_t k = 0; k < result.size(); ++k) {
                result[k] += term[k] / factorial;
            }
        }

        return result;
    }

    // ====== Exponential via Chebyshev Polynomials ======
    // Uses Chebyshev recursion: T0=I, T1=Ω̃, T_{k+1} = 2Ω̃ T_k - T_{k-1}
    // Coeffs c_k are given constants (Bessel-related), pre-filled
    CMatrix expm_chebyshev(const CMatrix& Omega, int N, int K)
    {
        // Helper: result += alpha * B
        auto mat_scale_add_inplace = [&](CMatrix& R, const CMatrix& B, complexd alpha)
            {
                size_t nn = (size_t)N * N;
                for (size_t i = 0; i < nn; ++i)
                    R[i] += alpha * B[i];
            };

        // 1. Norm scaling
        double alpha = mat_one_norm(Omega, N);
        if (alpha < 1e-15) return utils::eye(N);

        CMatrix Omega_scaled = mat_copy(Omega);
        mat_scale_inplace(Omega_scaled, N, complexd(1.0 / alpha, 0.0));

        // 2. Predefined coefficients
        static const double c_vals[] = {
            1.0, 0.5, 0.25, 0.125, 0.0625,
            0.03125, 0.015625, 0.0078125,
            0.00390625, 0.001953125,
            0.0009765625
        };
        int maxC = sizeof(c_vals) / sizeof(double);
        if (K >= maxC) K = maxC - 1;

        // 3. Recursion
        CMatrix Tkm1 = utils::eye(N);
        CMatrix Tk = mat_copy(Omega_scaled);
        CMatrix result(N * N, complexd(0, 0));

        mat_scale_add_inplace(result, Tkm1, complexd(c_vals[0], 0.0));
        mat_scale_add_inplace(result, Tk, complexd(c_vals[1], 0.0));

        for (int k = 2; k <= K; ++k)
        {
            CMatrix next(N * N, complexd(0, 0));

            // next = 2Ω̃Tk - Tkm1
            matmul(Omega_scaled, Tk, next, N);
            mat_scale_inplace(next, N, complexd(2.0, 0.0));
            mat_sub(next, Tkm1, next, N);

            mat_scale_add_inplace(result, next, complexd(c_vals[k], 0.0));

            Tkm1 = Tk;
            Tk = next;
        }

        return result;
    }



    // 4. Функции для разложения Магнуса
    double bernoulli_number(int j) {
        switch (j) {
        case 1: return -1.0 / 2.0;
        case 2: return 1.0 / 6.0;
        case 3: return 0.0;
        case 4: return -1.0 / 30.0;
        case 5: return 0.0;
        case 6: return 1.0 / 42.0;
        case 7: return 0.0;
        case 8: return -1.0 / 30.0;
        case 9: return 0.0;
        case 10: return 5.0 / 66.0;
        default: return 0.0;
        }
    }

    CMatrix compute_ad_Omega_k(const CMatrix& Omega, const CMatrix& A, int k, int N) {
        CMatrix result = mat_copy(A);
        for (int i = 0; i < k; ++i) {
            result = commutator(Omega, result, N);
        }
        return result;
    }

    vector<CMatrix> compute_S_n_j(int n, int j,
        const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples, int N)
    {
        // 🔹 Статический кэш для уже вычисленных (n, j)
        static std::map<std::pair<int, int>, vector<CMatrix>> cache;

        auto key = std::make_pair(n, j);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second; // Возвращаем уже готовое значение
        }

        vector<CMatrix> S_samples(A_samples.size());

        if (j == 1) {
            // S_n^(1)(t) = [Ω_(n-1), A(t)]
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = commutator(Omega[n - 1], A_samples[t], N);
        }
        else if (j == n - 1) {
            // S_n^(n-1)(t) = ad_{Ω_1}^{n-1}(A(t))
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = compute_ad_Omega_k(Omega[1], A_samples[t], n - 1, N);
        }
        else {
            // Общий случай: рекурсивное определение
            for (size_t t = 0; t < A_samples.size(); ++t) {
                CMatrix sum((size_t)N * N, complexd(0.0, 0.0));
                for (int m = 1; m <= n - j; ++m) {
                    // 🔹 Рекурсивный вызов, но с кэшированием
                    auto key_sub = std::make_pair(n - m, j - 1);
                    vector<CMatrix> S_nm_j1_samples;
                    auto it2 = cache.find(key_sub);
                    if (it2 != cache.end()) {
                        S_nm_j1_samples = it2->second;
                    }
                    else {
                        S_nm_j1_samples = compute_S_n_j(n - m, j - 1, Omega, A_samples, N);
                        cache[key_sub] = S_nm_j1_samples;
                    }

                    CMatrix comm = commutator(Omega[m], S_nm_j1_samples[t], N);
                    mat_add(sum, comm, sum, N);
                }
                S_samples[t] = sum;
            }
        }

        // 🔹 Сохраняем в кэш перед возвратом
        cache[key] = S_samples;
        return S_samples;
    }

    CMatrix matrix_exp_special(const CMatrix& A, int N, double dt)
    {
        // Копируем A, т.к. zheev разрушает input
        CMatrix copy = mat_copy(A);

        std::vector<double> w(N);  // Real eigenvalues
        lapack_int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'V', 'U', N,
            reinterpret_cast<lapack_complex_double*>(copy.data()), N, w.data());

        if (info != 0) {
            std::cerr << "LAPACKE_zheev failed with info = " << info << std::endl;
            return utils::eye(N);  // Возврат I на ошибке
        }

        // Создаём diag exp(-i w_k dt)
        CMatrix exp_D((size_t)N * N, complexd(0.0, 0.0));
        for (int k = 0; k < N; ++k) {
            double phase = -w[k] * dt;
            exp_D[utils::idx(k, k, N)] = complexd(std::cos(phase), std::sin(phase));
        }

        // temp = V * exp_D
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
        matmul(copy, exp_D, temp, N);

        // result = temp * V^\dagger = V exp_D V^\dagger
        // V^\dagger = conj transpose of V
        // Но поскольку V in row-major, для conj trans используем cblas_zgemm с CblasConjTrans
        const complexd alpha(1.0, 0.0);
        const complexd beta(0.0, 0.0);

        CMatrix result((size_t)N * N, complexd(0.0, 0.0));
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasConjTrans,
            N, N, N, &alpha,
            temp.data(), N,
            copy.data(), N, &beta,
            result.data(), N);

        return result;
    }

    // === Helper: right-nested commutator ===
    //   [A1, [A2, [A3, [..., An]]]]
    // list — вектор A_samples, perm — перестановка (индексы, начиная с 1 или 0), N — размер
    CMatrix right_nested_comm(const vector<CMatrix>& A_samples, const vector<int>& perm, int N)
    {
        int k = perm.size();
        if (k == 0) return CMatrix(N * N, complexd(0, 0));

        // начинаем с последнего: A(t1)
        CMatrix R = A_samples[0]; // считаем, что A_samples[0] соответствует t1

        // идём по перестановке в обратном порядке
        for (int i = k - 1; i >= 0; --i)
            R = commutator(A_samples[perm[i]], R, N);

        return R;
    }

    // =========================================================
    // ==================== OMEGA 5 ACC ========================
    // =========================================================


    // === Compute Omega_5 using ACC coefficients ===
    CMatrix compute_Omega5_ACC(const vector<CMatrix>& A, int N) {
        // 24 перестановки для Omega5
        static const vector<vector<int>> perms = {
            {1,2,3,4},{1,2,4,3},{1,3,2,4},{1,3,4,2},{1,4,2,3},{1,4,3,2},
            {0,2,3,4},{0,2,4,3},{0,3,2,4},{0,3,4,2},{0,4,2,3},{0,4,3,2},
            {2,0,3,4},{2,0,4,3},{2,3,0,4},{2,3,4,0},{2,4,0,3},{2,4,3,0},
            {3,0,1,4},{3,0,4,1},{3,1,0,4},{3,1,4,0},{3,4,0,1},{3,4,1,0}
        };

        static const vector<double> coeffs = {
            -1.0 / 20, 1.0 / 30, 1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60,
             1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60, 1.0 / 30, -1.0 / 60,
             1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60, -1.0 / 60, -1.0 / 60,
             1.0 / 30, -1.0 / 60, 1.0 / 30, -1.0 / 60, -1.0 / 60, -1.0 / 20
        };

        CMatrix sum(N * N, complexd(0, 0));

        for (size_t i = 0; i < perms.size(); ++i) {
            CMatrix comm = right_nested_comm(A, perms[i], N);
            mat_scale_inplace(comm, N, complexd(coeffs[i], 0));
            mat_add(sum, comm, sum, N);
        }

        return sum;
    }

    // =========================================================
    // ==================== OMEGA 6 ACC ========================
    // =========================================================

    // === Compute Omega_6 using ACC coefficients ===
    CMatrix compute_Omega6_ACC(const vector<CMatrix>& A, int N) {
        // 120 перестановок для Omega6
        static const vector<vector<int>> perms = { {
        {2,3,4,5,6},{2,3,4,6,5},{2,3,5,4,6},{2,3,5,6,4},
        {2,3,6,4,5},{2,3,6,5,4},{2,4,3,5,6},{2,4,3,6,5},
        {2,4,5,3,6},{2,4,5,6,3},{2,4,6,3,5},{2,4,6,5,3},
        {2,5,3,4,6},{2,5,3,6,4},{2,5,4,3,6},{2,5,4,6,3},
        {2,5,6,3,4},{2,5,6,4,3},{2,6,3,4,5},{2,6,3,5,4},
        {2,6,4,3,5},{2,6,4,5,3},{2,6,5,3,4},{2,6,5,4,3},
        {3,2,4,5,6},{3,2,4,6,5},{3,2,5,4,6},{3,2,5,6,4},
        {3,2,6,4,5},{3,2,6,5,4},{3,4,2,5,6},{3,4,2,6,5},
        {3,4,5,2,6},{3,4,5,6,2},{3,4,6,2,5},{3,4,6,5,2},
        {3,5,2,4,6},{3,5,2,6,4},{3,5,4,2,6},{3,5,4,6,2},
        {3,5,6,2,4},{3,5,6,4,2},{3,6,2,4,5},{3,6,2,5,4},
        {3,6,4,2,5},{3,6,4,5,2},{3,6,5,2,4},{3,6,5,4,2},
        {4,2,3,5,6},{4,2,3,6,5},{4,2,5,3,6},{4,2,5,6,3},
        {4,2,6,3,5},{4,2,6,5,3},{4,3,2,5,6},{4,3,2,6,5},
        {4,3,5,2,6},{4,3,5,6,2},{4,3,6,2,5},{4,3,6,5,2},
        {4,5,2,3,6},{4,5,2,6,3},{4,5,3,2,6},{4,5,3,6,2},
        {4,5,6,2,3},{4,5,6,3,2},{4,6,2,3,5},{4,6,2,5,3},
        {4,6,3,2,5},{4,6,3,5,2},{4,6,5,2,3},{4,6,5,3,2},
        {5,2,3,4,6},{5,2,3,6,4},{5,2,4,3,6},{5,2,4,6,3},
        {5,2,6,3,4},{5,2,6,4,3},{5,3,2,4,6},{5,3,2,6,4},
        {5,3,4,2,6},{5,3,4,6,2},{5,3,6,2,4},{5,3,6,4,2},
        {5,4,2,3,6},{5,4,2,6,3},{5,4,3,2,6},{5,4,3,6,2},
        {5,4,6,2,3},{5,4,6,3,2},{5,6,2,3,4},{5,6,2,4,3},
        {5,6,3,2,4},{5,6,3,4,2},{5,6,4,2,3},{5,6,4,3,2},
        {6,2,3,4,5},{6,2,3,5,4},{6,2,4,3,5},{6,2,4,5,3},
        {6,2,5,3,4},{6,2,5,4,3},{6,3,2,4,5},{6,3,2,5,4},
        {6,3,4,2,5},{6,3,4,5,2},{6,3,5,2,4},{6,3,5,4,2},
        {6,4,2,3,5},{6,4,2,5,3},{6,4,3,2,5},{6,4,3,5,2},
        {6,4,5,2,3},{6,4,5,3,2},{6,5,2,3,4},{6,5,2,4,3},
        {6,5,3,2,4},{6,5,3,4,2},{6,5,4,2,3},{6,5,4,3,2}
    } };

        static const vector<double> coeffs = {
        1.0 / 120, -1.0 / 180, -1.0 / 180,  1.0 / 240, -1.0 / 180,  1.0 / 360,
       -1.0 / 180,  1.0 / 240,  1.0 / 240, -1.0 / 360,  1.0 / 240, -1.0 / 720,
       -1.0 / 180,  1.0 / 240,  1.0 / 240, -1.0 / 360,  1.0 / 240, -1.0 / 720,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 180,  1.0 / 240,  1.0 / 240, -1.0 / 360,  1.0 / 240, -1.0 / 720,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 360,  1.0 / 720,  1.0 / 720, -1.0 / 1200,  1.0 / 720, -1.0 / 3600,
       -1.0 / 180,  1.0 / 240,  1.0 / 240, -1.0 / 360,  1.0 / 240, -1.0 / 720,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 360,  1.0 / 720,  1.0 / 720, -1.0 / 1200,  1.0 / 720, -1.0 / 3600,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 360,  1.0 / 720,  1.0 / 720, -1.0 / 1200,  1.0 / 720, -1.0 / 3600,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 360,  1.0 / 720,  1.0 / 720, -1.0 / 1200,  1.0 / 720, -1.0 / 3600,
        1.0 / 240, -1.0 / 360, -1.0 / 360,  1.0 / 720, -1.0 / 360,  1.0 / 1200,
       -1.0 / 360,  1.0 / 720,  1.0 / 720, -1.0 / 1200,  1.0 / 720, -1.0 / 3600
        };

        CMatrix sum(N * N, complexd(0, 0));

        for (size_t i = 0; i < perms.size(); ++i) {
            CMatrix comm = right_nested_comm(A, perms[i], N);
            mat_scale_inplace(comm, N, complexd(coeffs[i], 0));
            mat_add(sum, comm, sum, N);
        }

        return sum;
    }
}