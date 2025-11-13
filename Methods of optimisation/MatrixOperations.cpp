#include "MatrixOperations.h"
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


// =========================================================
// ============== Magnus–Chebyshev Method ==================
// =========================================================

    inline double BesselJ(int n, double x)
    {
#if __cplusplus >= 201703L
        return std::cyl_bessel_j(n, x);
#else
        return jn(n, x);
#endif
    }

    CMatrix expm_chebyshev(const CMatrix& Omega, int N, int M)
    {
        using namespace std;
        using namespace utils;
        using namespace matrix_ops;

        const double hbar = 1.0;
        const double h = 1.0; // если у тебя шаг времени delta_t != 1, передай его как параметр

        // === 1️ Вычисляем собственные значения i * Omega (должна быть эрмитова) ===
        vector<complexd> H(N * N);
        complexd I(0.0, 1.0);
        for (int k = 0; k < N * N; ++k)
            H[k] = I * Omega[k]; // H = i * Omega

        vector<double> w(N); // собственные значения
        int info = LAPACKE_zheev(LAPACK_ROW_MAJOR, 'N', 'U', N,
            reinterpret_cast<lapack_complex_double*>(H.data()), N, w.data());

        if (info != 0) {
            cerr << "LAPACKE_zheev failed, info=" << info << endl;
            return eye(N);
        }

        double omega_min = w.front();
        double omega_max = w.back();
        double Delta = (omega_max - omega_min) / 2.0;
        double beta = Delta + omega_min;
        double R = h * Delta / hbar;

        // === 2️ Строим масштабированную матрицу Omega_tilde ===
        CMatrix Omega_hat = Omega;
        complexd scale_hat = complexd(0.0, 1.0) * (hbar / h); // i * ℏ / h
        mat_scale_inplace(Omega_hat, N, scale_hat);

        CMatrix Id = eye(N);
        CMatrix shift = mat_scale(Id, complexd(Delta + omega_min, 0.0));

        CMatrix Omega_tilde = Omega_hat;
        mat_sub(Omega_tilde, shift, Omega_tilde, N);          // Ω̂ - I(Δ+Ω_min)
        mat_scale_inplace(Omega_tilde, N, complexd(1.0 / Delta, 0.0)); // / ΔΩ

        // === 3️ Рекуррент Чебышева ===
        CMatrix T0 = eye(N);
        CMatrix T1 = Omega_tilde;

        CMatrix result = mat_scale(T0, complexd(BesselJ(0, R), 0.0));

        complexd iC(0.0, 1.0);
        CMatrix tmp = mat_scale(T1, complexd(2.0, 0.0) * pow(iC, 1) * complexd(BesselJ(1, R), 0.0));
        mat_add(result, tmp, result, N);

        for (int n = 2; n <= M; ++n)
        {
            CMatrix Tn(N * N, complexd(0.0, 0.0));
            CMatrix temp2(N * N, complexd(0.0, 0.0));
            matmul(Omega_tilde, T1, temp2, N);
            mat_scale_inplace(temp2, N, complexd(2.0, 0.0));
            mat_sub(temp2, T0, Tn, N);

            complexd coeff = complexd(2.0, 0.0) * pow(iC, n) * complexd(BesselJ(n, R), 0.0);
            CMatrix term = mat_scale(Tn, coeff);
            mat_add(result, term, result, N);

            T0 = T1;
            T1 = Tn;
        }

        // === 4️ Умножаем на фазовый множитель exp(-i * βh / ℏ) ===
        complexd phase = exp(complexd(0.0, -1.0) * (beta * h / hbar));
        mat_scale_inplace(result, N, phase);


        return result;
    }
}