#include "MatrixOperations.h"
#include <mkl.h>
#include <iostream>
#include <map>

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
}