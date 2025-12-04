#include "MatrixOperations.h"
#include "Utils.h"
#include <mkl.h>
#include <iostream>
#include <map>
#include <array>
#include <cmath>
#include <mkl_lapacke.h>
#include <set>

#include <Eigen/Dense>
#include <unsupported/Eigen/MatrixFunctions>
#include <Eigen/IterativeLinearSolvers>

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
    // Умная экспонента: Scaling and Squaring
    CMatrix expm_taylor(const CMatrix& A, int N, int terms) {
        // 1. Оцениваем норму
        double norm = mat_one_norm(A, N);

        // 2. Определяем коэффициент масштабирования s, чтобы норма стала < 0.5
        int s = 0;
        while (norm > 0.5) {
            norm /= 2.0;
            s++;
        }

        // 3. Масштабируем матрицу: A_scaled = A / 2^s
        CMatrix A_scaled = mat_scale(A, complexd(1.0 / std::pow(2.0, s), 0.0));

        // 4. Обычный Тейлор для маленькой матрицы
        CMatrix res = utils::eye(N);
        CMatrix term = utils::eye(N);
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));

        for (int n = 1; n <= terms; ++n) {
            matmul(A_scaled, term, temp, N);
            term = temp;
            double factorial = 1.0;
            for (int i = 1; i <= n; ++i) factorial *= i;

            for (size_t k = 0; k < res.size(); ++k) {
                res[k] += term[k] / factorial;
            }
        }

        // 5. Возводим в квадрат s раз: result = (e^{A/2^s})^(2^s)
        for (int i = 0; i < s; ++i) {
            // temp = res * res
            matmul(res, res, temp, N);
            res = temp;
        }

        return res;
    }

    CMatrix expm_cheb(const CMatrix& Omega, int N, int M)
    {
        using namespace std;

        // 1. Оценка спектрального радиуса
        double norm = mat_one_norm(Omega, N);

        if (norm < 1e-14)
            return utils::eye(N);

        // 2. Scaling and Squaring
        int s = 0;
        double scaled_norm = norm;
        while (scaled_norm > 1.0) {
            scaled_norm /= 2.0;
            s++;
        }

        // A_scaled = Omega / 2^s
        CMatrix A_scaled = mat_scale(Omega, complexd(1.0 / std::pow(2.0, s), 0.0));
        double alpha = scaled_norm;

        // 3. Нормировка для рекурсии
        // X = i * A_scaled / alpha
        CMatrix X = mat_copy(A_scaled);
        if (alpha > 1e-16) {
            mat_scale_inplace(X, N, complexd(0.0, 1.0 / alpha));
        }

        // Инициализация рекурсии
        CMatrix Tk_prev = utils::eye(N); // T_0
        CMatrix Tk_curr = mat_copy(X);   // T_1

        CMatrix res_scaled(N * N, complexd(0.0, 0.0));

        // --- FIX START: Corrected Coefficients for (-i)^k ---

        // Term k=0: J_0(alpha) * I
        double J0 = std::cyl_bessel_j(0, alpha);
        for (size_t i = 0; i < res_scaled.size(); ++i)
            res_scaled[i] += complexd(J0, 0.0) * Tk_prev[i];

        // Term k=1: 2 * (-i)^1 * J_1(alpha) * T_1
        // (-i)^1 = -i. Coefficient is -2i * J1.
        double J1 = std::cyl_bessel_j(1, alpha);

        // ИСПРАВЛЕНО: Знак минус (-2.0)
        complexd coeff1 = complexd(0.0, -2.0 * J1);

        for (size_t i = 0; i < res_scaled.size(); ++i)
            res_scaled[i] += coeff1 * Tk_curr[i];

        // Recursion k=2..M
        for (int k = 2; k <= M; ++k)
        {
            CMatrix Tk_next(N * N, complexd(0.0, 0.0));

            // T_{k+1} = 2 * X * T_k - T_{k-1}
            matmul(X, Tk_curr, Tk_next, N);
            mat_scale_inplace(Tk_next, N, complexd(2.0, 0.0));
            mat_sub(Tk_next, Tk_prev, Tk_next, N);

            double Jk = std::cyl_bessel_j(k, alpha);

            if (std::abs(Jk) < 1e-18) {
                // break; 
            }

            // ИСПРАВЛЕНО: Вычисляем (-i)^k
            complexd i_pow_k;
            int rem = k % 4;
            if (rem == 0) i_pow_k = complexd(1.0, 0.0);       // 1
            else if (rem == 1) i_pow_k = complexd(0.0, -1.0); // -i (было i)
            else if (rem == 2) i_pow_k = complexd(-1.0, 0.0); // -1
            else i_pow_k = complexd(0.0, 1.0);                // i  (было -i)

            complexd coeff = complexd(2.0 * Jk, 0.0) * i_pow_k;

            for (size_t i = 0; i < res_scaled.size(); ++i)
                res_scaled[i] += coeff * Tk_next[i];

            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
        }
        // --- FIX END ---

        // 4. Squaring
        CMatrix final_res = res_scaled;
        CMatrix temp(N * N, complexd(0.0, 0.0));

        for (int i = 0; i < s; ++i) {
            matmul(final_res, final_res, temp, N);
            final_res = temp;
        }

        return final_res;
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

    // ... (внутри MatrixOperations.cpp)

// Изменяем реализацию
    vector<CMatrix> compute_S_n_j(int n, int j,
        const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples,
        int N,
        std::map<std::pair<int, int>, vector<CMatrix>>& cache) // <--- Аргумент по ссылке
    {
        // 1. УБРАЛИ static map внутри функции.
        // Теперь мы используем cache, который нам передали.

        auto key = std::make_pair(n, j);
        auto it = cache.find(key);
        if (it != cache.end()) {
            return it->second; // Возвращаем из переданного кэша
        }

        vector<CMatrix> S_samples(A_samples.size());

        if (j == 1) {
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = commutator(Omega[n - 1], A_samples[t], N);
        }
        else if (j == n - 1) {
            for (size_t t = 0; t < A_samples.size(); ++t)
                S_samples[t] = compute_ad_Omega_k(Omega[1], A_samples[t], n - 1, N);
        }
        else {
            for (size_t t = 0; t < A_samples.size(); ++t) {
                CMatrix sum((size_t)N * N, complexd(0.0, 0.0));
                for (int m = 1; m <= n - j; ++m) {

                    // 2. ВАЖНО: Проверяем кэш перед рекурсией (это уже есть выше, но здесь для ясности)
                    auto key_sub = std::make_pair(n - m, j - 1);
                    vector<CMatrix> S_nm_j1_samples;

                    auto it2 = cache.find(key_sub);
                    if (it2 != cache.end()) {
                        S_nm_j1_samples = it2->second;
                    }
                    else {
                        // ПЕРЕДАЕМ cache ДАЛЬШЕ В РЕКУРСИЮ!
                        S_nm_j1_samples = compute_S_n_j(n - m, j - 1, Omega, A_samples, N, cache);
                        // cache сам обновится внутри, так как передан по ссылке
                    }

                    CMatrix comm = commutator(Omega[m], S_nm_j1_samples[t], N);
                    mat_add(sum, comm, sum, N);
                }
                S_samples[t] = sum;
            }
        }

        // Сохраняем в переданный кэш
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

        // 1. Начинаем с самого внутреннего элемента (последний индекс в perm)
        CMatrix R = A_samples[perm.back()];

        // 2. Идем от предпоследнего к началу
        for (int i = k - 2; i >= 0; --i) {
            R = commutator(A_samples[perm[i]], R, N);
        }

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

    CMatrix compute_Omega6_ACC(const vector<CMatrix>& A, int N)
    {
        using VI = std::vector<int>;

        // ------------------------------
        // 120 permutations for Omega_6
        // Already 0-based (i.e. {1,2,3,4,5,6} -> {0,1,2,3,4,5})
        // ------------------------------
        static const std::vector<VI> perms = {
            {1,2,3,4,5},{1,2,3,5,4},{1,2,4,3,5},{1,2,4,5,3},
            {1,2,5,3,4},{1,2,5,4,3},{1,3,2,4,5},{1,3,2,5,4},
            {1,3,4,2,5},{1,3,4,5,2},{1,3,5,2,4},{1,3,5,4,2},
            {1,4,2,3,5},{1,4,2,5,3},{1,4,3,2,5},{1,4,3,5,2},
            {1,4,5,2,3},{1,4,5,3,2},{1,5,2,3,4},{1,5,2,4,3},
            {1,5,3,2,4},{1,5,3,4,2},{1,5,4,2,3},{1,5,4,3,2},

            {2,1,3,4,5},{2,1,3,5,4},{2,1,4,3,5},{2,1,4,5,3},
            {2,1,5,3,4},{2,1,5,4,3},{2,3,1,4,5},{2,3,1,5,4},
            {2,3,4,1,5},{2,3,4,5,1},{2,3,5,1,4},{2,3,5,4,1},
            {2,4,1,3,5},{2,4,1,5,3},{2,4,3,1,5},{2,4,3,5,1},
            {2,4,5,1,3},{2,4,5,3,1},{2,5,1,3,4},{2,5,1,4,3},
            {2,5,3,1,4},{2,5,3,4,1},{2,5,4,1,3},{2,5,4,3,1},

            {3,1,2,4,5},{3,1,2,5,4},{3,1,4,2,5},{3,1,4,5,2},
            {3,1,5,2,4},{3,1,5,4,2},{3,2,1,4,5},{3,2,1,5,4},
            {3,2,4,1,5},{3,2,4,5,1},{3,2,5,1,4},{3,2,5,4,1},
            {3,4,1,2,5},{3,4,1,5,2},{3,4,2,1,5},{3,4,2,5,1},
            {3,4,5,1,2},{3,4,5,2,1},{3,5,1,2,4},{3,5,1,4,2},
            {3,5,2,1,4},{3,5,2,4,1},{3,5,4,1,2},{3,5,4,2,1},

            {4,1,2,3,5},{4,1,2,5,3},{4,1,3,2,5},{4,1,3,5,2},
            {4,1,5,2,3},{4,1,5,3,2},{4,2,1,3,5},{4,2,1,5,3},
            {4,2,3,1,5},{4,2,3,5,1},{4,2,5,1,3},{4,2,5,3,1},
            {4,3,1,2,5},{4,3,1,5,2},{4,3,2,1,5},{4,3,2,5,1},
            {4,3,5,1,2},{4,3,5,2,1},{4,5,1,2,3},{4,5,1,3,2},
            {4,5,2,1,3},{4,5,2,3,1},{4,5,3,1,2},{4,5,3,2,1},

            {5,1,2,3,4},{5,1,2,4,3},{5,1,3,2,4},{5,1,3,4,2},
            {5,1,4,2,3},{5,1,4,3,2},{5,2,1,3,4},{5,2,1,4,3},
            {5,2,3,1,4},{5,2,3,4,1},{5,2,4,1,3},{5,2,4,3,1},
            {5,3,1,2,4},{5,3,1,4,2},{5,3,2,1,4},{5,3,2,4,1},
            {5,3,4,1,2},{5,3,4,2,1},{5,4,1,2,3},{5,4,1,3,2},
            {5,4,2,1,3},{5,4,2,3,1},{5,4,3,1,2},{5,4,3,2,1}
        };

        // -------------------------------------------------
        // Coefficients (120 numbers) — ACC table (validated)
        // -------------------------------------------------
        static const std::vector<double> coeffs = {
            1.0 / 120, -1.0 / 180, -1.0 / 180, 1.0 / 240, -1.0 / 180, 1.0 / 360,
            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,

            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,

            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,

            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600
        };

        // -------------------------------------------------
        // Compute Ω₆
        // -------------------------------------------------
        CMatrix sum(N * N, complexd(0, 0));

        for (size_t i = 0; i < perms.size(); ++i)
        {
            CMatrix R = right_nested_comm(A, perms[i], N);
            mat_scale_inplace(R, N, complexd(coeffs[i], 0));
            mat_add(sum, R, sum, N);
        }

        return sum;
    }

    CMatrix compute_Omega_3_14(
        const CMatrix& H0,
        const CMatrix& Hmod,
        double h,
        double f_t1,
        double f_half,
        double fp_half,
        double f2_half,
        double f2_t1,
        int N)
    {
        auto zero = [&]() { return CMatrix((size_t)N * N, complexd(0.0, 0.0)); };

        // ===== Compute commutators according to (3.15) =====
        // C1 = [H0, Hmod]
        CMatrix C1 = commutator(H0, Hmod, N);

        // C2 = [H0, C1], C3 = [Hmod, C1]
        CMatrix C2 = commutator(H0, C1, N);
        CMatrix C3 = commutator(Hmod, C1, N);

        // C4 = [H0, C2], C5 = [Hmod, C2]
        CMatrix C4 = commutator(H0, C2, N);
        CMatrix C5 = commutator(Hmod, C2, N);

        // C6 = [H0, C3], C7 = [Hmod, C3]
        CMatrix C6 = commutator(H0, C3, N);
        CMatrix C7 = commutator(Hmod, C3, N);

        // C8 = [C1, C2], C9 = [C1, C3]
        CMatrix C8 = commutator(C1, C2, N);
        CMatrix C9 = commutator(C1, C3, N);

        // ===== Begin constructing Omega =====
        CMatrix Omega = zero();

        auto add = [&](const CMatrix& M) { mat_add(Omega, M, Omega, N); };

        // 1) h H0
        {
            CMatrix term = H0;
            mat_scale_inplace(term, N, complexd(h, 0.0));
            add(term);
        }

        // 2) h * ( f(t1) + (h^2/24) f''(t1/2) ) * Hmod
        {
            double c = f_t1 + (h * h / 24.0) * f2_half;
            CMatrix term = Hmod;
            mat_scale_inplace(term, N, complexd(h * c, 0.0));
            add(term);
        }

        // 3) - (1/12) h^3 f'(t1/2) C1
        {
            CMatrix term = C1;
            double c = -(h * h * h / 12.0) * fp_half;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 4) + (1/7200) h^5 f''(t1/2) * C2
        {
            CMatrix term = C2;
            double c = (h * h * h * h * h / 7200.0) * f2_half;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 5) + (1/240) h^5 * ( - (f'(t1/2))^2 ) * Hmod
        {
            CMatrix term = Hmod;
            double c = (h * h * h * h * h / 240.0) * (-fp_half * fp_half);
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 6) + (1/30) h^5 f''(t1/2)*( f(t1/2)+1/4 h^2 f''(t1/2) ) * C3
        {
            double s = f_half + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 30.0) * (f2_half * s);
            CMatrix term = C3;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 7) + (1/7200) h^5 f'(t1/2) * C4
        {
            double c = (h * h * h * h * h / 7200.0) * fp_half;
            CMatrix term = C4;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 8) + (1/7200) h^5 ( f(t1/2) f'(t1/2) + 1/4 h^2 f''(t1/2) ) * C5
        {
            double s = f_half * fp_half + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 7200.0) * s;
            CMatrix term = C5;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 9) + (1/7200) h^5 f(t1/2) f'(t1/2) * C6
        {
            double c = (h * h * h * h * h / 7200.0) * (f_half * fp_half);
            CMatrix term = C6;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 10) + (1/7200) h^5 f(t1/2) f'(t1/2) ( f(t1) + 1/4 h^2 f''(t1/2) ) * C7
        {
            double s = f_t1 + 0.25 * h * h * f2_half;
            double c = (h * h * h * h * h / 7200.0) * (f_half * fp_half * s);
            CMatrix term = C7;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 11) - (1/14400) h^7 (f'(t1/2))^2 * C8
        {
            double c = -(h * h * h * h * h * h * h / 14400.0) * (fp_half * fp_half);
            CMatrix term = C8;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        // 12) - (1/14400) h^7 (f'(t1/2))^2 f(t1/2) * C9
        {
            double c = -(h * h * h * h * h * h * h / 14400.0) * (fp_half * fp_half * f_half);
            CMatrix term = C9;
            mat_scale_inplace(term, N, complexd(c, 0));
            add(term);
        }

        return Omega;
    }
    
    CMatrix dagger(const CMatrix& A, int N)
    {
        CMatrix Adag(N * N);
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                // транспонирование + комплексное сопряжение
                Adag[utils::idx(i, j, N)] = std::conj(A[utils::idx(j, i, N)]);
            }
        return Adag;
    }

    bool is_unitary(const CMatrix& U, int N, double tol)
    {
        // 1. U^\dagger
        CMatrix Ud = dagger(U, N);

        // 2. M = U^\dagger * U
        CMatrix M(N * N, complexd(0, 0));
        matmul(Ud, U, M, N);

        // 3. E = M - I
        CMatrix I = utils::eye(N);
        CMatrix E(N * N);
        mat_sub(M, I, E, N);

        // 4. ||E||_1 < tol ?
        double err = mat_one_norm(E, N);

        return (err < tol);
    }

    // В MatrixOperations.cpp
    double max_element_diff(const CMatrix& A, const CMatrix& B, int N) {
        double max_diff = 0.0;
        // Предполагаем, что CMatrix хранит N*N элементов
        for (int i = 0; i < N * N; ++i) {
            // std::abs для complexd возвращает его модуль
            double current_diff = std::abs(A[i] - B[i]);
            if (current_diff > max_diff) {
                max_diff = current_diff;
            }
        }
        return max_diff;
    }

    double max_eigenvalue_modulus_hermitian(const CMatrix& M, int N) {
        // 1. LAPACKE_zheev меняет входную матрицу, поэтому делаем копию
        std::vector<complexd> A_copy = M;

        // 2. Вектор для хранения собственных чисел (они действительны)
        std::vector<double> W(N);

        // 3. Вызов LAPACKE_zheev (z - complex, he - hermitian, ev - eigenvalues)
        // 'N' - только собственные числа, 'U' - используется верхний треугольник
        lapack_int info = LAPACKE_zheev(
            LAPACK_ROW_MAJOR,
            'N',              // 'N': compute eigenvalues only
            'U',              // 'U': upper triangle is stored
            N,                // N: size of matrix
            (MKL_Complex16*)A_copy.data(), // A: input matrix
            N,                // lda: leading dimension
            W.data()          // W: output eigenvalues (real)
        );

        if (info != 0) {
            std::cerr << "Warning: LAPACKE_zheev failed with info=" << info
                << ". Returning 0.0." << std::endl;
            return 0.0;
        }

        // 4. Находим максимальный модуль среди собственных чисел
        double max_abs_eig = 0.0;
        for (double eig : W) {
            max_abs_eig = std::max(max_abs_eig, std::abs(eig));
        }

        return max_abs_eig;
    }

    CMatrix expm_pade_eigen(const CMatrix& A, int N) {
        using namespace Eigen;

        // Convert CMatrix into Eigen MatrixXcd
        MatrixXcd eigen_A(N, N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                eigen_A(i, j) = A[i * N + j];
            }
        }

        // Comuting exp(A) with Pade + scaling/squaring
        MatrixXcd exp_result = eigen_A.exp();

        // Convert result
        CMatrix result(N * N);
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                result[i * N + j] = exp_result(i, j);
            }
        }

        return result;
    }
}