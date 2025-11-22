#include "MatrixOperations.h"
#include "Utils.h"
#include <mkl.h>
#include <iostream>
#include <map>
#include <array>
#include <cmath>
#include <mkl_lapacke.h>

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

        // 1. Оценка спектрального радиуса (alpha)
        // Так как Omega антиэрмитова, 1-нормы достаточно для оценки радиуса
        double alpha = mat_one_norm(Omega, N);

        if (alpha < 1e-14)
            return utils::eye(N);

        // 2. Нормировка и поворот к Эрмитову виду
        // Нам нужна матрица X с вещественными с.ч. в диапазоне [-1, 1].
        // Так как Omega ~ -i*H, то X = i * Omega / alpha
        CMatrix X = mat_copy(Omega);
        // Умножаем на i/alpha
        mat_scale_inplace(X, N, complexd(0.0, 1.0 / alpha));

        // 3. Инициализация рекурсии Чебышева
        // T_0(X) = I
        CMatrix Tk_prev = utils::eye(N);
        // T_1(X) = X
        CMatrix Tk_curr = mat_copy(X);

        CMatrix result(N * N, complexd(0.0, 0.0));

        // 4. Нулевой член разложения: J_0(alpha) * T_0
        // Используем cyl_bessel_j (обычный Бессель), так как экспонента мнимая
        double J0 = std::cyl_bessel_j(0, alpha);

        for (size_t i = 0; i < result.size(); ++i)
            result[i] += complexd(J0, 0.0) * Tk_prev[i];

        // 5. Первый член разложения: 2 * (-i)^1 * J_1(alpha) * T_1
        // (-i)^1 = -i
        double J1 = std::cyl_bessel_j(1, alpha);
        complexd coeff1 = complexd(0.0, -2.0 * J1); // -2i * J1

        for (size_t i = 0; i < result.size(); ++i)
            result[i] += coeff1 * Tk_curr[i];

        // 6. Рекурсия Чебышева для k >= 2
        // T_{k+1} = 2 * X * T_k - T_{k-1}

        for (int k = 2; k <= M; ++k)
        {
            CMatrix Tk_next(N * N, complexd(0.0, 0.0));

            // temp = X * Tk_curr
            matmul(X, Tk_curr, Tk_next, N);
            // next = 2 * temp
            mat_scale_inplace(Tk_next, N, complexd(2.0, 0.0));
            // next = next - Tk_prev
            mat_sub(Tk_next, Tk_prev, Tk_next, N);

            // Коэффициент: 2 * (-i)^k * J_k(alpha)
            double Jk = std::cyl_bessel_j(k, alpha);

            // Вычисляем (-i)^k
            complexd i_pow_k;
            int rem = k % 4;
            if (rem == 0) i_pow_k = complexd(1.0, 0.0);
            else if (rem == 1) i_pow_k = complexd(0.0, -1.0); // -i
            else if (rem == 2) i_pow_k = complexd(-1.0, 0.0);
            else i_pow_k = complexd(0.0, 1.0); // i

            complexd coeff = complexd(2.0 * Jk, 0.0) * i_pow_k;

            for (size_t i = 0; i < result.size(); ++i)
                result[i] += coeff * Tk_next[i];

            // Сдвиг для следующей итерации
            Tk_prev = Tk_curr;
            Tk_curr = Tk_next;
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

        // Начинаем с последнего в perm
        CMatrix R = A_samples[perm[k - 1]];

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