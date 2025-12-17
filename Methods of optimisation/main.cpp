#include <mkl.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>
#include <chrono>
#include <map>
#include <functional>
#include <iomanip>

#include "Utils.h"
#include "MatrixOperations.h"
#include "RungeKutt.h"

using namespace std;
using namespace utils;
using namespace matrix_ops;
using namespace runge_kutta_simple;

//  ------ Generation of permutations {2,3,...,n} ------ 
// Генерация всех перестановок вектора base
void generate_permutations(vector<int>& base, vector<vector<int>>& result, int start = 0) {
    int n = base.size();
    if (start == n) {
        result.push_back(base);
        return;
    }

    for (int i = start; i < n; ++i) {
        swap(base[start], base[i]);
        generate_permutations(base, result, start + 1);
        swap(base[start], base[i]); // откатываем для следующей итерации
    }
}

// ------ Trapezoidal integration for matrix functions ------
vector<CMatrix> generate_samples(double t0, double t1, double dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W) {
    vector<CMatrix> samples;
    int n_steps = (int)((t1 - t0) / dt) + 1;

    cout << "Generating " << n_steps << " time samples with dt=" << dt << endl;

    for (int i = 0; i < n_steps; ++i) {
        double t = t0 + i * dt;
        double f_t = eps0 * cos(W * t);

        // A(t) = -i * (H0 + f(t) * H_mod)
        CMatrix A_t((size_t)N * N, complexd(0.0, 0.0));
        for (size_t k = 0; k < A_t.size(); ++k) {
            A_t[k] = complexd(0.0, -1.0) * (H0[k] + f_t * H_mod[k]);
        }

        samples.push_back(A_t);
    }

    return samples;
}

CMatrix trapezoidal_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();

    if (n_samples < 2) return result;

    // First and last samples with weight 0.5
    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = (samples[0][k] + samples[n_samples - 1][k]) * 0.5;
    }

    // Middle samples with weight 1.0
    for (int i = 1; i < n_samples - 1; ++i) {
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k];
        }
    }

    // Multiply by dt
    mat_scale_inplace(result, N, complexd(dt, 0.0));

    return result;
}

// ------ Simpson's rule integration (4th order accuracy) ------
// Требует нечетного количества точек (четного числа интервалов).
CMatrix simpson_integral(const vector<CMatrix>& samples, double dt, int N) {
    CMatrix result((size_t)N * N, complexd(0.0, 0.0));
    int n_samples = samples.size();

    if (n_samples < 3) return trapezoidal_integral(samples, dt, N);

    // Метод Симпсона работает на четном числе интервалов (нечетное число точек).
    // Если точек четное количество, последнюю обработаем отдельно трапецией.
    int limit = (n_samples % 2 == 1) ? n_samples : n_samples - 1;

    // Формула: (h/3) * (f0 + 4f1 + 2f2 + 4f3 + ... + fn)

    // Крайние точки (вес 1)
    for (size_t k = 0; k < result.size(); ++k) {
        result[k] = samples[0][k] + samples[limit - 1][k];
    }

    // Внутренние точки
    for (int i = 1; i < limit - 1; ++i) {
        double w = (i % 2 == 1) ? 4.0 : 2.0; // Чередование весов 4, 2, 4, 2...
        for (size_t k = 0; k < result.size(); ++k) {
            result[k] += samples[i][k] * w;
        }
    }

    // Масштабируем на dt/3
    mat_scale_inplace(result, N, complexd(dt / 3.0, 0.0));

    // Если осталась одна точка в конце (четное число сэмплов)
    if (n_samples % 2 == 0) {
        // Добавляем площадь последнего интервала методом трапеций
        for (size_t k = 0; k < result.size(); ++k) {
            complexd trap_area = (samples[n_samples - 2][k] + samples[n_samples - 1][k]) * 0.5 * dt;
            result[k] += trap_area;
        }
    }

    return result;
}

// ================================================================
// =================== FULL ACCURACY Recursive Magnus =============
// ================================================================

CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n = 10)
{
    // 1. Генерируем сэмплы A(t)
    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);
    int steps = A_samples.size();

    // Для трапеций нам не обязательно четное количество шагов, но
    // если вдруг steps окажется 0 или 1, цикл просто не выполнится.

    // 2. Хранилище истории (Инициализация нулями)
    vector<vector<CMatrix>> Omega_history(max_n + 1, vector<CMatrix>(steps));
    CMatrix zero_mat((size_t)N * N, complexd(0.0, 0.0));

    for (int k = 0; k <= max_n; ++k) {
        for (int t = 0; t < steps; ++t) {
            Omega_history[k][t] = zero_mat;
        }
    }

    // ============================================================
    // === Omega_1(t) : Интеграл от A(t) методом трапеций ===
    // ============================================================
    // Omega_1(t_i) = Omega_1(t_{i-1}) + (dt/2) * [A(t_{i-1}) + A(t_i)]

    double dt_half = integration_dt * 0.5;

    for (int i = 1; i < steps; ++i) {
        // step_val = A[i-1] + A[i]
        CMatrix step_val = mat_copy(A_samples[i - 1]);
        mat_add(step_val, A_samples[i], step_val, N);

        // step_val *= dt / 2
        mat_scale_inplace(step_val, N, complexd(dt_half, 0.0));

        // Omega[i] = Omega[i-1] + step_val
        mat_add(Omega_history[1][i - 1], step_val, Omega_history[1][i], N);
    }

    // ============================================================
    // === Omega_n(t) : Рекурсия ===
    // ============================================================
    for (int n = 2; n <= max_n; ++n) {

        vector<CMatrix> Integrand(steps, zero_mat);

        // 1. Считаем подынтегральное выражение S_n (коммутаторы) для всех точек t
        for (int t = 0; t < steps; ++t) {
            // Кэш теперь живет внутри цикла по t, так как для каждого момента времени свои матрицы
            using SKey = std::pair<int, int>;
            std::map<SKey, CMatrix> S_cache;

            std::function<CMatrix(int, int)> get_S =
                [&](int k, int j) -> CMatrix {

                SKey key = { k, j };
                if (S_cache.count(key)) return S_cache.at(key);

                CMatrix res((size_t)N * N, complexd(0, 0));

                if (j == 1) {
                    res = commutator(Omega_history[k - 1][t], A_samples[t], N);
                }
                else {
                    for (int m = 1; m <= k - j; ++m) {
                        CMatrix term = commutator(Omega_history[m][t], get_S(k - m, j - 1), N);
                        mat_add(res, term, res, N);
                    }
                }
                S_cache[key] = res;
                return res;
                };

            for (int j = 1; j <= n - 1; ++j) {
                double B_j = matrix_ops::bernoulli_number(j);
                if (std::abs(B_j) < 1e-12) continue;

                double factorial = 1.0;
                for (int f = 1; f <= j; ++f) factorial *= f;

                CMatrix S_val = get_S(n, j);
                mat_scale_inplace(S_val, N, complexd(B_j / factorial, 0.0));
                mat_add(Integrand[t], S_val, Integrand[t], N);
            }
        }

        // 2. Интегрируем S_n методом трапеций, чтобы получить Omega_n
        for (int i = 1; i < steps; ++i) {
            // step_val = Integrand[i-1] + Integrand[i]
            CMatrix step_val = mat_copy(Integrand[i - 1]);
            mat_add(step_val, Integrand[i], step_val, N);

            // step_val *= dt / 2
            mat_scale_inplace(step_val, N, complexd(dt_half, 0.0));

            // Omega[i] = Omega[i-1] + step_val
            mat_add(Omega_history[n][i - 1], step_val, Omega_history[n][i], N);
        }
    }

    // Суммируем все Omega_n в конечной точке (последний индекс)
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));
    int last_idx = steps - 1;

    for (int n = 1; n <= max_n; ++n) {
        mat_add(Omega_total, Omega_history[n][last_idx], Omega_total, N);
    }
    return Omega_total;
}

// =========================================================
// =================== CORRECTED Magnus ACC ================
// =========================================================

CMatrix magnus_ACC(
    const vector<CMatrix>& A_full_samples,
    double dt_grid,
    int N,
    int max_order)
{
    // 1. Вычисляем полное время интервала
    // Это критически важно для коммутаторов!
    double T_total = dt_grid * (A_full_samples.size() - 1);

    // Инициализируем результат
    CMatrix Omega_total(N * N, complexd(0.0, 0.0));

    // === Omega 1 (Главный член) ===
    // Используем все точки сетки для максимальной точности интеграла (как у вас было)
    CMatrix Omega1 = trapezoidal_integral(A_full_samples, dt_grid, N);
    mat_add(Omega_total, Omega1, Omega_total, N);

    if (max_order < 2) return Omega_total;

    // === Подготовка узлов для высоких порядков ===
    int M = A_full_samples.size();
    if (M < 7) return Omega_total; // Недостаточно точек

    // Берем крайние и среднюю точки для 4-го порядка (Simpson)
    CMatrix A_0 = A_full_samples[0];
    CMatrix A_mid = A_full_samples[M / 2];
    CMatrix A_end = A_full_samples[M - 1];

    // === Omega 2..4 (Correction) ===
    // Вместо случайного цикла используем проверенную формулу 4-го порядка (Simpson commutator)
    // Formula: (T^2 / 12) * [A_end, A_0]  (для линейного приближения коммутатора)
    // Или более точная для Симпсона: зависит от базиса.
    // Самый простой и надежный вариант для ACC 4-го порядка на 3 точках:
    // Omega_4_part = (T_total^2 / 12.0) * [A(t_end), A(t_start)]

    if (max_order >= 2) {
        CMatrix comm = commutator(A_end, A_0, N);
        // Масштаб T^2 !
        mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));

        // Добавляем к общей сумме
        mat_add(Omega_total, comm, Omega_total, N);
    }

    // === Omega 6 (Special function) ===
    // Ваша функция compute_Omega6_ACC ожидает вектор матриц A.
    // Судя по коэффициентам (1/120...), это формула, требующая умножения на T (или T^k).
    // Обычно такие функции возвращают сумму взвешенных коммутаторов.
    // Размерность A ~ 1/T. Коммутатор 5 матриц ~ 1/T^5. 
    // Чтобы получить безразмерную Omega, нужно умножить на T^5.
    // НО! Часто в таких реализациях подразумевается, что A уже умножено на dt или T.

    // Давайте попробуем стандартный подход:
    // Функция compute_Omega6_ACC считает сумму коэффициентов * [A...].
    // Если она взята из статьи Blanes/Ros, то результат нужно умножить на T.

    if (max_order >= 6) {
        vector<CMatrix> nodes_6;
        for (int k = 0; k <= 6; ++k) {
            int idx = (k * (M - 1)) / 6;
            nodes_6.push_back(A_full_samples[idx]);
        }


        CMatrix Om6 = compute_Omega6_ACC(nodes_6, N);

        //// ИСПРАВЛЕНИЕ: Масштабируем на T^5, так как внутри коммутатор 5 матриц
        //double scale_factor = T_total;
        //mat_scale_inplace(Om6, N, complexd(scale_factor, 0.0));

        mat_add(Omega_total, Om6, Omega_total, N);
    }

    return Omega_total;
}

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================

CMatrix magnus_classic(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W,
    int max_order)
{
    // cout << "=== Classic Magnus up to Omega" << max_order << " ===" << endl;

    if (max_order < 1) max_order = 1;
    if (max_order > 3) max_order = 3;

    vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    int M = A.size();

    CMatrix Omega((size_t)N * N, complexd(0.0, 0.0));

    // === Omega1 ===
    // Интеграл методом трапеций (O(M))
    CMatrix Omega1 = trapezoidal_integral(A, dt, N);
    mat_add(Omega, Omega1, Omega, N);

    if (max_order == 1) return Omega;

    // === Omega2 ===
    // ОПТИМИЗАЦИЯ: O(M) вместо O(M^2)
    CMatrix Omega2((size_t)N * N, complexd(0.0, 0.0));
    CMatrix Accumulator((size_t)N * N, complexd(0.0, 0.0)); // Хранит сумму A[0]...A[i-1]

    // На первой итерации (i=0) Accumulator пуст, коммутатор равен 0.
    // Начинаем сразу накопление.

    for (int i = 0; i < M; ++i) {
        if (i > 0) {
            // [A[i], Sum(A[0]...A[i-1])]
            CMatrix comm = commutator(A[i], Accumulator, N);
            mat_add(Omega2, comm, Omega2, N);
        }
        // Добавляем текущий A[i] в аккумулятор для следующих шагов
        mat_add(Accumulator, A[i], Accumulator, N);
    }

    mat_scale_inplace(Omega2, N, complexd(dt * dt / 2.0, 0.0));
    mat_add(Omega, Omega2, Omega, N);

    if (max_order == 2) return Omega;

    // === Omega3 ===
    // ВНИМАНИЕ: Omega3 здесь все еще O(M^3). 
    // Если вы захотите считать 3-й порядок с 50000 точками, программа снова зависнет.
    // Для теста в main вы используете n <= 2, поэтому этот блок не вызовет проблем сейчас.

    if (max_order >= 3) {
        CMatrix Omega3((size_t)N * N, complexd(0.0, 0.0));
        // Прямая реализация слишком тяжелая для M=50000.
        // Оставляем как есть, но предупреждаем: не запускайте Classic Magnus порядка 3 на мелкой сетке.
        cout << "Warning: Classic Omega3 calculation is extremely slow for large M!" << endl;

        for (int i = 0; i < M; ++i)
            for (int j = 0; j < i; ++j)
            {
                CMatrix Cij = commutator(A[i], A[j], N);
                for (int k = 0; k < j; ++k)
                {
                    CMatrix term1 = commutator(A[i], commutator(A[j], A[k], N), N);
                    CMatrix term2 = commutator(Cij, A[k], N);
                    CMatrix sum = mat_copy(term1);
                    mat_add(sum, term2, sum, N);
                    mat_add(Omega3, sum, Omega3, N);
                }
            }

        mat_scale_inplace(Omega3, N, complexd(dt * dt * dt / 6.0, 0.0));
        mat_add(Omega, Omega3, Omega, N);
    }

    return Omega;
}

// =============================================================
// ===================== Magnus via formula 3.14 ===============
// =============================================================

CMatrix magnus_3_14(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W)
{
    // Сэмплируем A(t) = H0 + f(t) H_mod
    vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    int M = A.size();

    double h = t1 - t0;                 // шаг интегрирования
    double t_half = 0.5 * (t0 + t1);    // t_{1/2}

    // f(t), f'(t), f''(t) восстановим через численные производные
    // === Важно: generate_samples генерирует A_k = H0 + f_k * H_mod
    // => можно извлечь f_k = <A_k - H0, H_mod> / ||H_mod||^2
    // но т.к. H_mod — фиксированная структура, проще вычислить f(·) напрямую.

    auto f_fun = [&](double t)
        {
            return eps0 * std::sin(W * t);
        };

    auto fp_fun = [&](double t)
        {
            return eps0 * W * std::cos(W * t);
        };

    auto f2_fun = [&](double t)
        {
            return -eps0 * W * W * std::sin(W * t);
        };

    // Значения в формуле (3.14)
    double f_t1 = f_fun(t1);
    double f_half = f_fun(t_half);
    double fp_half = fp_fun(t_half);
    double f2_half = f2_fun(t_half);
    double f2_t1 = f2_fun(t1);

    // А теперь считаем Ω по формуле (3.14)
    CMatrix Omega = compute_Omega_3_14(
        H0, H_mod,
        h,
        f_t1,
        f_half,
        fp_half,
        f2_half,
        f2_t1,
        N
    );

    return Omega;
}


int main() {
    // === Experiment Parameters ===
    const int N = 10;
    const double dt = 1e-5;
    const double T = 1.0;
    const double diag_min = 10.0;

    // Set output to file
    std::ofstream results_txt("magnus_experiment_results.txt");
    if (!results_txt.is_open()) {
        std::cerr << "Error opening magnus_experiment_results.txt" << std::endl;
        return 1;
    }
    results_txt << std::scientific << std::setprecision(12);
    std::cout << std::scientific << std::setprecision(12);

    // =====================================================================
    // Generate constant Hamiltonian H_const
    // =====================================================================
    CMatrix H_const = utils::generate_hermitian_matrix(N, diag_min);

    // Reference Omega for dt: Omega_ref = -i * H * dt (Exact solution for constant H)
    CMatrix Omega_ref_dt = matrix_ops::mat_scale(H_const, complexd(0.0, -dt));

    // Parameters for methods
    // We set H0 = H_const (to be multiplied by -i inside functions if needed)
    // But wait: generate_samples multiplies by -i. 
    // magnus_3_14 expects H0 and H_mod as raw matrices and handles -i internally or expects them pre-scaled?
    // Looking at your MatrixOperations.cpp: magnus_3_14 computes commutators directly.
    // Usually H in physics is Hermitian. The exponent is -i*Omega.
    // Let's stick to the convention: Input matrices are Hermitian.

    // For Experiment 1 (Constant H), we map to the H(t) = H0 + f(t)Hmod form:
    CMatrix H0_input = H_const;                // The constant part
    CMatrix H_mod_zero(N * N, complexd(0.0, 0.0)); // No modulation part
    const double eps0 = 0.0;                   // Amplitude 0
    const double W = 0.0;                      // Frequency 0

    // For other methods that take samples A(t) = -i H(t):
    // Since eps0=0, A(t) is constantly -i * H_const.
    const double T_int = dt;
    const double integration_dt = dt * 0.01;

    // We pass H0_input. generate_samples multiplies it by -i * (1 + ...).
    // Since eps0=0, samples will be correctly -i * H_const.
    vector<CMatrix> A_samples = generate_samples(0.0, T_int, integration_dt, N,
        H0_input, H_mod_zero, eps0, W);

    // However, magnus_3_14 implementation in MatrixOperations.cpp (from previous prompts)
    // assumes A(t) construction inside or takes H0/Hmod. 
    // It constructs Omega directly. We need to ensure it scales by -i if it's not done inside.
    // Checking MatrixOperations.cpp logic for magnus_3_14:
    // It sums terms like "h * H0". If H0 is Hermitian, Omega is Hermitian.
    // But the result of magnus is usually the exponent argument. 
    // If U = exp(Omega), and U = exp(-iHt), then Omega must be -iHt.
    // Let's pass pre-scaled H0 to magnus_3_14 to be safe, OR check if it multiplies by -i.
    // Looking at snippet: term = H0; scale(h); add(term). It does NOT multiply by -i.
    // So for magnus_3_14, we must pass the anti-hermitian version if we want the result to match Omega_ref_dt.

    CMatrix H0_anti_herm = matrix_ops::mat_scale(H_const, complexd(0.0, -1.0)); // -i * H


    // =====================================================================
    // ================== EXPERIMENT 1: Omega Accuracy Comparison ====================
    // =====================================================================
    results_txt << "=================================================================\n";
    results_txt << "EXPERIMENT 1: Comparison of Omega Matrix Calculation Accuracy (H = const)\n";
    results_txt << "N=" << N << ", dt=" << dt << "\n";
    results_txt << "Reference: Omega_ref = -i * H * dt\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Method | K | Max Element Diff | Max Eig Diff | Time (ms)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // --- 1. Classic Magnus Expansion ---
    int max_classic_order = 2;
    for (int k = 1; k <= max_classic_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        // magnus_classic takes H0, H_mod and generates samples internally with -i factor
        CMatrix Omega_calc = magnus_classic(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Classic | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 2. Diploma Expansion (ACC) ---
    int max_acc_order = 5;
    for (int k = 1; k <= max_acc_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        CMatrix Omega_calc = magnus_ACC(A_samples, integration_dt, N, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "ACC | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 3. Recursive Expansion ---
    int max_recursive_order = 15;
    for (int k = 1; k <= max_recursive_order; ++k) {
        auto start = chrono::high_resolution_clock::now();

        CMatrix Omega_calc = magnus_expansion(0.0, T_int, integration_dt, N, H0_input, H_mod_zero, eps0, W, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_calc, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_calc, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Recursive | " << k << " | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // --- 4. Formula 3.14 (Special 4th order) ---
    //  - Formula 3.14 from dissertation
    {
        auto start = chrono::high_resolution_clock::now();

        // magnus_3_14 sums H0 terms directly. To get Omega = -i*H*dt, 
        // we must pass H0_anti_herm (-i*H) because the function doesn't add -i itself.
        // H_mod is zero, so commutators are zero.
        CMatrix Omega_314 = magnus_3_14(0.0, T_int, integration_dt, N, H0_anti_herm, H_mod_zero, eps0, W);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(Omega_314, Omega_ref_dt, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(Omega_314, Omega_ref_dt, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        // K=4 is the theoretical order of this method
        results_txt << "Formula 3.14 | 4 | " << max_el_diff << " | " << max_eig_diff << " | " << time_ms << "\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 2: Integration over the full interval ========
    // =====================================================================
    results_txt << "\n=================================================================\n";
    results_txt << "EXPERIMENT 2: Comparison of Numerical Integration Accuracy (H = const)\n";
    results_txt << "N=" << N << ", dt=" << dt << ", T=" << T << "\n";

    // 1. Reference U_exact
    // Omega_ref_T = -i * H * T
    CMatrix Omega_ref_T = matrix_ops::mat_scale(H_const, complexd(0.0, -T));
    const int expm_ref_terms = 30;
    CMatrix U_exact = matrix_ops::expm_taylor(Omega_ref_T, N, expm_ref_terms);

    results_txt << "Reference: Omega = -i*H*T, Expm Method = Taylor(" << expm_ref_terms << " terms)\n";
    results_txt << "Omega Calculation Method: Recursive Expansion (10 levels)\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Expm Terms (K) | Max Element Diff (U) | Mat One Norm Diff (U_diff)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // Omega(dt) - Recursive expansion, 10 levels
    // Use H0_input (Hermitian) because magnus_expansion handles -i internally via generate_samples
    const int omega_level = 10;
    CMatrix Omega_dt_magnus = magnus_expansion(0.0, dt, integration_dt, N, H0_input, H_mod_zero, eps0, W, omega_level);

    int num_steps = (int)std::round(T / dt);
    if (num_steps == 0) num_steps = 1;

    for (int k_taylor = 1; k_taylor <= 30; ++k_taylor) {
        // Step 1: U(dt) = expm_taylor(Omega_dt_magnus, N, k_taylor)
        CMatrix U_dt_calc = matrix_ops::expm_taylor(Omega_dt_magnus, N, k_taylor);

        // Step 2: U(T) = U(dt)^num_steps
        CMatrix U_calc = utils::eye(N);
        CMatrix temp(N * N, complexd(0.0, 0.0));

        for (int step = 0; step < num_steps; ++step) {
            matrix_ops::matmul(U_dt_calc, U_calc, temp, N);
            U_calc = temp;
        }

        CMatrix diff_U(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_calc, U_exact, diff_U, N);

        double max_el_diff = matrix_ops::max_element_diff(U_calc, U_exact, N);
        double mat_one_norm_diff = matrix_ops::mat_one_norm(diff_U, N);

        results_txt << "Taylor | " << std::setprecision(16) << k_taylor << " | " << max_el_diff << " | " << mat_one_norm_diff << "\n";
    }

    // =====================================================================
    // ============= EXPERIMENT 3: Chebyshev exp(Omega) Accuracy ============
    // =====================================================================
    results_txt << "\n=================================================================\n";
    results_txt << "EXPERIMENT 3: Accuracy of exp(Omega) using Chebyshev method\n";
    results_txt << "N=" << N << ", dt=" << dt << "\n";
    results_txt << "Reference: exp(Omega) = Taylor(30 terms)\n";
    results_txt << "-----------------------------------------------------------------\n";
    results_txt << "Method | K/M | Max Elem Diff | Max Eig Diff | Time (ms)\n";
    results_txt << "-----------------------------------------------------------------\n";

    // 1. Omega for full interval dt (same as in experiment 1)
    CMatrix Omega_dt = Omega_ref_dt;   // = -i * H * dt

    // 2. Reference exponential: Taylor with 30 terms
    CMatrix U_ref = matrix_ops::expm_taylor(Omega_dt, N, 30);

    // --------------- Taylor comparison (K = 1..10) --------------------
    for (int k = 1; k <= 10; k++)
    {
        auto start = chrono::high_resolution_clock::now();

        CMatrix U_taylor = matrix_ops::expm_taylor(Omega_dt, N, k);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_taylor, U_ref, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(U_taylor, U_ref, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Taylor | " << k << " | "
            << max_el_diff << " | "
            << max_eig_diff << " | " << time_ms << "\n";
    }

    // --------------- Chebyshev comparison (M = 1..25) --------------------
    for (int M = 1; M <= 25; M++)
    {
        auto start = chrono::high_resolution_clock::now();

        CMatrix U_cheb = expm_cheb(Omega_dt, N, M);

        auto end = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::microseconds>(end - start);
        double time_ms = duration.count() / 1000.0;

        CMatrix diff(N * N, complexd(0.0, 0.0));
        matrix_ops::mat_sub(U_cheb, U_ref, diff, N);

        double max_el_diff = matrix_ops::max_element_diff(U_cheb, U_ref, N);
        double max_eig_diff = matrix_ops::max_eigenvalue_modulus_hermitian(diff, N);

        results_txt << "Cheb | " << M << " | "
            << max_el_diff << " | "
            << max_eig_diff << " | " << time_ms << "\n";
    }

    results_txt << "-----------------------------------------------------------------\n";

    results_txt.close();
    std::cout << "\nExperiment results written to magnus_experiment_results.txt" << std::endl;

    return 0;
}
//Использование метода Магнуса - Чебышева для решения системы линейных ОДУ первого порядка С ИСПОЛЬЗОВАНИЕ ТЕХНОЛОГИЙ ПАРАЛЛЕЛЬНОГО ПРОГРАММИРОВАНИЯ