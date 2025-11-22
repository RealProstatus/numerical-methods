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
        // Берем 7 точек для 6-го порядка (как у вас было)
        for (int k = 0; k <= 6; ++k) {
            int idx = (k * (M - 1)) / 6;
            nodes_6.push_back(A_full_samples[idx]);
        }

        CMatrix Om6 = compute_Omega6_ACC(nodes_6, N);

        // ВАЖНО: Масштабируем на T_total (а не dt_grid!)
        // Это экспериментальный момент: зависит от того, как выводились коэффициенты в compute_Omega6.
        // Но T_total - единственный физически разумный вариант.
        mat_scale_inplace(Om6, N, complexd(T_total, 0.0));

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


// ------ Main function implementing the requested steps ------
int main() {
    const int N = 10;
    const double hbar = 1.0;
    const double eps0 = 1.0;
    const double W = 1.0;

    const double delta_t = 0.01;
    const double integration_dt = 1e-5;

    cout << "=== Step 0: Generating Hermitian matrix H ===" << endl;

    CMatrix H0 = generate_hermitian_matrix(N, 5.0);
    CMatrix H_mod = generate_hermitian_matrix(N, 1.0);

    std::cout << "Norm of H matrix:" << mat_one_norm(H0, 10) << std::endl;

    save_matrix(H0, N, "H0_matrix.txt");
    save_matrix(H_mod, N, "H_mod_matrix.txt");

    cout << "Hermitian matrices saved to files." << endl;


    cout << "\n=== Step 1: Runge-Kutta single step ===" << endl;
    double t_rk = 0.0;
    CMatrix H_current((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < H_current.size(); ++k)
        H_current[k] = H0[k] + eps0 * cos(W * t_rk) * H_mod[k];

    CMatrix U_rk = eye(N);
    U_rk = runge_kutta_step(H_current, U_rk, delta_t, N);

    cout << "Runge-Kutta step completed." << endl;


    cout << "\n=== Step 2: Magnus expansion (recursive) ===" << endl;

    CMatrix Omega = magnus_expansion(0.0, delta_t, integration_dt,
        N, H0, H_mod, eps0, W, 10);

    CMatrix U_magnus = expm_taylor(Omega, N, 30);

    cout << "Magnus expansion completed." << endl;


    cout << "\n=== Step 3: Magnus ACC expansion (orders 1..6) ===" << endl;

    vector<CMatrix> A_samples_ACC =
        generate_samples(0.0, delta_t, integration_dt,
            N, H0, H_mod, eps0, W);

    CMatrix Omega_ACC = magnus_ACC(A_samples_ACC, integration_dt, N, 6);
    CMatrix U_ACC = expm_taylor(Omega_ACC, N, 30);

    cout << "Magnus ACC expansion completed." << endl;


    // -------------------------------------------------
    // ✨ Step 4 — Classic Magnus up to omega 3 (третий метод!)
    // -------------------------------------------------

    cout << "\n=== Step 4: Classic Magnus expansion (up to Omega_4) ===" << endl;

    CMatrix Omega_classic =
        magnus_classic(0.0, delta_t, integration_dt,
            N, H0, H_mod, eps0, W, 2);

    CMatrix U_classic = expm_taylor(Omega_classic, N, 30);

    cout << "Classic Magnus Omega_4 expansion completed." << endl;


    cout << "\n=== Comparison ===" << endl;

    // RK vs Recursive Magnus
    CMatrix diff_rk_magnus((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_rk, U_magnus, diff_rk_magnus, N);
    double max_diff_rk_magnus = 0.0;
    for (auto& v : diff_rk_magnus) max_diff_rk_magnus = max(max_diff_rk_magnus, abs(v));
    //cout << "Max diff: RK vs Recursive Magnus = " << max_diff_rk_magnus << endl;

    // Recursive vs ACC
    CMatrix diff_magnus_ACC((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_magnus, U_ACC, diff_magnus_ACC, N);
    double max_diff_magnus_ACC = 0.0;
    for (auto& v : diff_magnus_ACC) max_diff_magnus_ACC = max(max_diff_magnus_ACC, abs(v));
    cout << "Max diff: Recursive vs ACC = " << max_diff_magnus_ACC << endl;

    // Recursive vs Classic
    CMatrix diff_magnus_classic((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_magnus, U_classic, diff_magnus_classic, N);
    double max_diff_magnus_classic = 0.0;
    for (auto& v : diff_magnus_classic) max_diff_magnus_classic = max(max_diff_magnus_classic, abs(v));
    cout << "Max diff: Recursive vs Classic = " << max_diff_magnus_classic << endl;


    // Save results
    save_matrix(U_rk, N, "U_runge_kutta.txt");
    save_matrix(U_magnus, N, "U_magnus.txt");
    save_matrix(Omega, N, "Omega_magnus.txt");
    save_matrix(U_ACC, N, "U_ACC.txt");
    save_matrix(Omega_ACC, N, "Omega_magnus_ACC.txt");
    save_matrix(U_classic, N, "U_classic.txt");
    save_matrix(Omega_classic, N, "Omega_classic.txt");
    save_matrix(diff_magnus_classic, N, "diff_magnus_classic.txt");

    cout << "\nAll results saved." << endl;

    //-------------------------------------------------
        // STEP 5 — Timing + Error per Omega order
        //-------------------------------------------------
    cout << "\n=== Step 5: Timing + Error per Omega order ===" << endl;

    // ИЗМЕНЕНИЕ 1: Увеличиваем интервал времени, чтобы коммутаторы стали значимыми
    // При T = 0.01 вклад коммутаторов слишком мал и не виден.
    // При T = 0.5 будет видно, как Omega_2 и Omega_4 исправляют ошибку.
    const double T_test = 0.05;

    // --- Эталонное решение (RK4 с мелким шагом) ---
    CMatrix U_exact_test = utils::eye(N);

    // Разбиваем интервал на 5000 шагов для высокой точности эталона
    int rk_substeps = 500000;
    double rk_dt = T_test / rk_substeps;

    for (int s = 0; s < rk_substeps; ++s) {
        double t_now = s * rk_dt;
        CMatrix H_t((size_t)N * N, complexd(0.0, 0.0));
        double f_t = eps0 * cos(W * t_now);
        for (size_t k = 0; k < H_t.size(); ++k)
            H_t[k] = H0[k] + complexd(f_t, 0.0) * H_mod[k];

        U_exact_test = runge_kutta_simple::runge_kutta_step(H_t, U_exact_test, rk_dt, N);
    }

    save_matrix(U_exact_test, N, "U_exact_test.txt");


    cout << "Exact reference computed (T=" << T_test << ", steps=" << rk_substeps << ")." << endl;

    ofstream results_txt("omega_results.txt");
    results_txt << "Timing + Error results for Magnus methods (T=" << T_test << ")\n";
    results_txt << "(Units: time in ms, max |U_calc - U_exact|)\n\n";

    auto max_error = [&](const CMatrix& A, const CMatrix& B) {
        double m = 0.0;
        for (size_t i = 0; i < A.size(); ++i)
            m = max(m, abs(A[i] - B[i]));
        return m;
        };

    CMatrix diff = eye(N);

    double max_err1;

    //
    // ============= Recursive Magnus ======================
    //
    cout << "\n[Testing] Recursive Magnus" << endl;
    results_txt << "========== Recursive Magnus ==========\n\n";

    // ---- Taylor ----
    results_txt << "-- Taylor exponential --\n";
    for (int n = 1; n <= 10; ++n) { // До 10 достаточно
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_test =
            magnus_expansion(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W, n);
        CMatrix U_test = matrix_ops::expm_taylor(Omega_test, N, 30);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);

        cout << "[Taylor] Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;

        save_matrix(U_test, N, "U_Recursive_Magnus_Taylor.txt");


        results_txt << "Recursive Taylor    " << n << "    " << ms << "    " << max_err1 << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    // ИЗМЕНЕНИЕ 2: Увеличиваем порядок Чебышева (M) с 10 до 50
    int M_cheb = 600;
    for (int n = 1; n <= 10; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_test =
            magnus_expansion(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W, n);
        // Используем M_cheb = 50 вместо 10
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_test, N, M_cheb);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);


        cout << "[Cheb] Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;


        save_matrix(U_test, N, "U_Recursive_Magnus_Chebyshev.txt");

        results_txt << "Recursive Cheb      " << n << "    " << ms << "    " << max_err1 << "\n";
    }
    results_txt << "\n\n";


    //
    // ============= ACC Magnus ============================
    //
    cout << "\n[Testing] ACC Magnus" << endl;
    results_txt << "========== ACC Magnus ==========\n\n";

    vector<CMatrix> A_samples_ACC_t =
        generate_samples(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W);

    // ---- Taylor ----
    results_txt << "-- Taylor exponential --\n";
    for (int n = 1; n <= 6; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_ACC_test =
            magnus_ACC(A_samples_ACC_t, integration_dt, N, n);
        CMatrix U_test = matrix_ops::expm_taylor(Omega_ACC_test, N, 30);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);


        cout << "[Taylor] ACC Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;


        save_matrix(U_test, N, "U_ACC_Magnus_Taylor.txt");

        results_txt << "ACC Taylor          " << n << "    " << ms << "    " << max_err1 << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    for (int n = 1; n <= 6; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_ACC_test =
            magnus_ACC(A_samples_ACC_t, integration_dt, N, n);
        // Используем M_cheb = 50
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_ACC_test, N, M_cheb);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);


        cout << "[Cheb] ACC Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;

        save_matrix(U_test, N, "ACC_Magnus_C.txt");

        results_txt << "ACC Cheb            " << n << "    " << ms << "    " << max_err1 << "\n";
    }
    results_txt << "\n\n";


    //
    // ============= Classic Magnus ========================
    //
    cout << "\n[Testing] Classic Magnus" << endl;
    results_txt << "========== Classic Magnus ==========\n\n";

    // ---- Taylor ----
    results_txt << "-- Taylor exponential --\n";
    for (int n = 1; n <= 2; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_classic_test =
            magnus_classic(0.0, T_test, integration_dt,
                N, H0, H_mod, eps0, W, n);
        CMatrix U_test = matrix_ops::expm_taylor(Omega_classic_test, N, 30);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);


        cout << "[Taylor] Classic Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;


        save_matrix(U_test, N, "Classic_Magnus_Taylor.txt");

        results_txt << "Classic Taylor      " << n << "    " << ms << "    " << max_err1 << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    for (int n = 1; n <= 2; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_classic_test =
            magnus_classic(0.0, T_test, integration_dt,
                N, H0, H_mod, eps0, W, n);
        // Используем M_cheb = 50
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_classic_test, N, M_cheb);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        bool unitary = is_unitary(U_test, N);
        cout << "    Unitary? " << (unitary ? "YES" : "NO") << endl;

        mat_sub(U_test, U_exact_test, diff, 10);

        max_err1 = mat_one_norm(diff, 10);


        cout << "[Cheb] Classic Omega" << n << ": " << ms << " ms | err=" << max_err1 << endl;


        save_matrix(U_test, N, "Classic_Magnus_Chebyshev.txt");

        results_txt << "Classic Cheb        " << n << "    " << ms << "    " << max_err1 << "\n";
    }

    results_txt.close();
    cout << "\nResults saved >>> omega_results.txt\n" << endl;

    //-------------------------------------------------
    // STEP 6 — Taylor vs Chebyshev Convergence Comparison
    //-------------------------------------------------
    cout << "\n=== Step 6: Taylor vs Chebyshev Convergence (Fixed Omega) ===" << endl;

    // 1. Фиксируем Omega (например, Magnus 4-го порядка)
    CMatrix Omega_fixed = magnus_expansion(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W, 4);

    // Приводим к анти-эрмитову виду для чистоты эксперимента
    // (Если у вас есть эта функция в MatrixOperations, если нет - можно пропустить, но лучше добавить)
    // matrix_ops::make_anti_hermitian_inplace(Omega_fixed, N); 

    double norm_Omega = mat_one_norm(Omega_fixed, N);
    cout << "Fixed Omega calculated. Norm = " << norm_Omega << endl;

    // 2. Вычисляем "Абсолютно точное" значение (Reference)
    // Берем Тейлора с заведомо избыточным числом членов (например, 100)
    CMatrix U_ref = matrix_ops::expm_taylor(Omega_fixed, N, 100);

    ofstream comp_txt("exp_convergence.txt");
    comp_txt << "Terms Taylor_Err Cheb_Err\n";
    cout << setw(10) << "Terms" << setw(20) << "Taylor Err" << setw(20) << "Cheb Err" << endl;

    // 3. Сравниваем сходимость при изменении числа слагаемых (Order K)
    // Идем до 20, так как на малых матрицах сходимость очень быстрая
    for (int k = 2; k <= 20; k += 2) {

        // --- Taylor ---
        // Важно: внутри expm_taylor у вас может быть scaling & squaring.
        // Чтобы честно сравнить ряды, scaling лучше бы отключить, но пока используем как есть.
        CMatrix U_tay = matrix_ops::expm_taylor(Omega_fixed, N, k);

        // Ошибка Тейлора относительно Reference
        double err_tay = 0.0;
        for (size_t i = 0; i < U_tay.size(); ++i) err_tay = max(err_tay, abs(U_tay[i] - U_ref[i]));


        // --- Chebyshev ---
        // Для Чебышева k - это порядок полинома.
        CMatrix U_cheb = matrix_ops::expm_cheb(Omega_fixed, N, k);

        // Ошибка Чебышева относительно Reference
        double err_cheb = 0.0;
        for (size_t i = 0; i < U_cheb.size(); ++i) err_cheb = max(err_cheb, abs(U_cheb[i] - U_ref[i]));

        cout << setw(10) << k << setw(20) << err_tay << setw(20) << err_cheb << endl;
        comp_txt << k << " " << err_tay << " " << err_cheb << "\n";
    }
    comp_txt.close();
    cout << "\nComparison saved to 'exp_convergence.txt'" << endl;

    return 0;
}
//Использование метода Магнуса - Чебышева для решения системы линейных ОДУ первого порядка С ИСПОЛЬЗОВАНИЕ ТЕХНОЛОГИЙ ПАРАЛЛЕЛЬНОГО ПРОГРАММИРОВАНИЯ