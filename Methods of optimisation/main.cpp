#include <mkl.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>
#include <chrono>

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


// ================================================================
// =================== Recursive Magnus up to 15 ==================
// ================================================================

CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n = 10) {
    vector<CMatrix> Omega_n(max_n + 1);

    // Генерируем выборки A(t)
    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);

    // Omega₁(t) = ∫₀ᵗ A(τ)dτ
    Omega_n[1] = trapezoidal_integral(A_samples, integration_dt, N);

    cout << "Computing Magnus expansion up to order " << max_n << "..." << endl;

    // Вычисляем члены высшего порядка согласно Wikipedia
    for (int n = 2; n <= max_n; ++n) {
        cout << "Computing Omega_" << n << "..." << endl;

        CMatrix sum_term((size_t)N * N, complexd(0.0, 0.0));

        for (int j = 1; j <= n - 1; ++j) {
            double B_j = matrix_ops::bernoulli_number(j);

            // Пропускаем нулевые числа Бернулли
            if (std::abs(B_j) < 1e-12) continue;

            // Вычисляем S_n^(j) согласно рекуррентной формуле
            vector<CMatrix> S_samples = matrix_ops::compute_S_n_j(n, j, Omega_n, A_samples, N);

            // ∫₀ᵗ S_n^(j)(τ)dτ
            CMatrix integral_term = trapezoidal_integral(S_samples, integration_dt, N);

            // B_j/j! * ∫₀ᵗ S_n^(j)(τ)dτ
            double factorial = 1.0;
            for (int k = 1; k <= j; ++k) factorial *= k;

            mat_scale_inplace(integral_term, N, complexd(B_j / factorial, 0.0));
            mat_add(sum_term, integral_term, sum_term, N);
        }

        Omega_n[n] = sum_term;
    }

    // Суммируем все члены разложения
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));
    for (int n = 1; n <= max_n; ++n) {
        cout << "||Omega_" << n << "|| = " << mat_one_norm(Omega_n[n], N) << endl;
        mat_add(Omega_total, Omega_n[n], Omega_total, N);
    }

    return Omega_total;
}

// =========================================================
// =================== Magnus ACC up to 6 ==================
// =========================================================

CMatrix magnus_ACC(
    const vector<CMatrix>& A_samples,
    double dt,
    int N,
    int max_order)   // ← новый параметр!
{
    if (max_order < 1) max_order = 1;
    if (max_order > 6) max_order = 6;

    vector<CMatrix> Omega(7); // [1..6]

    // === Omega1 ===
    Omega[1] = trapezoidal_integral(A_samples, dt, N);

    if (max_order == 1)
        return Omega[1];

    // === Omega2..Omega4 ===
    for (int n = 2; n <= min(4, max_order); ++n)
    {
        CMatrix sum(N * N, complexd(0, 0));

        vector<vector<int>> perms;
        vector<int> base(n - 1);
        for (int i = 0; i < n - 1; ++i) base[i] = i + 1;
        generate_permutations(base, perms);

        for (auto& p : perms)
        {
            p.push_back(0);
            CMatrix comm = right_nested_comm(A_samples, p, N);

            vector<CMatrix> comm_samples = { comm };
            CMatrix integ = trapezoidal_integral(comm_samples, dt, N);

            double cp = 1.0 / n;
            mat_scale_inplace(integ, N, complexd(cp, 0.0));
            mat_add(sum, integ, sum, N);
        }

        Omega[n] = sum;
    }

    if (max_order >= 5) {
        CMatrix Om5 = compute_Omega5_ACC(A_samples, N);
        // Скорее всего, здесь не хватает множителя dt
        mat_scale_inplace(Om5, N, complexd(dt, 0.0));
        Omega[5] = Om5;
    }

    if (max_order >= 6) {
        CMatrix Om6 = compute_Omega6_ACC(A_samples, N);
        // Исправление масштаба
        mat_scale_inplace(Om6, N, complexd(dt, 0.0));
        Omega[6] = Om6;
    }


    // === Суммируем Omega₁..Omegaₘₐₓ ===
    CMatrix Omega_total(N * N, complexd(0, 0));
    for (int i = 1; i <= max_order; ++i)
        mat_add(Omega_total, Omega[i], Omega_total, N);

    return Omega_total;
}

// =============================================================
// =================== Classic Magnus up to 3 ==================
// =============================================================

CMatrix magnus_classic(
    double t0, double t1, double dt,
    int N, const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W,
    int max_order)   // ← новый параметр!
{
    cout << "=== Classic Magnus up to Omega" << max_order << " ===" << endl;

    if (max_order < 1) max_order = 1;
    if (max_order > 3) max_order = 3; // max allowed

    vector<CMatrix> A = generate_samples(t0, t1, dt, N, H0, H_mod, eps0, W);
    int M = A.size();

    CMatrix Omega((size_t)N * N, complexd(0.0, 0.0));

    // === Omega1 ===
    CMatrix Omega1 = trapezoidal_integral(A, dt, N);
    mat_add(Omega, Omega1, Omega, N);

    if (max_order == 1) return Omega;

    // === Omega2 ===
    CMatrix Omega2((size_t)N * N, complexd(0.0, 0.0));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < i; ++j)
            mat_add(Omega2, commutator(A[i], A[j], N), Omega2, N);

    mat_scale_inplace(Omega2, N, complexd(dt * dt / 2.0, 0.0));
    mat_add(Omega, Omega2, Omega, N);

    if (max_order == 2) return Omega;

    // === Omega3 ===
    CMatrix Omega3((size_t)N * N, complexd(0.0, 0.0));
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < i; ++j)
        {
            CMatrix Cij = commutator(A[i], A[j], N);
            for (int k = 0; k < j; ++k)
            {
                CMatrix term1 = commutator(A[i],
                    commutator(A[j], A[k], N), N);
                CMatrix term2 = commutator(Cij, A[k], N);

                CMatrix sum = mat_copy(term1);
                mat_add(sum, term2, sum, N);
                mat_add(Omega3, sum, Omega3, N);
            }
        }

    mat_scale_inplace(Omega3, N, complexd(dt * dt * dt / 6.0, 0.0));
    mat_add(Omega, Omega3, Omega, N);

    return Omega;
}

//// ---- new: magnus_expansion_from_samples ----
//// A_samples: vector<CMatrix> — значения A(t) используемые в Magnus (т.е. уже с фактором -i, если нужно)
//// dt: time spacing between samples
//CMatrix magnus_expansion_from_samples(const vector<CMatrix>& A_samples, double dt, int N, int max_n = 10)
//{
//    int maxN = max_n;
//    vector<CMatrix> Omega_n(maxN + 1);
//    // Omega1
//    Omega_n[1] = trapezoidal_integral(A_samples, dt, N);
//
//    for (int n = 2; n <= maxN; ++n) {
//        CMatrix sum_term((size_t)N * N, complexd(0.0, 0.0));
//        for (int j = 1; j <= n - 1; ++j) {
//            double B_j = matrix_ops::bernoulli_number(j);
//            if (std::abs(B_j) < 1e-12) continue;
//            // compute S_n^j using same compute_S_n_j (it expects Omega vector and A_samples)
//            vector<CMatrix> S_samples = matrix_ops::compute_S_n_j(n, j, Omega_n, A_samples, N);
//            CMatrix integral_term = trapezoidal_integral(S_samples, dt, N);
//            double factorial = 1.0;
//            for (int k = 1; k <= j; ++k) factorial *= k;
//            mat_scale_inplace(integral_term, N, complexd(B_j / factorial, 0.0));
//            mat_add(sum_term, integral_term, sum_term, N);
//        }
//        Omega_n[n] = sum_term;
//    }
//
//    // sum Omega_n
//    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));
//    for (int n = 1; n <= maxN; ++n) mat_add(Omega_total, Omega_n[n], Omega_total, N);
//    return Omega_total;
//}
//
//// ---- new: magnus_classic_from_samples ----
//// compute classic Magnus terms using already-built samples A(t)
//// (this mimics your magnus_classic, but does not regenerate samples)
//CMatrix magnus_classic_from_samples(const vector<CMatrix>& A_samples, double dt, int N, int max_order)
//{
//    if (max_order < 1) max_order = 1;
//    if (max_order > 3) max_order = 3;
//
//    int M = (int)A_samples.size();
//    CMatrix Omega((size_t)N * N, complexd(0.0, 0.0));
//
//    // Omega1
//    CMatrix Omega1 = trapezoidal_integral(A_samples, dt, N);
//    mat_add(Omega, Omega1, Omega, N);
//    if (max_order == 1) return Omega;
//
//    // Omega2
//    CMatrix Omega2((size_t)N * N, complexd(0.0, 0.0));
//    for (int i = 0; i < M; ++i)
//        for (int j = 0; j < i; ++j)
//            mat_add(Omega2, commutator(A_samples[i], A_samples[j], N), Omega2, N);
//    mat_scale_inplace(Omega2, N, complexd(dt * dt / 2.0, 0.0));
//    mat_add(Omega, Omega2, Omega, N);
//    if (max_order == 2) return Omega;
//
//    // Omega3
//    CMatrix Omega3((size_t)N * N, complexd(0.0, 0.0));
//    for (int i = 0; i < M; ++i)
//        for (int j = 0; j < i; ++j)
//            for (int k = 0; k < j; ++k) {
//                CMatrix term1 = commutator(A_samples[i], commutator(A_samples[j], A_samples[k], N), N);
//                CMatrix term2 = commutator(commutator(A_samples[i], A_samples[j], N), A_samples[k], N);
//                CMatrix sum = mat_copy(term1);
//                mat_add(sum, term2, sum, N);
//                mat_add(Omega3, sum, Omega3, N);
//            }
//    mat_scale_inplace(Omega3, N, complexd(dt * dt * dt / 6.0, 0.0));
//    mat_add(Omega, Omega3, Omega, N);
//    return Omega;
//}


//void compare_all_methods(const CMatrix& A_phys, const CMatrix& B_phys, double h, int N)
//{
//    using namespace std;
//    cout << "\n==========================\n  Comparing Omega methods\n==========================\n";
//
//    // Build Magnus A(t) = -i*(A_phys + t * B_phys)
//    const double dt = 1e-5;                    // как у тебя
//    int nsteps = int(h / dt) + 1;
//    vector<CMatrix> samples(nsteps);
//    complexd minus_i(0.0, -1.0);
//
//    // build base matrices for Magnus (with -i factor)
//    CMatrix A_mag = A_phys;
//    CMatrix B_mag = B_phys;
//    for (int k = 0; k < N * N; ++k) {
//        A_mag[k] *= minus_i;   // -i * A_phys
//        B_mag[k] *= minus_i;   // -i * B_phys
//    }
//
//    for (int s = 0; s < nsteps; ++s) {
//        double t = s * dt;
//        CMatrix At((size_t)N * N);
//        for (int k = 0; k < N * N; ++k)
//            At[k] = A_mag[k] + complexd(t, 0.0) * B_mag[k];
//        samples[s] = At;
//    }
//
//    // analytic Omega for Magnus-representation:
//    // Omega_exact = integral_0^h (A_mag + t B_mag) dt = h*A_mag + 0.5*h^2 * B_mag
//    CMatrix Omega_exact = analytic_omega(h, A_mag, B_mag, N); // analytic_omega already does h*A + 0.5*h^2*B
//
//    auto max_err = [&](const CMatrix& X) {
//        double m = 0.0;
//        for (size_t i = 0; i < X.size(); ++i) m = max(m, abs(X[i] - Omega_exact[i]));
//        return m;
//        };
//
//    cout.setf(std::ios::scientific);
//    cout.precision(6);
//
//    // 1) Recursive Magnus from samples (1..10)
//    for (int n = 1; n <= 15; ++n) {
//        // compute via expansion built from samples
//        CMatrix Om = magnus_expansion_from_samples(samples, dt, N, n);
//        double err = max_err(Om);
//        cout << "Recursive Magnus Omega" << n << "  err = " << err << "\n";
//    }
//
//    // 2) ACC Magnus (uses samples directly)
//    for (int n = 1; n <= 6; ++n) {
//        CMatrix Om = magnus_ACC(samples, dt, N, n);
//        double err = max_err(Om);
//        cout << "ACC Magnus Omega" << n << "  err = " << err << "\n";
//    }
//
//    // 3) Classic Magnus from samples
//    for (int n = 1; n <= 2; ++n) {
//        CMatrix Om = magnus_classic_from_samples(samples, dt, N, n);
//        double err = max_err(Om);
//        cout << "Classic Magnus Omega" << n << "  err = " << err << "\n";
//    }
//
//    cout << "========================================\n\n";
//}


//int main() {
//    const int N = 10;
//    const double h = 0.01;
//
//    double scaleA = 1.0;
//    double scaleB = 1.0;
//    double noncomm_factor = 6.0; // усиливаем off-diag у B
//    double diag_offset = 50.0;
//
//    auto tmp = utils::generate_hermitian_pair_dd(N, scaleA, scaleB, noncomm_factor, diag_offset);
//    CMatrix A = tmp.first;
//    CMatrix B = tmp.second;
//
//    cout << "Running comparison...\n";
//    compare_all_methods(A, B, h, N);
//
//    return 0;
//}



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
    cout << "Max diff: RK vs Recursive Magnus = " << max_diff_rk_magnus << endl;

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

    const double T_test = delta_t;

    // --- ИСПРАВЛЕНИЕ ЭТАЛОНА ---
    // Вместо аналитического H0, используем численное решение Рунге-Кутты 
    // с очень мелким шагом, чтобы считать его "точным".

    CMatrix U_exact_test = utils::eye(N); // Начальное условие I

    // Используем существующую функцию runge_kutta_step, 
    // но разбиваем интервал T_test на много мелких шагов.
    int rk_substeps = 1000;
    double rk_dt = T_test / rk_substeps;

    for (int s = 0; s < rk_substeps; ++s) {
        double t_now = s * rk_dt;
        // Вычисляем полный Гамильтониан в точке t
        CMatrix H_t((size_t)N * N, complexd(0.0, 0.0));
        double f_t = eps0 * cos(W * t_now);

        for (size_t k = 0; k < H_t.size(); ++k)
            H_t[k] = H0[k] + complexd(f_t, 0.0) * H_mod[k];

        // Делаем шаг RK4
        U_exact_test = runge_kutta_simple::runge_kutta_step(H_t, U_exact_test, rk_dt, N);
    }

    cout << "Exact reference computed using fine-grained RK4." << endl;

    // Файл результата TXT
    ofstream results_txt("omega_results.txt");
    results_txt << "Timing + Error results for Magnus methods\n";
    results_txt << "(Units: time in ms, max |U_calc - U_exact|)\n\n";

    // Функция ошибки
    auto max_error = [&](const CMatrix& A, const CMatrix& B) {
        double m = 0.0;
        for (size_t i = 0; i < A.size(); ++i)
            m = max(m, abs(A[i] - B[i]));
        return m;
        };

    //
    // =====================================================
    // ============= Recursive Magnus ======================
    // =====================================================
    cout << "\n[Testing] Recursive Magnus" << endl;

    results_txt << "========== Recursive Magnus ==========\n\n";

    // ---- Taylor ----
    results_txt << "-- Taylor exponential --\n";
    for (int n = 1; n <= 15; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_test =
            magnus_expansion(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W, n);
        CMatrix U_test = matrix_ops::expm_taylor(Omega_test, N, 30);

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        cout << "[Taylor] Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "Recursive Taylor   " << n << "   "
            << ms << "   " << err << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    for (int n = 1; n <= 15; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_test =
            magnus_expansion(0.0, T_test, integration_dt, N, H0, H_mod, eps0, W, n);
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_test, N, 10);   

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        cout << "[Cheb] Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "Recursive Cheb     " << n << "   "
            << ms << "   " << err << "\n";
    }
    results_txt << "\n\n";


    //
    // =====================================================
    // ============= ACC Magnus ============================
    // =====================================================
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

        cout << "[Taylor] ACC Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "ACC Taylor        " << n << "   "
            << ms << "   " << err << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    for (int n = 1; n <= 6; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_ACC_test =
            magnus_ACC(A_samples_ACC_t, integration_dt, N, n);
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_ACC_test, N, 10);  

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        cout << "[Cheb] ACC Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "ACC Cheb          " << n << "   "
            << ms << "   " << err << "\n";
    }
    results_txt << "\n\n";


    //
    // =====================================================
    // ============= Classic Magnus ========================
    // =====================================================
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

        cout << "[Taylor] Classic Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "Classic Taylor    " << n << "   "
            << ms << "   " << err << "\n";
    }
    results_txt << "\n";

    // ---- Chebyshev ----
    results_txt << "-- Chebyshev exponential --\n";
    for (int n = 1; n <= 2; ++n) {
        auto t1 = chrono::high_resolution_clock::now();

        CMatrix Omega_classic_test =
            magnus_classic(0.0, T_test, integration_dt,
                N, H0, H_mod, eps0, W, n);
        CMatrix U_test =
            matrix_ops::expm_cheb(Omega_classic_test, N, 10); // ★ исправлено

        auto t2 = chrono::high_resolution_clock::now();
        double ms = chrono::duration<double, std::milli>(t2 - t1).count();
        double err = max_error(U_test, U_exact_test);

        cout << "[Cheb] Classic Omega" << n << ": " << ms << " ms | err=" << err << endl;
        results_txt << "Classic Cheb      " << n << "   " << ms << "   " << err << "\n";
    }

    results_txt.close();
    cout << "\nResults saved >>> omega_results.txt\n" << endl;

    return 0;
}
//Использование метода Магнуса - Чебышева для решения системы линейных ОДУ первого порядка С ИСПОЛЬЗОВАНИЕ ТЕХНОЛОГИЙ ПАРАЛЛЕЛЬНОГО ПРОГРАММИРОВАНИЯ