#include <mkl.h>
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>
#include <fstream>

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

// ------ Magnus expansion up to Omega_10 ------
CMatrix magnus_expansion(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int max_n = 10) {
    vector<CMatrix> Omega_n(max_n + 1);

    // Генерируем выборки A(t)
    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);

    // Ω₁(t) = ∫₀ᵗ A(τ)dτ
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

CMatrix magnus_ACC_up_to_6(
    const vector<CMatrix>& A_samples,
    double dt,
    int N)
{
    vector<CMatrix> Omega(7); // Omega[1..6]

    // === Ω1 ===
    Omega[1] = trapezoidal_integral(A_samples, dt, N);

    // === Ω2..Ω4 ===
    for (int n = 2; n <= 4; ++n)
    {
        CMatrix sum(N * N, complexd(0, 0));

        // Генерируем все перестановки множества {2,...,n}
        vector<vector<int>> perms;
        vector<int> base(n - 1);
        for (int i = 0; i < n - 1; ++i) base[i] = i + 1; // 1..n-1 (0-based для A_samples)
        generate_permutations(base, perms);

        for (auto& p : perms)
        {
            // Вставляем t1 в конец
            p.push_back(0); // t1 — первый элемент A_samples

            // Вычисляем правонестед-коммутатор через новую функцию
            CMatrix comm = right_nested_comm(A_samples, p, N);

            // Интегрируем по времени
            vector<CMatrix> comm_samples = { comm };
            CMatrix integ = trapezoidal_integral(comm_samples, dt, N);

            // Коэффициент cp (упрощённо)
            double cp = 1.0 / n;
            mat_scale_inplace(integ, N, complexd(cp, 0.0));

            mat_add(sum, integ, sum, N);
        }

        Omega[n] = sum;
    }

    // === Ω5 и Ω6 ===
    CMatrix O5 = compute_Omega5_ACC(A_samples, N);
    CMatrix O6 = compute_Omega6_ACC(A_samples, N);

    // === Суммируем все члены ===
    CMatrix Omega_total(N * N, complexd(0, 0));
    for (int i = 1; i <= 4; ++i)
        mat_add(Omega_total, Omega[i], Omega_total, N);
    mat_add(Omega_total, O5, Omega_total, N);
    mat_add(Omega_total, O6, Omega_total, N);

    return Omega_total;
}

// Главная функция для метода Магнуса–Чебышева
CMatrix magnus_chebyshev(double t0, double t1, double integration_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int M = 20)
{
    using namespace std;
    cout << "=== Magnus-Chebyshev method ===" << endl;

    vector<CMatrix> A_samples = generate_samples(t0, t1, integration_dt, N, H0, H_mod, eps0, W);
    cout << "Computing Ω₁ using trapezoidal rule..." << endl;

    CMatrix Omega = trapezoidal_integral(A_samples, integration_dt, N);

    cout << "Computing exp(Ω) using Chebyshev expansion (order M=" << M << ")..." << endl;
    CMatrix U_cheb = expm_chebyshev(Omega, N, M);

    cout << "Magnus-Chebyshev completed." << endl;
    return U_cheb;
}

// ------ Main function implementing the requested steps ------
int main() {
    const int N = 10;
    const double hbar = 1.0;
    const double eps0 = 1.0;
    const double W = 1.0;
    const double delta_t = 1e-5;
    const double integration_dt = 1e-8;

    cout << "=== Step 0: Generating Hermitian matrix H ===" << endl;

    // Generate Hermitian Hamiltonian H0
    CMatrix H0 = generate_hermitian_matrix(N, 5.0);

    // Generate modulation matrix H_mod (also Hermitian)
    CMatrix H_mod = generate_hermitian_matrix(N, 1.0);

    // Save matrices to shared resource (files)
    save_matrix(H0, N, "H0_matrix.txt");
    save_matrix(H_mod, N, "H_mod_matrix.txt");

    cout << "Hermitian matrices saved to H0_matrix.txt and H_mod_matrix.txt" << endl;

    cout << "\n=== Step 1: Runge-Kutta single step ===" << endl;

    // Create time-dependent Hamiltonian H(t) = H0 + eps0*cos(W*t)*H_mod
    double t_rk = 0.0;
    CMatrix H_current((size_t)N * N, complexd(0.0, 0.0));
    for (size_t k = 0; k < H_current.size(); ++k) {
        H_current[k] = H0[k] + eps0 * cos(W * t_rk) * H_mod[k];
    }

    // Initial state: identity matrix
    CMatrix U_rk = eye(N);

    // Perform one Runge-Kutta step using the provided function
    U_rk = runge_kutta_step(H_current, U_rk, delta_t, N);

    cout << "Runge-Kutta step completed. U(dt) computed." << endl;

    cout << "\n=== Step 2: Magnus expansion ===" << endl;

    cout << "2.1 Computing Magnus expansion up to Omega_10..." << endl;
    CMatrix Omega = magnus_expansion(0.0, delta_t, integration_dt, N, H0, H_mod, eps0, W, 10);

    cout << "2.2 Omega(0, " << delta_t << ") computed" << endl;

    cout << "2.3 Computing U(0," << delta_t << ") = exp(i*Omega) via Taylor series..." << endl;

    // Используем Omega напрямую без дополнительного i
    CMatrix U_magnus = expm_taylor(Omega, N, 30);

    cout << "Magnus expansion completed." << endl;

    cout << "\n=== Step 3: Magnus ACC expansion (orders 1..6) ===" << endl;

    cout << "3.1 Generating A(t) samples for ACC..." << endl;
    vector<CMatrix> A_samples_ACC = generate_samples(0.0, delta_t, integration_dt, N, H0, H_mod, eps0, W);

    cout << "3.2 Computing Magnus ACC expansion up to order 6..." << endl;
    CMatrix Omega_ACC = magnus_ACC_up_to_6(A_samples_ACC, integration_dt, N);

    cout << "3.3 Omega_ACC(0, " << delta_t << ") computed" << endl;

    cout << "3.4 Computing U_ACC(0," << delta_t << ") = exp(Omega_ACC) via Taylor series..." << endl;
    CMatrix U_ACC = expm_taylor(Omega_ACC, N, 30);

    cout << "Magnus ACC expansion completed." << endl;

    cout << "\n=== Step 4: Magnus–Chebyshev method ===" << endl;

    CMatrix U_cheb = magnus_chebyshev(0.0, delta_t, integration_dt, N, H0, H_mod, eps0, W, 300);

    cout << "U_Magnus_Chebyshev.txt saved." << endl;

    // Compare results
    cout << "\n=== Comparison ===" << endl;

    // 1️. Runge-Kutta vs Classic Magnus
    CMatrix diff_rk_magnus((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_rk, U_magnus, diff_rk_magnus, N);

    double max_diff_rk_magnus = 0.0;
    for (size_t k = 0; k < diff_rk_magnus.size(); ++k)
        max_diff_rk_magnus = max(max_diff_rk_magnus, abs(diff_rk_magnus[k]));

    cout << "Maximum difference between RK and Classic Magnus: " << max_diff_rk_magnus << endl;

    // 2️. Classic Magnus vs Magnus ACC
    CMatrix diff_magnus_ACC((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_magnus, U_ACC, diff_magnus_ACC, N);

    double max_diff_magnus_ACC = 0.0;
    for (size_t k = 0; k < diff_magnus_ACC.size(); ++k)
        max_diff_magnus_ACC = max(max_diff_magnus_ACC, abs(diff_magnus_ACC[k]));

    cout << "Maximum difference between Classic Magnus and Magnus ACC: " << max_diff_magnus_ACC << endl;

    // 3️. Runge-Kutta vs Magnus–Chebyshev
    CMatrix diff_rk_cheb((size_t)N * N, complexd(0.0, 0.0));
    mat_sub(U_rk, U_cheb, diff_rk_cheb, N);

    double max_diff_rk_cheb = 0.0;
    for (size_t k = 0; k < diff_rk_cheb.size(); ++k)
        max_diff_rk_cheb = max(max_diff_rk_cheb, abs(diff_rk_cheb[k]));

    cout << "Maximum difference between RK and Magnus-Chebyshev: " << max_diff_rk_cheb << endl;


    // Save results
    save_matrix(U_rk, N, "U_runge_kutta.txt");
    save_matrix(U_magnus, N, "U_magnus.txt");
    save_matrix(Omega, N, "Omega_magnus.txt");
    save_matrix(U_ACC, N, "U_ACC.txt");
    save_matrix(Omega_ACC, N, "Omega_magnus_ACC.txt");
    save_matrix(diff_magnus_ACC, N, "diff_magnus_ACC.txt");
    save_matrix(U_cheb, N, "U_Magnus_Chebyshev.txt");

    cout << "\nResults saved to files:" << endl;
    cout << "- U_runge_kutta.txt: Runge-Kutta result" << endl;
    cout << "- U_magnus.txt: Magnus expansion result" << endl;
    cout << "- Omega_magnus.txt: Magnus operator Omega" << endl;
    cout << "- U_magnus_ACC.txt: Magnus ACC method expansion result" << endl;
    cout << "- Omega_magnus_ACC.txt: Magnus ACC method operator Omega" << endl;
    cout << "- diff_magnus_ACC.txt: Difference between classic Magnus method and Magnus ACC method expansions" << endl;

    return 0;
}
//Использование метода Магнуса - Чебышева для решения системы линейных ОДУ первого порядка С ИСПОЛЬЗОВАНИЕ ТЕХНОЛОГИЙ ПАРАЛЛЕЛЬНОГО ПРОГРАММИРОВАНИЯ