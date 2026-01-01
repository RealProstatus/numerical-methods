#pragma once

#include "Utils.h"
#include <map>

using namespace std;
using complexd = complex<double>;
using CMatrix = vector<complexd>;

namespace matrix_ops {
    void matmul(const CMatrix& A, const CMatrix& B, CMatrix& C, int N,
        complexd alpha = complexd(1.0, 0.0), complexd beta = complexd(0.0, 0.0));

    double mat_one_norm(const CMatrix& A, int N);
    void mat_add(const CMatrix& A, const CMatrix& B, CMatrix& C, int N);
    void mat_sub(const CMatrix& A, const CMatrix& B, CMatrix& C, int N);
    void mat_scale_inplace(CMatrix& A, int N, complexd alpha);
    CMatrix mat_copy(const CMatrix& A);
    CMatrix mat_scale(const CMatrix& A, complexd alpha);

    // ВЫЧИСЛЕНИЕ КОММУТАТОРА: [A, B] = A*B - B*A
    // A, B - входные матрицы, N - размер
    // Возвращает: матрицу-коммутатор
    CMatrix commutator(const CMatrix& A, const CMatrix& B, int N);

    // ИТЕРАЦИОННЫЙ КОММУТАТОР: ad_X^k(Y) = [X, [X, ...[X, Y]...]] (k раз)
    // X, Y - входные матрицы, k - количество итераций, N - размер
    // Возвращает: результат k-кратного коммутирования
    CMatrix iterated_commutator(const CMatrix& X, const CMatrix& Y, int k, int N);

    // РАЗЛОЖЕНИЕ ТЕЙЛОРА: Вычисляет exp(A) ≈ I + A + A² / 2!+ A³ / 3!+ ... + Aⁿ / n!


    // Новые функции для разложения Магнуса
    double bernoulli_number(int j);

    // ИТЕРАЦИОННЫЙ КОММУТАТОР ДЛЯ МАГНУСА: ad_?^k(A) = [?, [?, ...[?, A]...]]
    // Omega - оператор, A - матрица, k - степень коммутирования, N - размер
    // Возвращает: результат применения ad-оператора k раз
    CMatrix compute_ad_Omega_k(const CMatrix& Omega, const CMatrix& A, int k, int N);

    // ВЫЧИСЛЕНИЕ S_n^(j) ДЛЯ РАЗЛОЖЕНИЯ МАГНУСА (рекурсивная формула)
    // n - порядок члена разложения, j - индекс в рекурсии
    // Omega - вектор вычисленных ??...??, A_samples - выборки A(t) во времени
    // N - размер матриц
    // Возвращает: вектор матриц S_n^(j) для всех моментов времени
    vector<CMatrix> compute_S_n_j(int n, int j,
        const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples,
        int N,
        std::map<std::pair<int, int>, vector<CMatrix>>& cache);

    CMatrix matrix_exp_special(const CMatrix& A, int N, double dt);

    CMatrix dagger(const CMatrix& A, int N);
    bool is_unitary(const CMatrix& U, int N, double tol = 1e-10);
    double max_element_diff(const CMatrix& A, const CMatrix& B, int N);
    double max_eigenvalue_modulus_hermitian(const CMatrix& M, int N);

    // Wrappers of library methods of e^A
}