#pragma once
#include <complex>
#include <vector>
#include "Utils.h"

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

    // ВЫЧИСЛЕНИЕ КОММУТАТОРА: [A, B] = A*B - B*A
    // A, B - входные матрицы, N - размер
    // Возвращает: матрицу-коммутатор
    CMatrix commutator(const CMatrix& A, const CMatrix& B, int N);

    // ИТЕРАЦИОННЫЙ КОММУТАТОР: ad_X^k(Y) = [X, [X, ...[X, Y]...]] (k раз)
    // X, Y - входные матрицы, k - количество итераций, N - размер
    // Возвращает: результат k-кратного коммутирования
    CMatrix iterated_commutator(const CMatrix& X, const CMatrix& Y, int k, int N);

    // РАЗЛОЖЕНИЕ ТЕЙЛОРА: Вычисляет exp(A) ≈ I + A + A² / 2!+ A³ / 3!+ ... + Aⁿ / n!
    CMatrix expm_taylor(const CMatrix& A, int N, int terms = 30);

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
    vector<CMatrix> compute_S_n_j(int n, int j, const vector<CMatrix>& Omega,
        const vector<CMatrix>& A_samples, int N);
}