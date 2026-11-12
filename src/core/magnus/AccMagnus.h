//#pragma once
//
//#include "MatrixOperations.h"
//
//using namespace std;
//
//// =========================================================
//// =================== CORRECTED Magnus ACC ================
//// =========================================================
//CMatrix magnus_ACC(
//    const vector<CMatrix>& A_full_samples,
//    double dt_grid,
//    int N,
//    int max_order);
//
//namespace matrix_ops {
//    //CMatrix compute_Omega3_ACC(const vector<CMatrix>& A, int N);
//    //CMatrix compute_Omega4_ACC(const vector<CMatrix>& A, int N);
//    CMatrix compute_Omega5_ACC(const vector<CMatrix>& A, int N);
//    CMatrix compute_Omega6_ACC(const vector<CMatrix>& A, int N);
//}
//
//
////  ------ Generation of permutations {2,3,...,n} ------ 
//void generate_permutations(vector<int>& base, vector<vector<int>>& result, int start = 0);

#pragma once

#include "MatrixOperations.h"
#include <vector>

// =========================================================
// =================== CORRECTED Magnus ACC ================
// =========================================================

/**
 * @brief Метод Магнуса повышенной точности (ACC).
 *
 * - max_order <= 4: Метод 4-го порядка (интеграл Симпсона + замыкающий коммутатор)
 * - max_order >= 6: Схема Гаусса-Магнуса 6-го порядка Бланеса-Касаса-Роса на 3 узлах
 *
 * @param A_full_samples Вектор замеров матрицы A(t) = -i * H(t) на равномерной сетке шага
 * @param dt_grid Шаг между замерами в A_full_samples
 * @param N Размерность матриц (N x N)
 * @param max_order Максимальный порядок метода (2, 4 или 6)
 * @return CMatrix Матрица омега \Omega(h)
 */
CMatrix magnus_ACC(
    const std::vector<CMatrix>& A_full_samples,
    double dt_grid,
    int N,
    int max_order);