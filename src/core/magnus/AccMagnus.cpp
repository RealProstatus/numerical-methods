//#include "AccMagnus.h"
//#include "ClassicMagnus.h"
//#include <mkl.h>
//#include <cstring>
//#include <cmath>
//
//using namespace std;
//using namespace matrix_ops;
//
//namespace matrix_ops {
//    // === Helper: right-nested commutator ===
//    //   [A1, [A2, [A3, [..., An]]]]
//    CMatrix right_nested_comm(const vector<CMatrix>& A_samples, const vector<int>& perm, int N)
//    {
//        int k = perm.size();
//        if (k == 0) return CMatrix(N * N, complexd(0, 0));
//        if (k == 1) return A_samples[perm.back()];
//
//        CMatrix R_curr = A_samples[perm.back()];
//        CMatrix R_next(N * N, complexd(0, 0));
//
//        for (int i = k - 2; i >= 0; --i) {
//            commutator_inplace(A_samples[perm[i]], R_curr, R_next, N);
//            std::swap(R_curr, R_next);
//        }
//
//        return R_curr;
//    }
//
//    // Fast right-nested commutator without heap allocations
//    void right_nested_comm_fast(const vector<CMatrix>& A_samples, const vector<int>& perm, int N, CMatrix& R_curr, CMatrix& R_next)
//    {
//        int k = perm.size();
//        if (k == 0) {
//            memset(R_curr.data(), 0, R_curr.size() * sizeof(complexd));
//            return;
//        }
//
//        cblas_zcopy(N * N, A_samples[perm.back()].data(), 1, R_curr.data(), 1);
//
//        if (k == 1) return;
//
//        for (int i = k - 2; i >= 0; --i) {
//            commutator_inplace(A_samples[perm[i]], R_curr, R_next, N);
//            std::swap(R_curr, R_next);
//        }
//    }
//
//    // =========================================================
//    // ==================== OMEGA 5 ACC ========================
//    // =========================================================
//
//    CMatrix compute_Omega5_ACC(const vector<CMatrix>& A, int N) {
//        static const vector<vector<int>> perms = {
//            {1,2,3,4},{1,2,4,3},{1,3,2,4},{1,3,4,2},{1,4,2,3},{1,4,3,2},
//            {0,2,3,4},{0,2,4,3},{0,3,2,4},{0,3,4,2},{0,4,2,3},{0,4,3,2},
//            {2,0,3,4},{2,0,4,3},{2,3,0,4},{2,3,4,0},{2,4,0,3},{2,4,3,0},
//            {3,0,1,4},{3,0,4,1},{3,1,0,4},{3,1,4,0},{3,4,0,1},{3,4,1,0}
//        };
//
//        static const vector<double> coeffs = {
//            -1.0 / 20, 1.0 / 30, 1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60,
//             1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60, 1.0 / 30, -1.0 / 60,
//             1.0 / 30, 1.0 / 30, 1.0 / 30, -1.0 / 60, -1.0 / 60, -1.0 / 60,
//             1.0 / 30, -1.0 / 60, 1.0 / 30, -1.0 / 60, -1.0 / 60, -1.0 / 20
//        };
//
//        CMatrix sum(N * N, complexd(0, 0));
//        CMatrix workspace_curr(N * N);
//        CMatrix workspace_next(N * N);
//
//        for (size_t i = 0; i < perms.size(); ++i) {
//            right_nested_comm_fast(A, perms[i], N, workspace_curr, workspace_next);
//            mat_scale_inplace(workspace_curr, N, complexd(coeffs[i], 0));
//            mat_add(sum, workspace_curr, sum, N);
//        }
//
//        return sum;
//    }
//
//    // =========================================================
//    // ==================== OMEGA 6 ACC ========================
//    // =========================================================
//
//    CMatrix compute_Omega6_ACC(const vector<CMatrix>& A, int N)
//    {
//        using VI = std::vector<int>;
//
//        static const std::vector<VI> perms = {
//            {1,2,3,4,5},{1,2,3,5,4},{1,2,4,3,5},{1,2,4,5,3},
//            {1,2,5,3,4},{1,2,5,4,3},{1,3,2,4,5},{1,3,2,5,4},
//            {1,3,4,2,5},{1,3,4,5,2},{1,3,5,2,4},{1,3,5,4,2},
//            {1,4,2,3,5},{1,4,2,5,3},{1,4,3,2,5},{1,4,3,5,2},
//            {1,4,5,2,3},{1,4,5,3,2},{1,5,2,3,4},{1,5,2,4,3},
//            {1,5,3,2,4},{1,5,3,4,2},{1,5,4,2,3},{1,5,4,3,2},
//
//            {2,1,3,4,5},{2,1,3,5,4},{2,1,4,3,5},{2,1,4,5,3},
//            {2,1,5,3,4},{2,1,5,4,3},{2,3,1,4,5},{2,3,1,5,4},
//            {2,3,4,1,5},{2,3,4,5,1},{2,3,5,1,4},{2,3,5,4,1},
//            {2,4,1,3,5},{2,4,1,5,3},{2,4,3,1,5},{2,4,3,5,1},
//            {2,4,5,1,3},{2,4,5,3,1},{2,5,1,3,4},{2,5,1,4,3},
//            {2,5,3,1,4},{2,5,3,4,1},{2,5,4,1,3},{2,5,4,3,1},
//
//            {3,1,2,4,5},{3,1,2,5,4},{3,1,4,2,5},{3,1,4,5,2},
//            {3,1,5,2,4},{3,1,5,4,2},{3,2,1,4,5},{3,2,1,5,4},
//            {3,2,4,1,5},{3,2,4,5,1},{3,2,5,1,4},{3,2,5,4,1},
//            {3,4,1,2,5},{3,4,1,5,2},{3,4,2,1,5},{3,4,2,5,1},
//            {3,4,5,1,2},{3,4,5,2,1},{3,5,1,2,4},{3,5,1,4,2},
//            {3,5,2,1,4},{3,5,2,4,1},{3,5,4,1,2},{3,5,4,2,1},
//
//            {4,1,2,3,5},{4,1,2,5,3},{4,1,3,2,5},{4,1,3,5,2},
//            {4,1,5,2,3},{4,1,5,3,2},{4,2,1,3,5},{4,2,1,5,3},
//            {4,2,3,1,5},{4,2,3,5,1},{4,2,5,1,3},{4,2,5,3,1},
//            {4,3,1,2,5},{4,3,1,5,2},{4,3,2,1,5},{4,3,2,5,1},
//            {4,3,5,1,2},{4,3,5,2,1},{4,5,1,2,3},{4,5,1,3,2},
//            {4,5,2,1,3},{4,5,2,3,1},{4,5,3,1,2},{4,5,3,2,1},
//
//            {5,1,2,3,4},{5,1,2,4,3},{5,1,3,2,4},{5,1,3,4,2},
//            {5,1,4,2,3},{5,1,4,3,2},{5,2,1,3,4},{5,2,1,4,3},
//            {5,2,3,1,4},{5,2,3,4,1},{5,2,4,1,3},{5,2,4,3,1},
//            {5,3,1,2,4},{5,3,1,4,2},{5,3,2,1,4},{5,3,2,4,1},
//            {5,3,4,1,2},{5,3,4,2,1},{5,4,1,2,3},{5,4,1,3,2},
//            {5,4,2,1,3},{5,4,2,3,1},{5,4,3,1,2},{5,4,3,2,1}
//        };
//
//        static const std::vector<double> coeffs = {
//            1.0 / 120, -1.0 / 180, -1.0 / 180, 1.0 / 240, -1.0 / 180, 1.0 / 360,
//            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
//            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,
//
//            -1.0 / 180, 1.0 / 240, 1.0 / 240, -1.0 / 360, 1.0 / 240, -1.0 / 720,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,
//
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,
//
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600,
//            1.0 / 240, -1.0 / 360, -1.0 / 360, 1.0 / 720, -1.0 / 360, 1.0 / 1200,
//            -1.0 / 360, 1.0 / 720, 1.0 / 720, -1.0 / 1200, 1.0 / 720, -1.0 / 3600
//        };
//
//        CMatrix sum(N * N, complexd(0, 0));
//        CMatrix workspace_curr(N * N);
//        CMatrix workspace_next(N * N);
//
//        for (size_t i = 0; i < perms.size(); ++i) {
//            right_nested_comm_fast(A, perms[i], N, workspace_curr, workspace_next);
//            mat_scale_inplace(workspace_curr, N, complexd(coeffs[i], 0));
//            mat_add(sum, workspace_curr, sum, N);
//        }
//
//        return sum;
//    }
//}
//
//// =========================================================
//// =================== CORRECTED Magnus ACC ================
//// =========================================================
//
//CMatrix magnus_ACC(const vector<CMatrix>& A_full_samples, double dt_grid, int N, int max_order)
//{
//    double T_total = dt_grid * (A_full_samples.size() - 1);
//    CMatrix Omega_total(N * N, complexd(0.0, 0.0));
//
//    // 1. Порядок 1 (интеграл Симпсона)
//    CMatrix Omega1 = simpson_integral(A_full_samples, dt_grid, N);
//    mat_add(Omega_total, Omega1, Omega_total, N);
//    if (max_order < 2) return Omega_total;
//
//    int M = A_full_samples.size();
//    if (M < 3) return Omega_total;
//
//    // 2. Порядок 2..4 (симметричный коммутатор, замыкающий схему до 4-го порядка)
//    CMatrix A_0 = A_full_samples[0];
//    CMatrix A_end = A_full_samples[M - 1];
//
//    if (max_order >= 2) {
//        CMatrix comm = commutator(A_end, A_0, N);
//        mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));
//        mat_add(Omega_total, comm, Omega_total, N);
//    }
//
//    // 3. Порядок 5
//    if (max_order >= 5 && M >= 5) {
//        vector<CMatrix> nodes_5;
//        for (int k = 0; k <= 4; ++k) {
//            int idx = (k * (M - 1)) / 4;
//            nodes_5.push_back(A_full_samples[idx]);
//        }
//        CMatrix Om5 = compute_Omega5_ACC(nodes_5, N);
//        double h4 = std::pow(T_total, 4.0);
//        mat_scale_inplace(Om5, N, complexd(h4, 0.0));
//        mat_add(Omega_total, Om5, Omega_total, N);
//    }
//
//    // 4. Порядок 6
//    if (max_order >= 6 && M >= 7) {
//        vector<CMatrix> nodes_6;
//        for (int k = 0; k <= 6; ++k) {
//            int idx = (k * (M - 1)) / 6;
//            nodes_6.push_back(A_full_samples[idx]);
//        }
//        CMatrix Om6 = compute_Omega6_ACC(nodes_6, N);
//        double h5 = std::pow(T_total, 5.0);
//        mat_scale_inplace(Om6, N, complexd(h5, 0.0));
//        mat_add(Omega_total, Om6, Omega_total, N);
//    }
//
//    return Omega_total;
//}
//
//void generate_permutations(vector<int>& base, vector<vector<int>>& result, int start) {
//    int n = base.size();
//    if (start == n) {
//        result.push_back(base);
//        return;
//    }
//    for (int i = start; i < n; ++i) {
//        swap(base[start], base[i]);
//        generate_permutations(base, result, start + 1);
//        swap(base[start], base[i]);
//    }
//}

#include "AccMagnus.h"
#include "ClassicMagnus.h"
#include <mkl.h>
#include <vector>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace matrix_ops;

namespace {

    // Полиномиальная интерполяция Лагранжа для вычисления A(t) в произвольной точке шага
    CMatrix interpolate_sample(const vector<CMatrix>& samples, double dt, int N, double t_target)
    {
        int M = (int)samples.size();
        if (M == 1) return samples[0];

        double x = t_target / dt;
        int degree = std::min(M - 1, 6); // Степень интерполянта до 6 включительно
        int center = (int)std::round(x);
        int start = std::max(0, std::min(center - degree / 2, M - 1 - degree));

        vector<double> x_nodes(degree + 1);
        for (int i = 0; i <= degree; ++i) {
            x_nodes[i] = start + i;
        }

        CMatrix result((size_t)N * N, complexd(0.0, 0.0));
        for (int i = 0; i <= degree; ++i) {
            double weight = 1.0;
            for (int j = 0; j <= degree; ++j) {
                if (i != j) {
                    weight *= (x - x_nodes[j]) / (x_nodes[i] - x_nodes[j]);
                }
            }
            mat_axpy(samples[start + i], result, N, complexd(weight, 0.0));
        }
        return result;
    }

} // namespace

// =========================================================
// =================== CORRECTED Magnus ACC ================
// =========================================================

CMatrix magnus_ACC(const vector<CMatrix>& A_full_samples, double dt_grid, int N, int max_order)
{
    int M = (int)A_full_samples.size();
    double T_total = dt_grid * (M - 1);
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));

    if (M < 2) return Omega_total;

    // -------------------------------------------------------------
    // Порядок 1–4: Проверенная квадратура Симпсона + замыкающий коммутатор
    // -------------------------------------------------------------
    if (max_order <= 4) {
        // Линейный член: интеграл Симпсона (локально O(h^5))
        CMatrix Omega1 = simpson_integral(A_full_samples, dt_grid, N);
        mat_add(Omega_total, Omega1, Omega_total, N);

        if (max_order >= 2 && M >= 3) {
            // Симметричный коммутатор на границах интервала: h^2 / 12 * [A(T), A(0)]
            CMatrix A_0 = A_full_samples[0];
            CMatrix A_end = A_full_samples[M - 1];
            CMatrix comm = commutator(A_end, A_0, N);
            mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));
            mat_add(Omega_total, comm, Omega_total, N);
        }
        return Omega_total;
    }

    // -------------------------------------------------------------
    // Порядок 6: Схема Гаусса-Магнуса Бланеса-Касаса-Роса (Blanes-Casas-Ros)
    // -------------------------------------------------------------
    const double sqrt15 = std::sqrt(15.0);
    const double c1 = 0.5 - sqrt15 / 10.0;
    const double c2 = 0.5;
    const double c3 = 0.5 + sqrt15 / 10.0;

    double t1 = c1 * T_total;
    double t2 = c2 * T_total;
    double t3 = c3 * T_total;

    CMatrix A1 = interpolate_sample(A_full_samples, dt_grid, N, t1);
    CMatrix A2 = (M % 2 == 1) ? A_full_samples[(M - 1) / 2] : interpolate_sample(A_full_samples, dt_grid, N, t2);
    CMatrix A3 = interpolate_sample(A_full_samples, dt_grid, N, t3);

    // 1. Безразмерные полиномиальные моменты alpha_i
    // alpha1 = h * A2
    CMatrix alpha1 = mat_copy(A2);
    mat_scale_inplace(alpha1, N, complexd(T_total, 0.0));

    // alpha2 = (sqrt(15) * h / 3) * (A3 - A1)
    CMatrix alpha2 = mat_copy(A3);
    mat_axpy(A1, alpha2, N, complexd(-1.0, 0.0));
    mat_scale_inplace(alpha2, N, complexd(sqrt15 * T_total / 3.0, 0.0));

    // alpha3 = (10 * h / 3) * (A3 - 2 * A2 + A1)
    CMatrix alpha3 = mat_copy(A3);
    mat_axpy(A2, alpha3, N, complexd(-2.0, 0.0));
    mat_axpy(A1, alpha3, N, complexd(1.0, 0.0));
    mat_scale_inplace(alpha3, N, complexd(10.0 * T_total / 3.0, 0.0));

    // 2. Вычисление коммутаторов
    // C1 = [alpha1, alpha2]
    CMatrix C1 = commutator(alpha1, alpha2, N);

    // R1 = alpha3 - C1
    CMatrix R1 = mat_copy(alpha3);
    mat_axpy(C1, R1, N, complexd(-1.0, 0.0));

    // C2 = [alpha2, R1]
    CMatrix C2 = commutator(alpha2, R1, N);

    // R2 = 2 * alpha3 + C1
    CMatrix R2 = mat_copy(C1);
    mat_axpy(alpha3, R2, N, complexd(2.0, 0.0));

    // C3 = [alpha1, [alpha1, R2]]
    CMatrix inner_comm = commutator(alpha1, R2, N);
    CMatrix C3 = commutator(alpha1, inner_comm, N);

    // 3. Сборка Omega:
    // Omega = alpha1 + (1/12)*alpha3 - (1/12)*C1 + (1/240)*C2 + (1/720)*C3
    mat_axpy(alpha1, Omega_total, N, complexd(1.0, 0.0));
    mat_axpy(alpha3, Omega_total, N, complexd(1.0 / 12.0, 0.0));
    mat_axpy(C1, Omega_total, N, complexd(-1.0 / 12.0, 0.0));
    mat_axpy(C2, Omega_total, N, complexd(1.0 / 240.0, 0.0));
    mat_axpy(C3, Omega_total, N, complexd(1.0 / 720.0, 0.0));

    return Omega_total;
}