#include "AccMagnus.h"
#include "ClassicMagnus.h"

using namespace std;
using namespace matrix_ops;

namespace matrix_ops {
    // === Helper: right-nested commutator ===
    //   [A1, [A2, [A3, [..., An]]]]
    // list: vector of A_samples, perm: permutation indices (0-based), N: matrix size
    CMatrix right_nested_comm(const vector<CMatrix>& A_samples, const vector<int>& perm, int N)
    {
        int k = perm.size();
        if (k == 0) return CMatrix(N * N, complexd(0, 0));

        // 1. Start from the innermost element (last index in perm)
        CMatrix R = A_samples[perm.back()];

        // 2. Go from the second-last to the beginning
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
        // 24 permutations for Omega5
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
}

// =========================================================
// =================== CORRECTED Magnus ACC ================
// =========================================================

CMatrix magnus_ACC(const vector<CMatrix>& A_full_samples, double dt_grid, int N, int max_order)
{
    double T_total = dt_grid * (A_full_samples.size() - 1);
    CMatrix Omega_total(N * N, complexd(0.0, 0.0));

    //CMatrix Omega1 = trapezoidal_integral(A_full_samples, dt_grid, N);
    CMatrix Omega1 = simpson_integral(A_full_samples, dt_grid, N);
    mat_add(Omega_total, Omega1, Omega_total, N);
    if (max_order < 2) return Omega_total;

    int M = A_full_samples.size();
    if (M < 3) return Omega_total; // Для 2 и 4 порядка нужно минимум 3 точки

    CMatrix A_0 = A_full_samples[0];
    CMatrix A_mid = A_full_samples[M / 2];
    CMatrix A_end = A_full_samples[M - 1];

    if (max_order >= 2) {
        CMatrix comm = commutator(A_end, A_0, N);
        mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));
        mat_add(Omega_total, comm, Omega_total, N);
    }

    if (max_order >= 6 && M >= 7) { // <-- Добавили проверку M >= 7 сюда
        vector<CMatrix> nodes_6;
        for (int k = 0; k <= 6; ++k) {
            int idx = (k * (M - 1)) / 6;
            nodes_6.push_back(A_full_samples[idx]);
        }
        CMatrix Om6 = compute_Omega6_ACC(nodes_6, N);
        mat_add(Omega_total, Om6, Omega_total, N);
    }

    return Omega_total;
}


//  ------ Generation of permutations {2,3,...,n} ------ 
void generate_permutations(vector<int>& base, vector<vector<int>>& result, int start) {
    int n = base.size();
    if (start == n) {
        result.push_back(base);
        return;
    }
    for (int i = start; i < n; ++i) {
        swap(base[start], base[i]);
        generate_permutations(base, result, start + 1);
        swap(base[start], base[i]);
    }
}