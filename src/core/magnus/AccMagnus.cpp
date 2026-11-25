#include "AccMagnus.h"
#include "ClassicMagnus.h"

#include <mkl.h>
#include <algorithm>
#include <cmath>
#include <vector>

using namespace std;
using namespace matrix_ops;

namespace {

    // Полиномиальная интерполяция Лагранжа по безразмерной координате c_unit in [0, 1]
    CMatrix interpolate_sample_unit(const vector<CMatrix>& samples, int N, double c_unit)
    {
        const int M = static_cast<int>(samples.size());
        if (M == 1) return samples[0];

        const double x = c_unit * static_cast<double>(M - 1);
        const int degree = std::min(M - 1, 6);
        const int center = static_cast<int>(std::llround(x));
        const int start = std::max(0, std::min(center - degree / 2, M - 1 - degree));

        CMatrix result((size_t)N * N, complexd(0.0, 0.0));
        for (int i = 0; i <= degree; ++i) {
            const double xi = static_cast<double>(start + i);
            double weight = 1.0;
            for (int j = 0; j <= degree; ++j) {
                if (i != j) {
                    const double xj = static_cast<double>(start + j);
                    weight *= (x - xj) / (xi - xj);
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
    const int M = static_cast<int>(A_full_samples.size());
    const double T_total = dt_grid * static_cast<double>(M - 1);
    CMatrix Omega_total((size_t)N * N, complexd(0.0, 0.0));

    if (M < 2) return Omega_total;

    // Порядок 1: линейный член Omega_1 на квадратуре Симпсона (без коммутаторов)
    if (max_order <= 1) {
        return simpson_integral(A_full_samples, dt_grid, N);
    }

    // Порядок 2..4: квадратура Симпсона + симметричный замыкающий коммутатор (4-й порядок)
    if (max_order <= 4) {
        CMatrix Omega1 = simpson_integral(A_full_samples, dt_grid, N);
        mat_add(Omega_total, Omega1, Omega_total, N);

        if (M >= 3) {
            const CMatrix& A_0 = A_full_samples[0];
            const CMatrix& A_end = A_full_samples[M - 1];
            CMatrix comm = commutator(A_end, A_0, N);
            mat_scale_inplace(comm, N, complexd(T_total * T_total / 12.0, 0.0));
            mat_add(Omega_total, comm, Omega_total, N);
        }
        return Omega_total;
    }

    // Порядок 6: Схема Гаусса-Магнуса Бланеса-Касаса-Роса на 3 узлах
    const double sqrt15 = std::sqrt(15.0);
    const double c1 = 0.5 - sqrt15 / 10.0;
    const double c2 = 0.5;
    const double c3 = 0.5 + sqrt15 / 10.0;

    CMatrix A1 = interpolate_sample_unit(A_full_samples, N, c1);
    CMatrix A2 = (M % 2 == 1) ? A_full_samples[(M - 1) / 2] : interpolate_sample_unit(A_full_samples, N, c2);
    CMatrix A3 = interpolate_sample_unit(A_full_samples, N, c3);

    // 1. Безразмерные полиномиальные моменты alpha_i
    CMatrix alpha1 = mat_copy(A2);
    mat_scale_inplace(alpha1, N, complexd(T_total, 0.0));

    CMatrix alpha2 = mat_copy(A3);
    mat_axpy(A1, alpha2, N, complexd(-1.0, 0.0));
    mat_scale_inplace(alpha2, N, complexd(sqrt15 * T_total / 3.0, 0.0));

    CMatrix alpha3 = mat_copy(A3);
    mat_axpy(A2, alpha3, N, complexd(-2.0, 0.0));
    mat_axpy(A1, alpha3, N, complexd(1.0, 0.0));
    mat_scale_inplace(alpha3, N, complexd(10.0 * T_total / 3.0, 0.0));

    // 2. Коммутаторы
    CMatrix C1 = commutator(alpha1, alpha2, N);

    CMatrix R1 = mat_copy(alpha3);
    mat_axpy(C1, R1, N, complexd(-1.0, 0.0));
    CMatrix C2 = commutator(alpha2, R1, N);

    CMatrix R2 = mat_copy(C1);
    mat_axpy(alpha3, R2, N, complexd(2.0, 0.0));
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
