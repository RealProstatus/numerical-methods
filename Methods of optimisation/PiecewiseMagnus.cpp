#include "PiecewiseMagnus.h"
#include "AccMagnus.h"
#include "ClassicMagnus.h"
#include "ChebyshevExpm.h"

#include <functional>

using namespace std;
using namespace matrix_ops;

CMatrix piecewise_magnus_solver(double t0, double T, double macro_dt, int N,
    const CMatrix& H0, const CMatrix& H_mod,
    double eps0, double W, int order)
{
    CMatrix U_total = utils::eye(N);
    int num_steps = (int)std::round((T - t0) / macro_dt);

    // Для 4-го порядка нам нужно 3 точки (начало, середина, конец шага). 
    // Значит микро-шаг = macro_dt / 2.0
    int sub_steps = (order >= 6) ? 6 : 2;
    double micro_dt = macro_dt / sub_steps;

    for (int step = 0; step < num_steps; ++step) {
        double t_curr = t0 + step * macro_dt;
        double t_next = t_curr + macro_dt;

        // 1 Генерируем сэмплы только для текущего короткого шага
        vector<CMatrix> step_samples = generate_samples(t_curr, t_next, micro_dt, N, H0, H_mod, eps0, W);

        // 2 Считаем локальную Омегу
        CMatrix Omega_local = magnus_ACC(step_samples, micro_dt, N, order);

        // 3 Берем экспоненту
        CMatrix U_local = matrix_ops::expm_cheb(Omega_local, N, 15);

        // 4 Обновляем общее решение: U(t+dt) = U_local * U(t)
        CMatrix temp((size_t)N * N, complexd(0.0, 0.0));
        matrix_ops::matmul(U_local, U_total, temp, N);
        U_total = temp;
    }

    return U_total;
}