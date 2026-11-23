#include "magnus/solvers.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "core/expm/ChebyshevExpm.h"
#include "core/expm/EigenPadeExpm.h"
#include "core/expm/TaylorExpm.h"
#include "core/magnus/AccMagnus.h"
#include "core/magnus/ClassicMagnus.h"
#include "core/magnus/RecursiveMagnus.h"
#include "core/magnus/Formula314Magnus.h"
#include "core/matrix/MatrixOperations.h"
#include "core/utils/ProblemSetup.h"
#include "solvers/single_cpu/Formula314ChebyshevSolver.h"
#include "solvers/single_cpu/PiecewiseMagnus.h"


#if defined(MAGNUS_HAS_MPI)
#include <mpi.h>
#include "solvers/multi_cpu/MpiMagnusChebyshev.h"
#endif

namespace magnus {

    namespace {

        CMatrix apply_expm(const CMatrix& Omega, int N, const SolverOptions& opts) {
            switch (opts.expm_method) {
            case ExpmMethod::Chebyshev:
                return matrix_ops::expm_cheb(Omega, N, opts.chebyshev_degree);
            case ExpmMethod::PadeEigen:
                return matrix_ops::expm_pade_eigen(Omega, N);
            case ExpmMethod::Taylor:
                return matrix_ops::expm_taylor(Omega, N, opts.taylor_terms);
            default:
                return matrix_ops::expm_cheb(Omega, N, opts.chebyshev_degree);
            }
        }

        void propagate_arbitrary_modulation_mkl(
            int N,
            const CMatrix& H0,
            const CMatrix& H1,
            const double* time_nodes,
            std::size_t num_time_nodes,
            const double* mod_grid,
            std::size_t num_mod_points,
            const SolverOptions& opts,
            CMatrix& U_total
        ) {
            const std::size_t steps = num_time_nodes - 1;
            const std::size_t total_elems = static_cast<std::size_t>(N) * N;

            // Переходим к антиэрмитовым генераторам A0 = -i/hbar * H0, A1 = -i/hbar * H1
            const Complex gen_factor(0.0, -1.0 / opts.hbar);
            CMatrix A0 = matrix_ops::mat_copy(H0);
            CMatrix A1 = matrix_ops::mat_copy(H1);
            matrix_ops::mat_scale_inplace(A0, N, gen_factor);
            matrix_ops::mat_scale_inplace(A1, N, gen_factor);

            U_total.assign(total_elems, Complex(0.0, 0.0));
            for (int i = 0; i < N; ++i) {
                U_total[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
            }

            // Унифицированный быстрый путь для 3-узловой квадратуры Гаусса-Лежандра (3 * steps).
            // Поддерживает все порядки (2, 4, 6) и произвольную неравномерную сетку time_nodes.
            if (num_mod_points == 3 * steps) {
                CMatrix Omega(total_elems, Complex(0.0, 0.0));
                CMatrix U_next(total_elems, Complex(0.0, 0.0));

                const bool is_order2 = (opts.magnus_method == MagnusMethod::Classic2);
                const bool is_order4 = (opts.magnus_method == MagnusMethod::Classic4 ||
                    opts.magnus_method == MagnusMethod::Acc4 ||
                    opts.magnus_method == MagnusMethod::Formula314_4);
                const bool is_order6 = !is_order2 && !is_order4;

                CMatrix C1, C2, C3, C4, C5, C7;
                if (!is_order2) {
                    C1.assign(total_elems, Complex(0.0, 0.0));
                    matrix_ops::commutator_inplace(A0, A1, C1, N);
                }
                if (is_order6) {
                    C2.assign(total_elems, Complex(0.0, 0.0));
                    C3.assign(total_elems, Complex(0.0, 0.0));
                    C4.assign(total_elems, Complex(0.0, 0.0));
                    C5.assign(total_elems, Complex(0.0, 0.0));
                    C7.assign(total_elems, Complex(0.0, 0.0));

                    matrix_ops::commutator_inplace(A0, C1, C2, N);
                    matrix_ops::commutator_inplace(A1, C1, C3, N);
                    matrix_ops::commutator_inplace(A0, C2, C4, N);
                    // По тождеству Якоби C6 = [A0, [A1, C1]] == [A1, [A0, C1]] = C5
                    matrix_ops::commutator_inplace(A1, C2, C5, N);
                    matrix_ops::commutator_inplace(A1, C3, C7, N);
                }

                const double s15 = std::sqrt(15.0);
                for (std::size_t step = 0; step < steps; ++step) {
                    const double h = time_nodes[step + 1] - time_nodes[step];
                    const double fv1 = mod_grid[3 * step + 0];
                    const double fv2 = mod_grid[3 * step + 1];
                    const double fv3 = mod_grid[3 * step + 2];

                    std::fill(Omega.begin(), Omega.end(), Complex(0.0, 0.0));
                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));

                    if (is_order2) {
                        matrix_ops::mat_axpy(A1, Omega, N, Complex(h * fv2, 0.0));
                    }
                    else {
                        const double fm = fv2;
                        const double d1 = (s15 * h / 3.0) * (fv3 - fv1);
                        const double d2 = (10.0 * h / 3.0) * (fv3 - 2.0 * fv2 + fv1);

                        const double c_A1 = h * fm + (1.0 / 12.0) * d2;
                        const double c_C1 = -(1.0 / 12.0) * h * d1;

                        matrix_ops::mat_axpy(A1, Omega, N, Complex(c_A1, 0.0));
                        matrix_ops::mat_axpy(C1, Omega, N, Complex(c_C1, 0.0));

                        if (is_order6) {
                            const double h2 = h * h;
                            const double h3 = h2 * h;
                            const double c_C2 = (1.0 / 360.0) * h2 * d2;
                            const double c_C3 = -(1.0 / 240.0) * h * (d1 * d1) + (1.0 / 360.0) * h2 * fm * d2;
                            const double c_C4 = (1.0 / 720.0) * h3 * d1;
                            const double c_C56 = (1.0 / 360.0) * h3 * fm * d1; // c_C5 + c_C6
                            const double c_C7 = (1.0 / 720.0) * h3 * (fm * fm) * d1;

                            matrix_ops::mat_axpy(C2, Omega, N, Complex(c_C2, 0.0));
                            matrix_ops::mat_axpy(C3, Omega, N, Complex(c_C3, 0.0));
                            matrix_ops::mat_axpy(C4, Omega, N, Complex(c_C4, 0.0));
                            matrix_ops::mat_axpy(C5, Omega, N, Complex(c_C56, 0.0));
                            matrix_ops::mat_axpy(C7, Omega, N, Complex(c_C7, 0.0));
                        }
                    }

                    CMatrix U_step = apply_expm(Omega, N, opts);
                    matrix_ops::matmul(U_step, U_total, U_next, N);
                    U_total.swap(U_next);
                }
                return;
            }

            // Ветка для Acc4 / Acc6 через штатный magnus_ACC
            if (opts.magnus_method == MagnusMethod::Acc4 || opts.magnus_method == MagnusMethod::Acc6) {
                const int order = (opts.magnus_method == MagnusMethod::Acc6) ? 6 : 4;
                CMatrix U_next(total_elems, Complex(0.0, 0.0));

                for (std::size_t step = 0; step < steps; ++step) {
                    const double dt = time_nodes[step + 1] - time_nodes[step];
                    std::vector<CMatrix> samples;
                    double micro_dt = dt;

                    if (num_mod_points >= 2 * steps + 1) {
                        std::size_t start_step = step;
                        if (order == 6 && steps >= 3) {
                            if (step == 0) start_step = 0;
                            else if (step + 2 >= steps) start_step = steps - 3;
                            else start_step = step - 1;

                            micro_dt = 0.5 * dt;
                            samples.reserve(7);
                            for (std::size_t k = 0; k <= 6; ++k) {
                                const double f_val = mod_grid[2 * start_step + k];
                                CMatrix Ak = matrix_ops::mat_copy(A0);
                                matrix_ops::mat_axpy(A1, Ak, N, Complex(f_val, 0.0));
                                samples.push_back(std::move(Ak));
                            }
                            if (start_step != step) {
                                const std::size_t offset = 2 * (step - start_step);
                                std::vector<CMatrix> local_samples(samples.begin() + offset, samples.begin() + offset + 3);
                                samples = std::move(local_samples);
                            }
                        }
                        else {
                            micro_dt = 0.5 * dt;
                            samples.reserve(3);
                            for (std::size_t k = 0; k <= 2; ++k) {
                                const double f_val = mod_grid[2 * step + k];
                                CMatrix Ak = matrix_ops::mat_copy(A0);
                                matrix_ops::mat_axpy(A1, Ak, N, Complex(f_val, 0.0));
                                samples.push_back(std::move(Ak));
                            }
                        }
                    }
                    else {
                        micro_dt = dt;
                        for (std::size_t k = 0; k <= 1; ++k) {
                            const double f_val = mod_grid[step + k];
                            CMatrix Ak = matrix_ops::mat_copy(A0);
                            matrix_ops::mat_axpy(A1, Ak, N, Complex(f_val, 0.0));
                            samples.push_back(std::move(Ak));
                        }
                    }

                    CMatrix Omega = magnus_ACC(samples, micro_dt, N, order);
                    CMatrix U_step = apply_expm(Omega, N, opts);
                    matrix_ops::matmul(U_step, U_total, U_next, N);
                    U_total.swap(U_next);
                }
                return;
            }

            // Предвычисление коммутаторов от антиэрмитовых A0, A1 (один раз до цикла)
            CMatrix C1(total_elems, Complex(0.0, 0.0));
            matrix_ops::commutator_inplace(A0, A1, C1, N);

            CMatrix C2(total_elems, Complex(0.0, 0.0));
            CMatrix C3(total_elems, Complex(0.0, 0.0));
            CMatrix C4(total_elems, Complex(0.0, 0.0));
            CMatrix C5(total_elems, Complex(0.0, 0.0));
            CMatrix C6(total_elems, Complex(0.0, 0.0));
            CMatrix C7(total_elems, Complex(0.0, 0.0));

            const bool use_order6 = (opts.magnus_method == MagnusMethod::Formula314_6 ||
                opts.magnus_method == MagnusMethod::Recursive);
            if (use_order6) {
                matrix_ops::commutator_inplace(A0, C1, C2, N);
                matrix_ops::commutator_inplace(A1, C1, C3, N);
                matrix_ops::commutator_inplace(A0, C2, C4, N);
                matrix_ops::commutator_inplace(A1, C2, C5, N);
                matrix_ops::commutator_inplace(A0, C3, C6, N);
                matrix_ops::commutator_inplace(A1, C3, C7, N);
            }

            CMatrix Omega(total_elems, Complex(0.0, 0.0));
            CMatrix U_next(total_elems, Complex(0.0, 0.0));

            for (std::size_t step = 0; step < steps; ++step) {
                const double h = time_nodes[step + 1] - time_nodes[step];
                const double h2 = h * h;
                const double h3 = h2 * h;
                const double h5 = h3 * h2;

                // Точная 3-узловая квадратура Гаусса-Лежандра (3 * steps точек)
                if (num_mod_points == 3 * steps && use_order6) {
                    const double fv1 = mod_grid[3 * step + 0];
                    const double fv2 = mod_grid[3 * step + 1];
                    const double fv3 = mod_grid[3 * step + 2];
                    const double s15 = std::sqrt(15.0);

                    const double fm = fv2;
                    const double d1 = (s15 * h / 3.0) * (fv3 - fv1);
                    const double d2 = (10.0 * h / 3.0) * (fv3 - 2.0 * fv2 + fv1);

                    const double c_A1 = h * fm + (1.0 / 12.0) * d2;
                    const double c_C1 = -(1.0 / 12.0) * h * d1;
                    const double c_C2 = (1.0 / 360.0) * h2 * d2;
                    const double c_C3 = -(1.0 / 240.0) * h * (d1 * d1) + (1.0 / 360.0) * h2 * fm * d2;
                    const double c_C4 = (1.0 / 720.0) * h3 * d1;
                    const double c_C5 = (1.0 / 720.0) * h3 * fm * d1;
                    const double c_C6 = c_C5;
                    const double c_C7 = (1.0 / 720.0) * h3 * (fm * fm) * d1;

                    std::fill(Omega.begin(), Omega.end(), Complex(0.0, 0.0));
                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                    matrix_ops::mat_axpy(A1, Omega, N, Complex(c_A1, 0.0));
                    matrix_ops::mat_axpy(C1, Omega, N, Complex(c_C1, 0.0));
                    matrix_ops::mat_axpy(C2, Omega, N, Complex(c_C2, 0.0));
                    matrix_ops::mat_axpy(C3, Omega, N, Complex(c_C3, 0.0));
                    matrix_ops::mat_axpy(C4, Omega, N, Complex(c_C4, 0.0));
                    matrix_ops::mat_axpy(C5, Omega, N, Complex(c_C5, 0.0));
                    matrix_ops::mat_axpy(C6, Omega, N, Complex(c_C6, 0.0));
                    matrix_ops::mat_axpy(C7, Omega, N, Complex(c_C7, 0.0));

                    CMatrix U_step = apply_expm(Omega, N, opts);
                    matrix_ops::matmul(U_step, U_total, U_next, N);
                    U_total.swap(U_next);
                    continue;
                }

                double f0 = 0.0, f_half = 0.0, f1 = 0.0;
                double fp_half = 0.0, f2_half = 0.0;

                if (num_mod_points >= 2 * steps + 1) {
                    f0 = mod_grid[2 * step];
                    f_half = mod_grid[2 * step + 1];
                    f1 = mod_grid[2 * step + 2];
                    const double tau = 0.5 * h;

                    if (step > 0 && step + 1 < steps) {
                        const double fm2 = mod_grid[2 * step - 1];
                        const double fp2 = mod_grid[2 * step + 3];
                        fp_half = (fm2 - 8.0 * f0 + 8.0 * f1 - fp2) / (12.0 * tau);
                        f2_half = (-fm2 + 16.0 * f0 - 30.0 * f_half + 16.0 * f1 - fp2) / (12.0 * tau * tau);
                    }
                    else if (step == 0 && steps >= 2) {
                        const double g3 = mod_grid[3];
                        const double g4 = mod_grid[4];
                        fp_half = (-3.0 * f0 - 10.0 * f_half + 18.0 * f1 - 6.0 * g3 + g4) / (12.0 * tau);
                        f2_half = (11.0 * f0 - 20.0 * f_half + 6.0 * f1 + 4.0 * g3 - g4) / (12.0 * tau * tau);
                    }
                    else if (step + 1 == steps && steps >= 2) {
                        const double gm3 = mod_grid[2 * steps - 4];
                        const double gm2 = mod_grid[2 * steps - 3];
                        fp_half = (-gm3 + 6.0 * gm2 - 18.0 * f0 + 10.0 * f_half + 3.0 * f1) / (12.0 * tau);
                        f2_half = (-gm3 + 4.0 * gm2 + 6.0 * f0 - 20.0 * f_half + 11.0 * f1) / (12.0 * tau * tau);
                    }
                    else {
                        fp_half = (f1 - f0) / h;
                        f2_half = 4.0 * (f0 - 2.0 * f_half + f1) / h2;
                    }
                }
                else if (num_mod_points >= steps + 1) {
                    f0 = mod_grid[step];
                    f1 = mod_grid[step + 1];
                    f_half = 0.5 * (f0 + f1);
                    fp_half = (f1 - f0) / h;
                    f2_half = 0.0;
                }

                std::fill(Omega.begin(), Omega.end(), Complex(0.0, 0.0));

                if (opts.magnus_method == MagnusMethod::Classic2) {
                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                    matrix_ops::mat_axpy(A1, Omega, N, Complex(h * f_half, 0.0));
                }
                else {
                    const double c_A1 = h * (0.1 * f0 + 0.8 * f_half + 0.1 * f1) + (h3 / 60.0) * f2_half;
                    const double c_C1 = -(h2 / 20.0) * (f1 - f0) - (h3 / 30.0) * fp_half;

                    matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                    matrix_ops::mat_axpy(A1, Omega, N, Complex(c_A1, 0.0));
                    matrix_ops::mat_axpy(C1, Omega, N, Complex(c_C1, 0.0));

                    if (use_order6) {
                        matrix_ops::mat_axpy(C2, Omega, N, Complex((h5 / 720.0) * f2_half, 0.0));
                        matrix_ops::mat_axpy(C3, Omega, N, Complex((h5 / 720.0) * f_half * f2_half - (h5 / 240.0) * fp_half * fp_half, 0.0));
                        matrix_ops::mat_axpy(C4, Omega, N, Complex((h5 / 720.0) * fp_half, 0.0));
                        matrix_ops::mat_axpy(C5, Omega, N, Complex((h5 / 720.0) * f_half * fp_half, 0.0));
                        matrix_ops::mat_axpy(C6, Omega, N, Complex((h5 / 720.0) * f_half * fp_half, 0.0));
                        matrix_ops::mat_axpy(C7, Omega, N, Complex((h5 / 720.0) * f_half * f_half * fp_half, 0.0));
                    }
                }

                CMatrix U_step = apply_expm(Omega, N, opts);
                matrix_ops::matmul(U_step, U_total, U_next, N);
                U_total.swap(U_next);
            }
        }

    } // namespace

    bool has_mpi_backend() noexcept {
#if defined(MAGNUS_HAS_MPI)
        return true;
#else
        return false;
#endif
    }

    void compute_expm_cpu(
        std::size_t dim,
        const Complex* A_in,
        Complex* expA_out,
        ExpmMethod method,
        int degree_or_terms
    ) {
        const int N = static_cast<int>(dim);
        CMatrix A(A_in, A_in + dim * dim);
        CMatrix res;
        switch (method) {
        case ExpmMethod::Chebyshev:
            res = matrix_ops::expm_cheb(A, N, degree_or_terms);
            break;
        case ExpmMethod::PadeEigen:
            res = matrix_ops::expm_pade_eigen(A, N);
            break;
        case ExpmMethod::Taylor:
            res = matrix_ops::expm_taylor(A, N, degree_or_terms);
            break;
        }
        std::memcpy(expA_out, res.data(), dim * dim * sizeof(Complex));
    }

    void propagate_general_gl_mkl(
        std::size_t dim,
        const double* time_nodes,
        std::size_t num_time_nodes,
        const Complex* A_gl_nodes,
        const SolverOptions& opts,
        Complex* U_out_ptr
    ) {
        if (num_time_nodes < 2) {
            throw std::invalid_argument("time_nodes must contain at least 2 points.");
        }

        const int N = static_cast<int>(dim);
        const std::size_t steps = num_time_nodes - 1;
        const std::size_t elems = dim * dim;
        const double s15 = std::sqrt(15.0);

        CMatrix U_total(elems, Complex(0.0, 0.0));
        for (int i = 0; i < N; ++i) {
            U_total[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
        }
        CMatrix U_next(elems, Complex(0.0, 0.0));

        CMatrix A1(elems), A2(elems), A3(elems);
        CMatrix alpha1(elems), alpha2(elems), alpha3(elems);
        CMatrix C1(elems), C2(elems), C3(elems), R1(elems), R2(elems), inner(elems), Omega(elems);

        for (std::size_t step = 0; step < steps; ++step) {
            const double h = time_nodes[step + 1] - time_nodes[step];
            const Complex* base = A_gl_nodes + step * 3 * elems;
            std::memcpy(A1.data(), base + 0 * elems, elems * sizeof(Complex));
            std::memcpy(A2.data(), base + 1 * elems, elems * sizeof(Complex));
            std::memcpy(A3.data(), base + 2 * elems, elems * sizeof(Complex));

            if (opts.magnus_method == MagnusMethod::Classic2) {
                std::memcpy(Omega.data(), A2.data(), elems * sizeof(Complex));
                matrix_ops::mat_scale_inplace(Omega, N, Complex(h, 0.0));
            }
            else if (opts.magnus_method == MagnusMethod::Classic4 ||
                opts.magnus_method == MagnusMethod::Acc4 ||
                opts.magnus_method == MagnusMethod::Formula314_4) {
                // Для 3-узловой сетки GL [A3, A1] = -(sqrt(15)/5) * h * [Am, Am'],
                // поэтому коэффициент sqrt(15)/36 даёт точный член -(1/12) * h^3 * [Am, Am'].
                const double s15_36 = s15 / 36.0;
                std::memcpy(Omega.data(), A1.data(), elems * sizeof(Complex));
                matrix_ops::mat_scale_inplace(Omega, N, Complex(5.0 * h / 18.0, 0.0));
                matrix_ops::mat_axpy(A2, Omega, N, Complex(4.0 * h / 9.0, 0.0));
                matrix_ops::mat_axpy(A3, Omega, N, Complex(5.0 * h / 18.0, 0.0));

                matrix_ops::commutator_inplace(A3, A1, C1, N);
                matrix_ops::mat_axpy(C1, Omega, N, Complex(s15_36 * h * h, 0.0));
            }
            else {
                // MagnusGL6 (Blanes-Casas-Ros, 6-й порядок)
                std::memcpy(alpha1.data(), A2.data(), elems * sizeof(Complex));
                matrix_ops::mat_scale_inplace(alpha1, N, Complex(h, 0.0));

                std::memcpy(alpha2.data(), A3.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(A1, alpha2, N, Complex(-1.0, 0.0));
                matrix_ops::mat_scale_inplace(alpha2, N, Complex(s15 * h / 3.0, 0.0));

                std::memcpy(alpha3.data(), A3.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(A2, alpha3, N, Complex(-2.0, 0.0));
                matrix_ops::mat_axpy(A1, alpha3, N, Complex(1.0, 0.0));
                matrix_ops::mat_scale_inplace(alpha3, N, Complex(10.0 * h / 3.0, 0.0));

                matrix_ops::commutator_inplace(alpha1, alpha2, C1, N);

                std::memcpy(R1.data(), alpha3.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(C1, R1, N, Complex(-1.0, 0.0));
                matrix_ops::commutator_inplace(alpha2, R1, C2, N);

                std::memcpy(R2.data(), C1.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(alpha3, R2, N, Complex(2.0, 0.0));
                matrix_ops::commutator_inplace(alpha1, R2, inner, N);
                matrix_ops::commutator_inplace(alpha1, inner, C3, N);

                std::memcpy(Omega.data(), alpha1.data(), elems * sizeof(Complex));
                matrix_ops::mat_axpy(alpha3, Omega, N, Complex(1.0 / 12.0, 0.0));
                matrix_ops::mat_axpy(C1, Omega, N, Complex(-1.0 / 12.0, 0.0));
                matrix_ops::mat_axpy(C2, Omega, N, Complex(1.0 / 240.0, 0.0));
                matrix_ops::mat_axpy(C3, Omega, N, Complex(1.0 / 720.0, 0.0));
            }

            CMatrix U_step = apply_expm(Omega, N, opts);
            matrix_ops::matmul(U_step, U_total, U_next, N);
            U_total.swap(U_next);
        }

        std::memcpy(U_out_ptr, U_total.data(), elems * sizeof(Complex));
    }

    void propagate_cpu_raw(
        std::size_t dim,
        const Complex* H0_ptr,
        const Complex* H1_ptr,
        const double* time_nodes,
        std::size_t num_time_nodes,
        const double* modulation_grid,
        std::size_t num_mod_points,
        bool is_harmonic,
        double epsilon,
        double omega,
        const SolverOptions& opts,
        Complex* U_out_ptr
    ) {
        if (num_time_nodes < 2) {
            throw std::invalid_argument("time_nodes must contain at least 2 points (t_start, t_final).");
        }

        const int N = static_cast<int>(dim);
        const std::size_t total_elems = dim * dim;
        const int steps = static_cast<int>(num_time_nodes - 1);
        const double t_start = time_nodes[0];
        const double t_final = time_nodes[num_time_nodes - 1];

        CMatrix H0(H0_ptr, H0_ptr + total_elems);
        CMatrix H1(H1_ptr, H1_ptr + total_elems);
        CMatrix U_res;

#if defined(MAGNUS_HAS_MPI)
        if (opts.backend == BackendType::MpiCpu && is_harmonic) {
            int initialized = 0;
            MPI_Initialized(&initialized);
            if (!initialized) {
                throw std::runtime_error("MPI is not initialized.");
            }
            MPI_Comm comm = (opts.mpi_comm_fortran >= 0)
                ? MPI_Comm_f2c(opts.mpi_comm_fortran)
                : MPI_COMM_WORLD;
            U_res = mpi_magnus_chebyshev_solver(
                H0, H1, N, t_start, t_final, steps, epsilon, omega, opts.chebyshev_degree, comm
            );
            if (!U_res.empty()) {
                std::memcpy(U_out_ptr, U_res.data(), total_elems * sizeof(Complex));
            }
            return;
        }
#endif

        if (is_harmonic) {
            bool is_uniform = true;
            const double uniform_dt = (t_final - t_start) / static_cast<double>(steps);
            for (int s = 0; s < steps; ++s) {
                const double h_s = time_nodes[s + 1] - time_nodes[s];
                if (std::abs(h_s - uniform_dt) > 1e-12 * std::max(1.0, std::abs(uniform_dt))) {
                    is_uniform = false;
                    break;
                }
            }

            // Масштабируем на 1/hbar, так как core-методы (Classic, Acc, Recursive) работают при hbar = 1
            CMatrix H0_eff = H0;
            CMatrix H1_eff = H1;
            if (std::abs(opts.hbar - 1.0) > 1e-15) {
                const Complex inv_hbar(1.0 / opts.hbar, 0.0);
                matrix_ops::mat_scale_inplace(H0_eff, N, inv_hbar);
                matrix_ops::mat_scale_inplace(H1_eff, N, inv_hbar);
            }

            if (opts.magnus_method == MagnusMethod::Formula314_6 &&
                opts.expm_method == ExpmMethod::Chebyshev &&
                is_uniform && omega > 0.0) {
                utils::PeriodicHamiltonian ham;
                ham.dimension = N;
                ham.H0 = H0;
                ham.H1 = H1;
                ham.epsilon = epsilon;
                ham.omega = omega;
                ham.hbar = opts.hbar;

                single_cpu::Formula314ChebyshevOptions f_opts;
                f_opts.chebyshev_degree = opts.chebyshev_degree;
                U_res = single_cpu::solve_formula314_chebyshev(ham, t_start, t_final, steps, f_opts);
            }
            else {
                U_res.assign(total_elems, Complex(0.0, 0.0));
                for (int i = 0; i < N; ++i) {
                    U_res[static_cast<std::size_t>(i) * N + i] = Complex(1.0, 0.0);
                }
                CMatrix U_next(total_elems, Complex(0.0, 0.0));

                if (opts.magnus_method == MagnusMethod::Classic2 ||
                    opts.magnus_method == MagnusMethod::Classic4) {
                    const int classic_order = (opts.magnus_method == MagnusMethod::Classic2) ? 1 : 2;
                    const int sub_steps = 20;

                    for (int s = 0; s < steps; ++s) {
                        const double t_curr = time_nodes[s];
                        const double t_next = time_nodes[s + 1];
                        const double sub_dt = (t_next - t_curr) / sub_steps;
                        CMatrix Omega = magnus_classic(
                            t_curr, t_next, sub_dt, N, H0_eff, H1_eff, epsilon, omega, classic_order
                        );
                        CMatrix U_step = apply_expm(Omega, N, opts);
                        matrix_ops::matmul(U_step, U_res, U_next, N);
                        U_res.swap(U_next);
                    }
                }
                else if (opts.magnus_method == MagnusMethod::Acc4 ||
                    opts.magnus_method == MagnusMethod::Acc6) {
                    const int acc_order = (opts.magnus_method == MagnusMethod::Acc6) ? 6 : 4;
                    const int sub_steps = 20;

                    for (int s = 0; s < steps; ++s) {
                        const double t_curr = time_nodes[s];
                        const double t_next = time_nodes[s + 1];
                        const double sub_dt = (t_next - t_curr) / sub_steps;
                        std::vector<CMatrix> samples = generate_samples(
                            t_curr, t_next, sub_dt, N, H0_eff, H1_eff, epsilon, omega
                        );
                        CMatrix Omega = magnus_ACC(samples, sub_dt, N, acc_order);
                        CMatrix U_step = apply_expm(Omega, N, opts);
                        matrix_ops::matmul(U_step, U_res, U_next, N);
                        U_res.swap(U_next);
                    }
                }
                else if (opts.magnus_method == MagnusMethod::Recursive) {
                    const int max_n = std::max(1, opts.recursive_order);
                    const int sub_steps = 20;

                    for (int s = 0; s < steps; ++s) {
                        const double t_curr = time_nodes[s];
                        const double t_next = time_nodes[s + 1];
                        const double sub_dt = (t_next - t_curr) / sub_steps;
                        CMatrix Omega = magnus_expansion(
                            t_curr, t_next, sub_dt, N, H0_eff, H1_eff, epsilon, omega, max_n
                        );
                        CMatrix U_step = apply_expm(Omega, N, opts);
                        matrix_ops::matmul(U_step, U_res, U_next, N);
                        U_res.swap(U_next);
                    }
                }
                else if (opts.magnus_method == MagnusMethod::Formula314_4) {
                    // Аналитическая формула 3.14, усечённая до 4-го порядка
                    const Complex minus_i(0.0, -1.0);
                    CMatrix A0 = matrix_ops::mat_copy(H0_eff);
                    CMatrix A1 = matrix_ops::mat_copy(H1_eff);
                    matrix_ops::mat_scale_inplace(A0, N, minus_i);
                    matrix_ops::mat_scale_inplace(A1, N, minus_i);

                    CMatrix C1(total_elems, Complex(0.0, 0.0));
                    matrix_ops::commutator_inplace(A0, A1, C1, N);
                    CMatrix Omega(total_elems, Complex(0.0, 0.0));

                    for (int s = 0; s < steps; ++s) {
                        const double t_curr = time_nodes[s];
                        const double t_next = time_nodes[s + 1];
                        const double h = t_next - t_curr;
                        const double h2 = h * h;
                        const double h3 = h2 * h;
                        const double t_half = 0.5 * (t_curr + t_next);

                        const double f_t0 = epsilon * std::cos(omega * t_curr);
                        const double f_t1 = epsilon * std::cos(omega * t_next);
                        const double f_half = epsilon * std::cos(omega * t_half);
                        const double fp_half = -epsilon * omega * std::sin(omega * t_half);
                        const double f2_half = -epsilon * omega * omega * std::cos(omega * t_half);

                        const double c_A1 = h * (0.1 * f_t0 + 0.8 * f_half + 0.1 * f_t1) + (h3 / 60.0) * f2_half;
                        const double c_C1 = -(h2 / 20.0) * (f_t1 - f_t0) - (h3 / 30.0) * fp_half;

                        std::fill(Omega.begin(), Omega.end(), Complex(0.0, 0.0));
                        matrix_ops::mat_axpy(A0, Omega, N, Complex(h, 0.0));
                        matrix_ops::mat_axpy(A1, Omega, N, Complex(c_A1, 0.0));
                        matrix_ops::mat_axpy(C1, Omega, N, Complex(c_C1, 0.0));

                        CMatrix U_step = apply_expm(Omega, N, opts);
                        matrix_ops::matmul(U_step, U_res, U_next, N);
                        U_res.swap(U_next);
                    }
                }
                else {
                    // Formula314_6 на неравномерной сетке или с PadeEigen/Taylor:
                    // напрямую вызываем magnus_3_14 из core/magnus/Formula314Magnus.cpp
                    // (magnus_3_14 ожидает антиэрмитовые генераторы A0 = -i/hbar * H0, A1 = -i/hbar * H1)
                    const Complex minus_i(0.0, -1.0);
                    CMatrix A0 = matrix_ops::mat_copy(H0_eff);
                    CMatrix A1 = matrix_ops::mat_copy(H1_eff);
                    matrix_ops::mat_scale_inplace(A0, N, minus_i);
                    matrix_ops::mat_scale_inplace(A1, N, minus_i);

                    for (int s = 0; s < steps; ++s) {
                        const double t_curr = time_nodes[s];
                        const double t_next = time_nodes[s + 1];
                        const double h = t_next - t_curr;
                        CMatrix Omega = magnus_3_14(t_curr, t_next, h, N, A0, A1, epsilon, omega);
                        CMatrix U_step = apply_expm(Omega, N, opts);
                        matrix_ops::matmul(U_step, U_res, U_next, N);
                        U_res.swap(U_next);
                    }
                }
            }
        }
        else {
            propagate_arbitrary_modulation_mkl(
                N, H0, H1, time_nodes, num_time_nodes, modulation_grid, num_mod_points, opts, U_res
            );
        }

        std::memcpy(U_out_ptr, U_res.data(), total_elems * sizeof(Complex));
    }

} // namespace magnus

#if !defined(MAGNUS_HAS_CUDA)
namespace magnus {

    bool has_cuda_backend() noexcept {
        return false;
    }

    void propagate_cuda_raw(
        std::size_t /*dim*/,
        const Complex* /*d_H0_ptr*/,
        const Complex* /*d_H1_ptr*/,
        double /*t_start*/,
        double /*t_final*/,
        int /*steps*/,
        double /*epsilon*/,
        double /*omega*/,
        const SolverOptions& /*opts*/,
        Complex* /*d_U_out_ptr*/,
        bool /*pointers_on_device*/
    ) {
        throw std::runtime_error("CUDA backend is not available. Recompile with BUILD_SINGLE_GPU=ON.");
    }

    void propagate_cuda_gl3_raw(
        std::size_t /*dim*/,
        const Complex* /*H0_ptr*/,
        const Complex* /*H1_ptr*/,
        const double* /*time_nodes*/,
        std::size_t /*num_time_nodes*/,
        const double* /*gl_nodes*/,
        const SolverOptions& /*opts*/,
        Complex* /*U_out_ptr*/
    ) {
        throw std::runtime_error("CUDA backend is not available. Recompile with BUILD_SINGLE_GPU=ON.");
    }

    std::tuple<double, double, double> benchmark_gpu_events_raw(
        int /*N*/, const Complex* /*H0_ptr*/, const Complex* /*H1_ptr*/,
        double /*t0*/, double /*tf*/, int /*steps*/,
        double /*eps*/, double /*omega*/, int /*cheb_degree*/,
        Complex* /*U_out_ptr*/
    ) {
        throw std::runtime_error("CUDA backend is not available. Recompile with BUILD_SINGLE_GPU=ON.");
    }

    std::pair<std::size_t, std::size_t> get_cuda_mem_info_raw() {
        return { 0, 0 };
    }

} // namespace magnus
#endif