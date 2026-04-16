#include "Formula314ChebyshevSolver.h"

#include "ChebyshevExpm.h"
#include "Formula314Magnus.h"
#include "MatrixOperations.h"
#include "Utils.h"

namespace single_cpu {
    namespace {
        CMatrix make_generator_component(const CMatrix& H, int N, double hbar) {
            CMatrix A = matrix_ops::mat_copy(H);
            matrix_ops::mat_scale_inplace(A, N, complexd(0.0, -1.0 / hbar));
            return A;
        }

        CMatrix compute_local_propagator(
            double t_left,
            double t_right,
            int N,
            const CMatrix& A0,
            const CMatrix& A1,
            double epsilon,
            double omega,
            int chebyshev_degree)
        {
            const double h = t_right - t_left;
            CMatrix Omega = magnus_3_14(t_left, t_right, h, N, A0, A1, epsilon, omega);
            return matrix_ops::expm_cheb(Omega, N, chebyshev_degree);
        }
    }

    CMatrix solve_formula314_chebyshev(
        const utils::EvolutionProblem& problem,
        const Formula314ChebyshevOptions& options)
    {
        const int N = problem.hamiltonian.dimension;
        const CMatrix A0 = make_generator_component(problem.hamiltonian.H0, N, problem.hamiltonian.hbar);
        const CMatrix A1 = make_generator_component(problem.hamiltonian.H1, N, problem.hamiltonian.hbar);

        CMatrix U_total = utils::eye(N);
        CMatrix product_buffer(static_cast<size_t>(N) * static_cast<size_t>(N), complexd(0.0, 0.0));

        for (int step = 0; step < problem.steps; ++step) {
            const CMatrix U_local = compute_local_propagator(
                problem.time_nodes[step],
                problem.time_nodes[step + 1],
                N,
                A0,
                A1,
                problem.hamiltonian.epsilon,
                problem.hamiltonian.omega,
                options.chebyshev_degree);

            matrix_ops::matmul(U_local, U_total, product_buffer, N);
            U_total.swap(product_buffer);
        }

        return U_total;
    }

    CMatrix solve_formula314_chebyshev(
        const utils::PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        const Formula314ChebyshevOptions& options)
    {
        utils::EvolutionProblem problem = utils::prepare_evolution_problem(
            hamiltonian,
            t_start,
            t_final,
            steps);

        return solve_formula314_chebyshev(problem, options);
    }

    Formula314ChebyshevTrajectory solve_formula314_chebyshev_trajectory(
        const utils::EvolutionProblem& problem,
        const Formula314ChebyshevOptions& options)
    {
        Formula314ChebyshevTrajectory trajectory;
        trajectory.time_nodes = problem.time_nodes;
        trajectory.U_nodes.reserve(static_cast<size_t>(problem.steps) + 1);

        const int N = problem.hamiltonian.dimension;
        const CMatrix A0 = make_generator_component(problem.hamiltonian.H0, N, problem.hamiltonian.hbar);
        const CMatrix A1 = make_generator_component(problem.hamiltonian.H1, N, problem.hamiltonian.hbar);

        CMatrix U_total = utils::eye(N);
        CMatrix product_buffer(static_cast<size_t>(N) * static_cast<size_t>(N), complexd(0.0, 0.0));

        trajectory.U_nodes.push_back(U_total);

        for (int step = 0; step < problem.steps; ++step) {
            const CMatrix U_local = compute_local_propagator(
                problem.time_nodes[step],
                problem.time_nodes[step + 1],
                N,
                A0,
                A1,
                problem.hamiltonian.epsilon,
                problem.hamiltonian.omega,
                options.chebyshev_degree);

            matrix_ops::matmul(U_local, U_total, product_buffer, N);
            U_total.swap(product_buffer);
            trajectory.U_nodes.push_back(U_total);
        }

        return trajectory;
    }

    Formula314ChebyshevTrajectory solve_formula314_chebyshev_trajectory(
        const utils::PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        const Formula314ChebyshevOptions& options)
    {
        utils::EvolutionProblem problem = utils::prepare_evolution_problem(
            hamiltonian,
            t_start,
            t_final,
            steps);

        return solve_formula314_chebyshev_trajectory(problem, options);
    }
}
