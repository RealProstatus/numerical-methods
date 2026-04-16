#pragma once

#include "ProblemSetup.h"

#include <vector>

namespace single_cpu {

    // Numerical settings for Formula 3.14 + Chebyshev propagation
    struct Formula314ChebyshevOptions {
        // Number of Chebyshev terms used by expm_cheb
        int chebyshev_degree = 32;
    };

    // Full node-by-node solution U(t_n) for U(t_start)=I
    struct Formula314ChebyshevTrajectory {
        // Time nodes matching matrices in U_nodes
        std::vector<double> time_nodes;

        // Evolution matrices U(t_n,t_start), with U_nodes.front()=I
        std::vector<CMatrix> U_nodes;
    };

    // Compute only the final evolution matrix U(t_final,t_start)
    CMatrix solve_formula314_chebyshev(
        const utils::EvolutionProblem& problem,
        const Formula314ChebyshevOptions& options = Formula314ChebyshevOptions());

    // Prepare the evolution problem and compute U(t_final,t_start)
    CMatrix solve_formula314_chebyshev(
        const utils::PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        const Formula314ChebyshevOptions& options = Formula314ChebyshevOptions());

    // Compute U(t_n,t_start) for every prepared time node
    Formula314ChebyshevTrajectory solve_formula314_chebyshev_trajectory(
        const utils::EvolutionProblem& problem,
        const Formula314ChebyshevOptions& options = Formula314ChebyshevOptions());

    // Prepare the evolution problem and compute U(t_n,t_start) for every time node
    Formula314ChebyshevTrajectory solve_formula314_chebyshev_trajectory(
        const utils::PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        const Formula314ChebyshevOptions& options = Formula314ChebyshevOptions());
}
