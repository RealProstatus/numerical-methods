#pragma once

#include "Utils.h"

#include <string>
#include <vector>

namespace utils {

    // Periodic Hermitian Hamiltonian H(t)=H0+epsilon*cos(omega*t)*H1
    struct PeriodicHamiltonian {
        // Matrix dimension N for dense N x N row-major matrices
        int dimension = 0;

        // Static Hermitian part H0
        CMatrix H0;

        // Hermitian modulation matrix H1
        CMatrix H1;

        // Dimensionless modulation amplitude epsilon
        double epsilon = 1.0;

        // Angular frequency omega
        double omega = 0.0;

        // Reduced Planck constant in dU/dt=-i*H(t)*U/hbar
        double hbar = 1.0;

        // Optional problem name for diagnostics and output files
        string name = "periodic_hamiltonian";

        // Validate dimensions, scalar parameters and Hermiticity of H0/H1
        void validate(double hermitian_tol = 1e-10) const;

        // Return one physical period 2*pi/omega
        double period() const;

        // Evaluate f(t)=epsilon*cos(omega*t)
        double modulation(double t) const;

        // Build the Hermitian Hamiltonian H(t)
        CMatrix hamiltonian_at(double t) const;

        // Adapter for current Magnus backends that expect A(t)=-i*H(t)/hbar
        CMatrix generator_at(double t) const;
    };

    // Prepared evolution problem for U(t), U(t_start)=I
    struct EvolutionProblem {
        // Hamiltonian in dU/dt=-i*H(t)*U/hbar
        PeriodicHamiltonian hamiltonian;

        // Left boundary t_start of the evolution interval
        double t_start = 0.0;

        // Right boundary t_final of the evolution interval
        double t_final = 0.0;

        // Number of uniform time steps
        int steps = 0;

        // Materialized grid nodes t_0,...,t_steps
        vector<double> time_nodes;

        // Build a prepared problem on one period [t_start,t_start+2*pi/omega]
        static EvolutionProblem one_period(
            const PeriodicHamiltonian& hamiltonian,
            double t_start,
            int steps,
            double hermitian_tol = 1e-10);

        // Validate Hamiltonian, time interval and prepared grid if it exists
        void validate(double hermitian_tol = 1e-10) const;

        // Validate input and materialize the uniform time grid
        void prepare(double hermitian_tol = 1e-10);

        // Return the uniform step h=(t_final-t_start)/steps
        double step_size() const;

        // Return grid node t_index
        double node(int index) const;

        // Return the left endpoint t_n of a step
        double left(int step_index) const;

        // Return the right endpoint t_{n+1} of a step
        double right(int step_index) const;

        // Return the midpoint t_n+h/2 of a step
        double midpoint(int step_index) const;

        // Return a compact diagnostic summary
        string summary() const;
    };

    // Build and prepare a uniform-grid evolution problem for U(t)
    EvolutionProblem prepare_evolution_problem(
        const PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        double hermitian_tol = 1e-10);
}
