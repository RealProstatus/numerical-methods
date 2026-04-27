#include "ProblemSetup.h"

#include "MatrixOperations.h"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace utils {
    namespace {
        constexpr double kPi = 3.14159265358979323846;

        void validate_square_matrix_size(const CMatrix& matrix, int N, const string& name) {
            if (N <= 0) {
                throw std::invalid_argument("Matrix dimension must be positive for " + name);
            }

            const size_t expected_size = static_cast<size_t>(N) * static_cast<size_t>(N);
            if (matrix.size() != expected_size) {
                std::ostringstream out;
                out << name << " has size " << matrix.size()
                    << ", expected " << expected_size << " for N=" << N;
                throw std::invalid_argument(out.str());
            }
        }

        double max_hermitian_defect(const CMatrix& matrix, int N) {
            const CMatrix matrix_dagger = matrix_ops::dagger(matrix, N);
            return matrix_ops::max_element_diff(matrix, matrix_dagger, N);
        }

        void validate_hermitian(const CMatrix& matrix, int N, const string& name, double tol) {
            validate_square_matrix_size(matrix, N, name);

            const double defect = max_hermitian_defect(matrix, N);
            if (defect > tol) {
                std::ostringstream out;
                out << name << " is not Hermitian: max defect = " << defect
                    << ", tolerance = " << tol;
                throw std::invalid_argument(out.str());
            }
        }

        void validate_time_interval(double t_start, double t_final, int steps) {
            if (steps <= 0) {
                throw std::invalid_argument("time grid steps must be positive");
            }

            if (!(t_final > t_start)) {
                throw std::invalid_argument("time grid requires t_final > t_start");
            }
        }

        void materialize_uniform_grid(EvolutionProblem& problem) {
            const double h = problem.step_size();
            problem.time_nodes.clear();
            problem.time_nodes.reserve(static_cast<size_t>(problem.steps) + 1);

            for (int n = 0; n <= problem.steps; ++n) {
                problem.time_nodes.push_back(problem.t_start + static_cast<double>(n) * h);
            }

            problem.time_nodes.back() = problem.t_final;
        }
    }

    void PeriodicHamiltonian::validate(double hermitian_tol) const {
        if (omega <= 0.0) {
            throw std::invalid_argument("omega must be positive");
        }

        if (hbar <= 0.0) {
            throw std::invalid_argument("hbar must be positive");
        }

        validate_hermitian(H0, dimension, "H0", hermitian_tol);
        validate_hermitian(H1, dimension, "H1", hermitian_tol);
    }

    double PeriodicHamiltonian::period() const {
        return 2.0 * kPi / omega;
    }

    double PeriodicHamiltonian::modulation(double t) const {
        return epsilon * std::cos(omega * t);
    }

    CMatrix PeriodicHamiltonian::hamiltonian_at(double t) const {
        CMatrix result = matrix_ops::mat_copy(H0);
        matrix_ops::mat_axpy(H1, result, dimension, complexd(modulation(t), 0.0));
        return result;
    }

    CMatrix PeriodicHamiltonian::generator_at(double t) const {
        CMatrix result = hamiltonian_at(t);
        matrix_ops::mat_scale_inplace(result, dimension, complexd(0.0, -1.0 / hbar));
        return result;
    }

    EvolutionProblem EvolutionProblem::one_period(
        const PeriodicHamiltonian& hamiltonian,
        double t_start,
        int steps,
        double hermitian_tol)
    {
        hamiltonian.validate(hermitian_tol);
        EvolutionProblem problem;
        problem.hamiltonian = hamiltonian;
        problem.t_start = t_start;
        problem.t_final = t_start + hamiltonian.period();
        problem.steps = steps;

        validate_time_interval(problem.t_start, problem.t_final, problem.steps);
        materialize_uniform_grid(problem);
        return problem;
    }

    void EvolutionProblem::validate(double hermitian_tol) const {
        hamiltonian.validate(hermitian_tol);
        validate_time_interval(t_start, t_final, steps);
    }

    void EvolutionProblem::prepare(double hermitian_tol) {
        hamiltonian.validate(hermitian_tol);
        validate_time_interval(t_start, t_final, steps);
        materialize_uniform_grid(*this);
    }

    double EvolutionProblem::step_size() const {
        return (t_final - t_start) / static_cast<double>(steps);
    }

    double EvolutionProblem::node(int index) const {
        return time_nodes[index];
    }

    double EvolutionProblem::left(int step_index) const {
        return time_nodes[step_index];
    }

    double EvolutionProblem::right(int step_index) const {
        return time_nodes[step_index + 1];
    }

    double EvolutionProblem::midpoint(int step_index) const {
        return 0.5 * (time_nodes[step_index] + time_nodes[step_index + 1]);
    }

    string EvolutionProblem::summary() const {
        std::ostringstream out;
        out << "EvolutionProblem{name=" << hamiltonian.name
            << ", N=" << hamiltonian.dimension
            << ", epsilon=" << hamiltonian.epsilon
            << ", omega=" << hamiltonian.omega
            << ", hbar=" << hamiltonian.hbar
            << ", t_start=" << t_start
            << ", t_final=" << t_final
            << ", steps=" << steps
            << ", dt=" << step_size()
            << "}";
        return out.str();
    }

    EvolutionProblem prepare_evolution_problem(
        const PeriodicHamiltonian& hamiltonian,
        double t_start,
        double t_final,
        int steps,
        double hermitian_tol)
    {
        EvolutionProblem problem;
        problem.hamiltonian = hamiltonian;
        problem.t_start = t_start;
        problem.t_final = t_final;
        problem.steps = steps;
        problem.prepare(hermitian_tol);
        return problem;
    }
}
