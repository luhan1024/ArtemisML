#include "optimizer.h"

#include <cmath>
#include <stdexcept>

namespace art::optim
{
namespace
{
    void validateOptions(const OptimizerOptions& options)
    {
        if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0)
        {
            throw std::invalid_argument("Learning rate must be positive and finite");
        }

        if (options.max_iterations == 0)
        {
            throw std::invalid_argument("Maximum iterations must be greater than zero");
        }

        if (!std::isfinite(options.tolerance) || options.tolerance < 0.0)
        {
            throw std::invalid_argument("Tolerance must be non-negative and finite");
        }
    }

    void validateValue(double value)
    {
        if (!std::isfinite(value))
        {
            throw std::runtime_error("Objective value is not finite");
        }
    }

    void validateGradient(
        const ParameterVector& parameters,
        const ParameterVector& gradient
    )
    {
        if (gradient.size() != parameters.size())
        {
            throw std::runtime_error("Gradient dimension does not match parameter dimension");
        }

        if (!gradient.allFinite())
        {
            throw std::runtime_error("Gradient contains a non-finite value");
        }
    }
}

HessianMatrix OptimizationProblem::hessian(const ParameterVector&) const
{
    throw std::logic_error("This optimization problem does not provide a Hessian");
}

OptimizationResult GradientDescent::optimize(
    const OptimizationProblem& problem,
    const ParameterVector& initial_parameters,
    const OptimizerOptions& options
) const
{
    validateOptions(options);

    if (initial_parameters.size() == 0)
    {
        throw std::invalid_argument("Initial parameter vector must not be empty");
    }

    ParameterVector parameters = initial_parameters;

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        const double current_value = problem.value(parameters);
        validateValue(current_value);

        const ParameterVector current_gradient = problem.gradient(parameters);
        validateGradient(parameters, current_gradient);

        if (current_gradient.norm() <= options.tolerance)
        {
            return {parameters, current_value, iteration, true};
        }

        parameters -= options.learning_rate * current_gradient;
    }

    const double final_value = problem.value(parameters);
    validateValue(final_value);

    return {parameters, final_value, options.max_iterations, false};
}

OptimizationResult NewtonOptimizer::optimize(
    const OptimizationProblem& problem,
    const ParameterVector& initial_parameters,
    const OptimizerOptions& options
) const
{
    validateOptions(options);

    if (initial_parameters.size() == 0)
    {
        throw std::invalid_argument("Initial parameter vector must not be empty");
    }

    ParameterVector parameters = initial_parameters;

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        const double current_value = problem.value(parameters);
        validateValue(current_value);

        const ParameterVector current_gradient = problem.gradient(parameters);
        validateGradient(parameters, current_gradient);

        if (current_gradient.norm() <= options.tolerance)
        {
            return {parameters, current_value, iteration, true};
        }

        const HessianMatrix current_hessian = problem.hessian(parameters);

        if (current_hessian.rows() != parameters.size() ||
            current_hessian.cols() != parameters.size())
        {
            throw std::runtime_error("Hessian dimension does not match parameter dimension");
        }

        Eigen::LDLT<HessianMatrix> solver(current_hessian);

        if (solver.info() != Eigen::Success)
        {
            throw std::runtime_error("Failed to factorize Hessian");
        }

        const ParameterVector step = solver.solve(current_gradient);

        if (solver.info() != Eigen::Success || !step.allFinite())
        {
            throw std::runtime_error("Failed to solve Newton step");
        }

        parameters -= step;
    }

    const double final_value = problem.value(parameters);
    validateValue(final_value);

    return {parameters, final_value, options.max_iterations, false};
}
}
