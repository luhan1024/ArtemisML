/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #      #   #    #       #   #   #  #
 *   #   #  #   #    #    #####  #   #  #####  ####   #   #  #####
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor  : Yihan Wang
 *
 * ============================================================================
 */
#include "art/optim/optimizer.h"

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

    void validateParameters(const ParameterVector& parameters)
    {
        if (parameters.size() == 0)
        {
            throw std::invalid_argument("Initial parameter vector must not be empty");
        }

        if (!parameters.allFinite())
        {
            throw std::invalid_argument("Initial parameter vector contains a non-finite value");
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

    void validateHessian(
        const ParameterVector& parameters,
        const HessianMatrix& hessian
    )
    {
        if (hessian.rows() != parameters.size() ||
            hessian.cols() != parameters.size())
        {
            throw std::runtime_error("Hessian dimension does not match parameter dimension");
        }

        if (!hessian.allFinite())
        {
            throw std::runtime_error("Hessian contains a non-finite value");
        }
    }
}

HessianMatrix OptimizationProblem::hessian(const ParameterVector&) const
{
    throw std::logic_error("This optimization problem does not provide a Hessian");
}

OptimizationStepResult GradientDescent::step(
    const OptimizationProblem& problem,
    const ParameterVector& parameters,
    OptimizerState& state,
    const OptimizerOptions& options
) const
{
    validateOptions(options);
    const double learning_rate = use_legacy_option_learning_rate_
        ? options.learning_rate : learning_rate_;
    if (!std::isfinite(learning_rate) || learning_rate <= 0.0)
    {
        throw std::invalid_argument("Gradient descent learning rate must be positive and finite");
    }
    validateParameters(parameters);

    const double current_value = problem.value(parameters);
    validateValue(current_value);

    const ParameterVector current_gradient = problem.gradient(parameters);
    validateGradient(parameters, current_gradient);

    const ParameterVector next_parameters =
        parameters - learning_rate * current_gradient;
    if (!next_parameters.allFinite())
    {
        throw std::runtime_error("Gradient descent produced a non-finite parameter");
    }

    ++state.step;
    return {
        next_parameters,
        current_gradient,
        current_value,
        (next_parameters - parameters).norm()
    };
}

GradientDescent::GradientDescent() noexcept
    : learning_rate_(0.01), use_legacy_option_learning_rate_(true)
{
}

GradientDescent::GradientDescent(double learning_rate) noexcept
    : learning_rate_(learning_rate), use_legacy_option_learning_rate_(false)
{
}

OptimizationStepResult NewtonOptimizer::step(
    const OptimizationProblem& problem,
    const ParameterVector& parameters,
    OptimizerState& state,
    const OptimizerOptions& options
) const
{
    validateOptions(options);
    validateParameters(parameters);

    const double current_value = problem.value(parameters);
    validateValue(current_value);

    const ParameterVector current_gradient = problem.gradient(parameters);
    validateGradient(parameters, current_gradient);

    const HessianMatrix current_hessian = problem.hessian(parameters);
    validateHessian(parameters, current_hessian);

    // A rank-deficient Hessian still has a well-defined least-norm Newton
    // direction. CompleteOrthogonalDecomposition avoids rejecting valid
    // logistic problems after sigmoid curvature becomes numerically small.
    Eigen::CompleteOrthogonalDecomposition<HessianMatrix> solver(current_hessian);
    const ParameterVector direction = solver.solve(current_gradient);
    if (!direction.allFinite())
    {
        throw std::runtime_error("Failed to solve Newton step");
    }

    // Preserve the original Newton semantics: one call takes the full
    // Newton direction.  learning_rate is the gradient-descent step size;
    // using it here would turn the former full Newton update into a damped
    // update and break the existing convergence contract.
    const ParameterVector next_parameters = parameters - direction;
    if (!next_parameters.allFinite())
    {
        throw std::runtime_error("Newton method produced a non-finite parameter");
    }

    ++state.step;
    return {
        next_parameters,
        current_gradient,
        current_value,
        (next_parameters - parameters).norm()
    };
}
}
