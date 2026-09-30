/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
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
    validateParameters(parameters);

    const double current_value = problem.value(parameters);
    validateValue(current_value);

    const ParameterVector current_gradient = problem.gradient(parameters);
    validateGradient(parameters, current_gradient);

    const ParameterVector next_parameters =
        parameters - options.learning_rate * current_gradient;
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

    Eigen::FullPivLU<HessianMatrix> rank_check(current_hessian);
    if (rank_check.rank() < current_hessian.rows())
    {
        throw std::runtime_error("Hessian is singular");
    }

    Eigen::LDLT<HessianMatrix> solver(current_hessian);
    if (solver.info() != Eigen::Success)
    {
        throw std::runtime_error("Failed to factorize Hessian");
    }

    const ParameterVector direction = solver.solve(current_gradient);
    if (solver.info() != Eigen::Success || !direction.allFinite())
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
