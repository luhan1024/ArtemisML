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
#pragma once

#include <Eigen/Dense>

#include <cstddef>
namespace art::optim
{
using ParameterVector = Eigen::VectorXd;
using HessianMatrix = Eigen::MatrixXd;

class OptimizationProblem
{
public:
    virtual ~OptimizationProblem() = default;

    virtual double value(const ParameterVector& parameters) const = 0;
    virtual ParameterVector gradient(const ParameterVector& parameters) const = 0;

    virtual HessianMatrix hessian(const ParameterVector& parameters) const;
};

struct OptimizerOptions
{
    double learning_rate = 0.01;
    std::size_t max_iterations = 1000;
    double tolerance = 1e-6;
};

struct OptimizationResult
{
    ParameterVector parameters;
    double final_value = 0.0;
    std::size_t iterations = 0;
    bool converged = false;
};

struct OptimizerState
{
    // Number of successfully completed calls to Optimizer::step().
    std::size_t step = 0;
};

struct OptimizationStepResult
{
    // The candidate produced by this step.
    ParameterVector parameters;
    // Gradient and objective evaluated at the input parameters.
    ParameterVector gradient;
    double objective_value = 0.0;
    // Norm of candidate.parameters - input parameters.
    double step_norm = 0.0;
};

class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual OptimizationStepResult step(
        const OptimizationProblem& problem,
        const ParameterVector& parameters,
        OptimizerState& state,
        const OptimizerOptions& options
    ) const = 0;
};

class GradientDescent final : public Optimizer
{
public:
    OptimizationStepResult step(
        const OptimizationProblem& problem,
        const ParameterVector& parameters,
        OptimizerState& state,
        const OptimizerOptions& options
    ) const override;
};

class NewtonOptimizer final : public Optimizer
{
public:
    OptimizationStepResult step(
        const OptimizationProblem& problem,
        const ParameterVector& parameters,
        OptimizerState& state,
        const OptimizerOptions& options
    ) const override;
};
}
