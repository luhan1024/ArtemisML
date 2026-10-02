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
    // Compatibility field for the default-constructed GradientDescent only.
    // New code should configure GradientDescent with GradientDescent(lr).
    // Explicitly configured GradientDescent and NewtonOptimizer ignore this
    // field. It is retained so existing step(problem, parameters, state,
    // options) callers continue to compile during migration.
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
    // Compatibility constructor: uses options.learning_rate at step time.
    GradientDescent() noexcept;

    // Preferred constructor: the learning rate belongs to this optimizer and
    // options.learning_rate is ignored.
    explicit GradientDescent(double learning_rate) noexcept;

    OptimizationStepResult step(
        const OptimizationProblem& problem,
        const ParameterVector& parameters,
        OptimizerState& state,
        const OptimizerOptions& options
    ) const override;

private:
    double learning_rate_;
    bool use_legacy_option_learning_rate_;
};

class NewtonOptimizer final : public Optimizer
{
public:
    // Newton uses the full Hessian direction and has no learning-rate
    // parameter. OptimizerOptions::learning_rate is ignored.
    OptimizationStepResult step(
        const OptimizationProblem& problem,
        const ParameterVector& parameters,
        OptimizerState& state,
        const OptimizerOptions& options
    ) const override;
};
}
