/*
 * +----------------------------------------------------------------------------+
 * |                                                                            |
 * | .###.  ####.  #####  #####  #...#  #####  .####  #...#  #....              |
 * | #...#  #...#  ..#..  #....  ##.##  ..#..  #....  ##.##  #....              |
 * | #####  ####.  ..#..  ####.  #.#.#  ..#..  .###.  #.#.#  #....              |
 * | #...#  #.#..  ..#..  #....  #...#  ..#..  ....#  #...#  #....              |
 * | #...#  #..##  ..#..  #####  #...#  #####  ####.  #...#  #####              |
 * |                                                                            |
 * | Han Lu & Yihan Wang                                                        |
 * +----------------------------------------------------------------------------+
 */
#pragma once

#include <Eigen/Dense>

#include <cstddef>
#include <functional>

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

class Optimizer
{
public:
    virtual ~Optimizer() = default;

    virtual OptimizationResult optimize(
        const OptimizationProblem& problem,
        const ParameterVector& initial_parameters,
        const OptimizerOptions& options
    ) const = 0;
};

class GradientDescent final : public Optimizer
{
public:
    OptimizationResult optimize(
        const OptimizationProblem& problem,
        const ParameterVector& initial_parameters,
        const OptimizerOptions& options
    ) const override;
};

class NewtonOptimizer final : public Optimizer
{
public:
    OptimizationResult optimize(
        const OptimizationProblem& problem,
        const ParameterVector& initial_parameters,
        const OptimizerOptions& options
    ) const override;
};
}
