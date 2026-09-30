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
#include "art/optim/optimizer.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace art::optim;

class QuadraticProblem final : public OptimizationProblem
{
public:
    double value(const ParameterVector& parameters) const override
    {
        const ParameterVector difference = parameters - target;
        return 0.5 * difference.dot(difference);
    }

    ParameterVector gradient(const ParameterVector& parameters) const override
    {
        return parameters - target;
    }

    HessianMatrix hessian(const ParameterVector&) const override
    {
        return HessianMatrix::Identity(2, 2);
    }

private:
    const ParameterVector target = (ParameterVector(2) << 3.0, -2.0).finished();
};

int main()
{
    QuadraticProblem problem;
    const ParameterVector initial_parameters = ParameterVector::Zero(2);

    OptimizerOptions gradient_options;
    gradient_options.learning_rate = 0.1;
    gradient_options.max_iterations = 1000;
    gradient_options.tolerance = 1e-8;

    GradientDescent gradient_descent;
    const OptimizationResult gradient_result = gradient_descent.optimize(
        problem,
        initial_parameters,
        gradient_options
    );

    assert(gradient_result.converged);
    assert(std::abs(gradient_result.parameters(0) - 3.0) < 1e-5);
    assert(std::abs(gradient_result.parameters(1) + 2.0) < 1e-5);

    NewtonOptimizer newton;
    const OptimizationResult newton_result = newton.optimize(
        problem,
        initial_parameters,
        gradient_options
    );

    assert(newton_result.converged);
    assert(std::abs(newton_result.parameters(0) - 3.0) < 1e-10);
    assert(std::abs(newton_result.parameters(1) + 2.0) < 1e-10);

    std::cout << "Optimizer test passed.\n";
    return 0;
}
