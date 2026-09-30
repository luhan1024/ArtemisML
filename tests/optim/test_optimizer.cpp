/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
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
    OptimizerState gradient_state;
    const OptimizationStepResult gradient_step = gradient_descent.step(
        problem,
        initial_parameters,
        gradient_state,
        gradient_options
    );

    assert(gradient_state.step == 1);
    assert(std::abs(gradient_step.parameters(0) - 0.3) < 1e-12);
    assert(std::abs(gradient_step.parameters(1) + 0.2) < 1e-12);
    assert(std::abs(gradient_step.step_norm - std::sqrt(0.13)) < 1e-12);

    NewtonOptimizer newton;
    OptimizerState newton_state;
    OptimizerOptions newton_options = gradient_options;
    newton_options.learning_rate = 1.0;
    const OptimizationStepResult newton_step = newton.step(
        problem,
        initial_parameters,
        newton_state,
        newton_options
    );

    assert(newton_state.step == 1);
    assert(std::abs(newton_step.parameters(0) - 3.0) < 1e-10);
    assert(std::abs(newton_step.parameters(1) + 2.0) < 1e-10);

    std::cout << "Optimizer test passed.\n";
    return 0;
}
