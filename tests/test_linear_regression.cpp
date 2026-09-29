#include "dataset.h"
#include "linear_regression.h"

#include <cassert>
#include <cmath>
#include <iostream>

using namespace art::data;
using namespace art::linear_model;
using namespace art::optim;

int main()
{
    const Dataset dataset = Dataset::from_csv("data.csv", "label");

    LinearRegression model;
    NewtonOptimizer optimizer;
    OptimizerOptions options;
    options.max_iterations = 20;
    options.tolerance = 1e-10;

    const OptimizationResult newton_result = model.fit(dataset, optimizer, options);

    const Eigen::VectorXd predictions = model.predict(dataset.features);
    const double error = model.mean_squared_error(dataset.features, dataset.labels);

    assert(predictions.size() == 3);
    assert(std::isfinite(error));
    assert(error < 1.0);
    assert(model.parameters().size() == 3);
    assert(newton_result.converged);

    LinearRegression gradient_model;
    GradientDescent gradient_optimizer;
    OptimizerOptions gradient_options;
    gradient_options.learning_rate = 1e-5;
    gradient_options.max_iterations = 200000;
    gradient_options.tolerance = 1e-6;

    const OptimizationResult gradient_result = gradient_model.fit(
        dataset,
        gradient_optimizer,
        gradient_options
    );
    const double gradient_error = gradient_model.mean_squared_error(
        dataset.features,
        dataset.labels
    );

    assert(std::isfinite(gradient_error));
    assert(gradient_error < 1.0);

    std::cout << "Newton iterations: " << newton_result.iterations << '\n';
    std::cout << "Gradient descent iterations: "
              << gradient_result.iterations << '\n';
    std::cout << "Gradient descent converged: "
              << std::boolalpha << gradient_result.converged << '\n';
    std::cout << "Gradient descent final loss: "
              << gradient_result.final_value << '\n';

    std::cout << "Linear regression test passed.\n";
    return 0;
}
