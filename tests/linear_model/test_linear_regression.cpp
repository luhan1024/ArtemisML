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
#include "art/data/dataset.h"
#include "art/linear_model/linear_regression.h"
#include "art/linear_model/ridge.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

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

    // The derived overload must not hide the base protocol.
    LinearRegression base_model;
    art::base::Regressor& base_regressor = base_model;
    base_regressor.fit(dataset.features, dataset.labels);
    assert(base_model.is_fitted());
    assert(std::isfinite(base_regressor.score(dataset.features, dataset.labels)));

    Ridge ridge(0.1);
    art::base::Regressor& base_ridge = ridge;
    base_ridge.fit(dataset.features, dataset.labels);
    assert(ridge.is_fitted());
    assert(ridge.parameters().size() == 3);
    assert(std::isfinite(base_ridge.score(dataset.features, dataset.labels)));

    bool dimension_rejected = false;
    try
    {
        base_model.predict(Eigen::MatrixXd::Zero(1, 1));
    }
    catch (const std::invalid_argument&)
    {
        dimension_rejected = true;
    }
    assert(dimension_rejected);

    bool empty_mse_rejected = false;
    try
    {
        model.mean_squared_error(Eigen::MatrixXd::Zero(0, 2), Eigen::VectorXd{});
    }
    catch (const std::invalid_argument&)
    {
        empty_mse_rejected = true;
    }
    assert(empty_mse_rejected);

    Dataset invalid_dataset;
    invalid_dataset.features = Eigen::MatrixXd::Zero(0, 2);
    invalid_dataset.labels = Eigen::VectorXd{};
    bool empty_fit_rejected = false;
    try
    {
        base_model.fit(invalid_dataset.features, invalid_dataset.labels);
    }
    catch (const std::invalid_argument&)
    {
        empty_fit_rejected = true;
    }
    assert(empty_fit_rejected);
    assert(!base_model.is_fitted());

    bool stale_parameters_rejected = false;
    try
    {
        base_model.parameters();
    }
    catch (const std::logic_error&)
    {
        stale_parameters_rejected = true;
    }
    assert(stale_parameters_rejected);

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
