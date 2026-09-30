/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #      #   #    #       #   #  #  #
 *   #   #  #   #    #    #####  #   #  #####  ####   #   #  #####
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor  : Yihan Wang
 *
 * ============================================================================
 */
#include "art/linear_model/logistic_regression.h"
#include "art/optim/optimizer.h"

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>

int main()
{
    Eigen::MatrixXd features(6, 1);
    features << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0;
    const Eigen::VectorXd targets =
        (Eigen::VectorXd(6) << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished();

    art::linear_model::LogisticRegression model(0.1, 10000, 1e-8);
    model.fit(features, targets);

    const Eigen::MatrixXd probabilities = model.predict_proba(features);
    const Eigen::VectorXd predictions = model.predict(features);

    assert(model.class_count() == 2);
    assert(model.coefficients().rows() == 2);
    assert(model.coefficients().cols() == 1);
    assert(probabilities.rows() == features.rows());
    assert(probabilities.cols() == 2);
    assert(predictions.size() == targets.size());
    assert(model.score(features, targets) == 1.0);
    assert(!model.loss_history().empty());
    assert(model.loss_history().back() <= model.loss_history().front());
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row)
    {
        assert(std::abs(probabilities.row(row).sum() - 1.0) < 1e-12);
    }

    std::size_t true_positive = 0;
    std::size_t false_positive = 0;
    std::size_t false_negative = 0;
    for (Eigen::Index row = 0; row < targets.size(); ++row)
    {
        const bool actual = targets(row) == 1.0;
        const bool predicted = predictions(row) == 1.0;
        if (actual && predicted) ++true_positive;
        if (!actual && predicted) ++false_positive;
        if (actual && !predicted) ++false_negative;
    }
    const double accuracy = model.score(features, targets);
    const double precision = static_cast<double>(true_positive) /
        static_cast<double>(true_positive + false_positive);
    const double recall = static_cast<double>(true_positive) /
        static_cast<double>(true_positive + false_negative);
    const double f1 = 2.0 * precision * recall / (precision + recall);

    std::cout << std::fixed << std::setprecision(8);
    std::cout << "Training loss: initial=" << model.loss_history().front()
              << ", final=" << model.loss_history().back()
              << ", steps=" << model.loss_history().size() << '\n';
    const std::size_t interval = std::max<std::size_t>(
        1, model.loss_history().size() / 5
    );
    for (std::size_t index = 0; index < model.loss_history().size(); index += interval)
    {
        std::cout << "  step " << index + 1 << ": loss="
                  << model.loss_history()[index] << '\n';
    }
    std::cout << "Metrics: accuracy=" << accuracy
              << ", precision=" << precision
              << ", recall=" << recall
              << ", F1=" << f1 << '\n';

    art::optim::GradientDescent explicit_gradient(0.001);
    art::linear_model::LogisticRegression explicit_model;
    explicit_model.fit(features, targets, explicit_gradient);
    assert(explicit_model.score(features, targets) == 1.0);

    art::optim::NewtonOptimizer newton;
    art::linear_model::LogisticRegression newton_model;
    newton_model.fit(features, targets, newton);
    assert(newton_model.score(features, targets) == 1.0);
    assert(!newton_model.training_history().empty());
    assert(newton_model.training_history().front().gradient_norm >= 0.0);

    art::linear_model::LogisticRegressionOptions regularized_options;
    regularized_options.l2_penalty = 0.1;
    art::linear_model::LogisticRegression regularized_model(
        regularized_options
    );
    regularized_model.fit(features, targets);
    assert(regularized_model.score(features, targets) == 1.0);

    art::linear_model::LogisticRegressionOptions invalid_options;
    invalid_options.l2_penalty = -1.0;
    bool invalid_l2_rejected = false;
    try
    {
        art::linear_model::LogisticRegression invalid_model(invalid_options);
        invalid_model.fit(features, targets);
    }
    catch (const std::invalid_argument&)
    {
        invalid_l2_rejected = true;
    }
    assert(invalid_l2_rejected);

    const Eigen::MatrixXd multiclass_features =
        (Eigen::MatrixXd(3, 1) << -1.0, 0.0, 1.0).finished();
    const Eigen::VectorXd multiclass_targets =
        (Eigen::VectorXd(3) << 0.0, 1.0, 2.0).finished();
    bool newton_rejected = false;
    try
    {
        art::linear_model::LogisticRegression multiclass_model;
        multiclass_model.fit(multiclass_features, multiclass_targets, newton);
    }
    catch (const std::invalid_argument&)
    {
        newton_rejected = true;
    }
    assert(newton_rejected);

    std::cout << "Binary logistic regression test Passed.\n";
    return 0;
}
