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

#include <Eigen/Dense>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>

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
    std::cout << "Binary logistic regression test Passed.\n";
    return 0;
}
