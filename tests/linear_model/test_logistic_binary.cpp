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

#include <cassert>
#include <cmath>
#include <iostream>

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
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row)
    {
        assert(std::abs(probabilities.row(row).sum() - 1.0) < 1e-12);
    }

    std::cout << "Binary logistic regression test prepared.\n";
    return 0;
}
