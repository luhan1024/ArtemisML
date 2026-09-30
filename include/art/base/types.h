/*
 * ============================================================================
 *                         A R T E M I S M L
 *                         A R T E M I S M L
 * ============================================================================
 * Project: ArtemisML - C++ machine learning library
 * Main contributors: Han Lu & Yihan Wang
 * ============================================================================
 */
#pragma once

#include <Eigen/Dense>

namespace art::base
{
    // The current numerical backend represents tabular features and vector
    // targets with Eigen. Keeping these names in base gives higher layers a
    // stable protocol without coupling the protocol to a concrete estimator.
    using FeatureInput = Eigen::MatrixXd;
    using DataInput = Eigen::MatrixXd;
    using TargetInput = Eigen::VectorXd;
    using PredictionOutput = Eigen::VectorXd;
}
