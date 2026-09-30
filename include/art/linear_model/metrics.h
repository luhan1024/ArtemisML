/*
 * ============================================================================
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor : Yihan Wang
 *
 * ============================================================================
 */
#pragma once

#include <variant>

namespace art::linear_model
{
    struct RegressionMetrics
    {
        double mean_squared_error = 0.0;
        double mean_absolute_error = 0.0;
        double r2_score = 0.0;
    };

    struct ClassificationMetrics
    {
        double accuracy = 0.0;
        double precision = 0.0;
        double recall = 0.0;
        double f1_score = 0.0;
    };

    using EvaluationResult = std::variant<
        ClassificationMetrics,
        RegressionMetrics
    >;
}
