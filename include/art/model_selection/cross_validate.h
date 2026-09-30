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
#pragma once

#include "art/base/predictor.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <vector>

namespace art::model_selection
{
    using EstimatorFactory =
        std::function<std::unique_ptr<base::Estimator>()>;
    using PredictorFactory =
        std::function<std::unique_ptr<base::Predictor>()>;

    // This callback deliberately owns the fit/predict protocol. It permits
    // adapters for Pipeline without making model_selection depend on pipeline.
    using FitPredictFunction = std::function<
        base::PredictionOutput(
            base::Estimator&,
            const base::FeatureInput&,
            const base::TargetInput&,
            const base::FeatureInput&)>;

    using Scorer = std::function<double(
        const base::PredictionOutput&,
        const base::TargetInput&)>;

    struct CrossValidationResult
    {
        std::vector<double> fold_scores;
        double mean_score = 0.0;
        double standard_deviation = 0.0;
    };

    CrossValidationResult cross_validate(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const EstimatorFactory& estimator_factory,
        const FitPredictFunction& fit_predict,
        const Scorer& scorer,
        std::size_t folds = 5,
        std::optional<std::uint64_t> random_seed = std::nullopt
    );

    CrossValidationResult cross_validate(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const PredictorFactory& predictor_factory,
        const Scorer& scorer,
        std::size_t folds = 5,
        std::optional<std::uint64_t> random_seed = std::nullopt
    );
}
