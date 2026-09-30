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
#include "art/model_selection/cross_validate.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace art::model_selection
{
    namespace
    {
        void validate_inputs(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const std::size_t folds,
            const EstimatorFactory& estimator_factory,
            const FitPredictFunction& fit_predict,
            const Scorer& scorer)
        {
            if (features.rows() <= 0 || features.cols() <= 0 || targets.size() <= 0)
            {
                throw std::invalid_argument("cross_validate requires non-empty input");
            }
            if (features.rows() != targets.size())
            {
                throw std::invalid_argument(
                    "cross_validate features and targets have different sample counts");
            }
            if (folds < 2 || folds > static_cast<std::size_t>(features.rows()))
            {
                throw std::invalid_argument("folds must be in [2, sample_count]");
            }
            if (!estimator_factory || !fit_predict || !scorer)
            {
                throw std::invalid_argument(
                    "cross_validate requires factory, fit/predict callback and scorer");
            }
        }

        CrossValidationResult cross_validate_impl(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const EstimatorFactory& estimator_factory,
            const FitPredictFunction& fit_predict,
            const Scorer& scorer,
            const std::size_t folds,
            const std::optional<std::uint64_t> random_seed)
        {
            validate_inputs(
                features, targets, folds, estimator_factory, fit_predict, scorer);

            std::vector<Eigen::Index> indices(features.rows());
            std::iota(indices.begin(), indices.end(), Eigen::Index{0});
            std::mt19937_64 generator(
                random_seed.value_or(std::random_device{}()));
            std::shuffle(indices.begin(), indices.end(), generator);

            CrossValidationResult result;
            result.fold_scores.reserve(folds);
            double score_sum = 0.0;
            double squared_deviation = 0.0;
            const auto sample_count = static_cast<std::size_t>(features.rows());

            for (std::size_t fold = 0; fold < folds; ++fold)
            {
                const auto validation_begin = fold * sample_count / folds;
                const auto validation_end = (fold + 1) * sample_count / folds;
                const auto validation_count = validation_end - validation_begin;
                const auto training_count = sample_count - validation_count;

                base::FeatureInput training_features(training_count, features.cols());
                base::TargetInput training_targets(training_count);
                base::FeatureInput validation_features(validation_count, features.cols());
                base::TargetInput validation_targets(validation_count);

                std::size_t train_row = 0;
                std::size_t validation_row = 0;
                for (std::size_t position = 0; position < sample_count; ++position)
                {
                    const auto source_row = indices[position];
                    if (position >= validation_begin && position < validation_end)
                    {
                        validation_features.row(static_cast<Eigen::Index>(validation_row)) =
                            features.row(source_row);
                        validation_targets(static_cast<Eigen::Index>(validation_row++)) =
                            targets(source_row);
                    }
                    else
                    {
                        training_features.row(static_cast<Eigen::Index>(train_row)) =
                            features.row(source_row);
                        training_targets(static_cast<Eigen::Index>(train_row++)) =
                            targets(source_row);
                    }
                }

                auto estimator = estimator_factory();
                if (!estimator)
                {
                    throw std::invalid_argument(
                        "estimator_factory returned a null estimator");
                }
                const auto predictions = fit_predict(
                    *estimator,
                    training_features,
                    training_targets,
                    validation_features);
                if (predictions.size() != validation_targets.size())
                {
                    throw std::invalid_argument(
                        "fit/predict callback returned the wrong sample count");
                }
                const auto score = scorer(predictions, validation_targets);
                if (!std::isfinite(score))
                {
                    throw std::invalid_argument(
                        "scorer returned a non-finite value");
                }
                result.fold_scores.push_back(score);
                score_sum += score;
            }

            result.mean_score = score_sum / static_cast<double>(folds);
            for (const auto score : result.fold_scores)
            {
                const auto difference = score - result.mean_score;
                squared_deviation += difference * difference;
            }
            result.standard_deviation = std::sqrt(
                squared_deviation / static_cast<double>(folds));
            return result;
        }
    }

    CrossValidationResult cross_validate(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const EstimatorFactory& estimator_factory,
        const FitPredictFunction& fit_predict,
        const Scorer& scorer,
        const std::size_t folds,
        const std::optional<std::uint64_t> random_seed)
    {
        return cross_validate_impl(
            features, targets, estimator_factory, fit_predict, scorer,
            folds, random_seed);
    }

    CrossValidationResult cross_validate(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const PredictorFactory& predictor_factory,
        const Scorer& scorer,
        const std::size_t folds,
        const std::optional<std::uint64_t> random_seed)
    {
        if (!predictor_factory)
        {
            throw std::invalid_argument("predictor_factory must not be empty");
        }
        const EstimatorFactory estimator_factory = [predictor_factory]()
        {
            return std::unique_ptr<base::Estimator>(predictor_factory().release());
        };
        const FitPredictFunction fit_predict = [](
            base::Estimator& estimator,
            const base::FeatureInput& training_features,
            const base::TargetInput& training_targets,
            const base::FeatureInput& validation_features)
        {
            auto* predictor = dynamic_cast<base::Predictor*>(&estimator);
            if (predictor == nullptr)
            {
                throw std::invalid_argument(
                    "predictor_factory did not create a Predictor");
            }
            predictor->fit(training_features, training_targets);
            return predictor->predict(validation_features);
        };
        return cross_validate_impl(
            features, targets, estimator_factory, fit_predict, scorer,
            folds, random_seed);
    }
}
