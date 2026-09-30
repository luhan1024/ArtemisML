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
#include "art/model_selection/grid_search.h"
#include "art/model_selection/train_test_split.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace
{
    class MeanPredictor final : public art::base::Predictor
    {
    public:
        explicit MeanPredictor(double offset = 0.0) : offset_(offset) {}

        int fit_count() const noexcept { return fit_count_; }

    protected:
        void do_fit(
            const art::base::FeatureInput&,
            const art::base::TargetInput& targets) override
        {
            if (targets.size() == 0)
            {
                throw std::invalid_argument("empty target");
            }
            prediction_ = targets.mean() + offset_;
            ++fit_count_;
        }

        art::base::PredictionOutput do_predict(
            const art::base::FeatureInput& features) const override
        {
            return art::base::PredictionOutput::Constant(
                features.rows(), prediction_);
        }

    private:
        double offset_;
        double prediction_ = 0.0;
        int fit_count_ = 0;
    };

    art::model_selection::Scorer squared_error =
        [](const art::base::PredictionOutput& predictions,
           const art::base::TargetInput& targets)
        {
            return (predictions - targets).squaredNorm();
        };
}

int main()
{
    const art::base::FeatureInput features =
        (art::base::FeatureInput(8, 1) << 0, 1, 2, 3, 4, 5, 6, 7).finished();
    const art::base::TargetInput targets =
        (art::base::TargetInput(8) << 0, 1, 2, 3, 4, 5, 6, 7).finished();

    const auto first_split = art::model_selection::train_test_split(
        features, targets, 0.25, 42);
    const auto second_split = art::model_selection::train_test_split(
        features, targets, 0.25, 42);
    assert(first_split.features_train.rows() == 6);
    assert(first_split.features_test.rows() == 2);
    assert((first_split.features_train.array() ==
            second_split.features_train.array()).all());
    assert((first_split.features_test.array() ==
            second_split.features_test.array()).all());
    for (Eigen::Index train_row = 0;
         train_row < first_split.features_train.rows(); ++train_row)
    {
        for (Eigen::Index test_row = 0;
             test_row < first_split.features_test.rows(); ++test_row)
        {
            assert(first_split.features_train(train_row, 0) !=
                   first_split.features_test(test_row, 0));
        }
    }

    bool split_rejected = false;
    try
    {
        art::model_selection::train_test_split(features, targets, 1.0, 42);
    }
    catch (const std::invalid_argument&)
    {
        split_rejected = true;
    }
    assert(split_rejected);

    int created = 0;
    const auto predictor_factory = [&created]()
    {
        ++created;
        return std::make_unique<MeanPredictor>();
    };
    const auto cv = art::model_selection::cross_validate(
        features, targets, predictor_factory, squared_error, 4, 42);
    assert(cv.fold_scores.size() == 4);
    assert(created == 4);
    assert(std::isfinite(cv.mean_score));
    assert(std::isfinite(cv.standard_deviation));

    bool folds_rejected = false;
    try
    {
        art::model_selection::cross_validate(
            features, targets, predictor_factory, squared_error, 1, 42);
    }
    catch (const std::invalid_argument&)
    {
        folds_rejected = true;
    }
    assert(folds_rejected);

    int grid_created = 0;
    const art::model_selection::GridSearch search(
        [&grid_created](const art::model_selection::ParameterMap& parameters)
        {
            ++grid_created;
            return std::make_unique<MeanPredictor>(parameters.at("offset"));
        },
        squared_error);
    const auto result = search.fit(
        features,
        targets,
        {{"offset", {-1.0, 0.0, 1.0}}},
        art::model_selection::GridSearchOptions{2, 42, false});
    assert(result.trials.size() == 3);
    assert(result.best_index.has_value());
    assert(result.best_parameters.at("offset") == 0.0);
    assert(grid_created == 6);
    for (const auto& trial : result.trials)
    {
        assert(trial.score.has_value());
        assert(trial.error.empty());
    }

    const art::model_selection::GridSearch failing_search(
        [](const art::model_selection::ParameterMap& parameters)
        {
            if (parameters.at("offset") == 2.0)
            {
                throw std::runtime_error("intentional parameter failure");
            }
            return std::make_unique<MeanPredictor>(parameters.at("offset"));
        },
        squared_error);
    const auto failing_result = failing_search.search(
        features,
        targets,
        {{"offset", {0.0, 2.0}}},
        art::model_selection::GridSearchOptions{2, 42, false});
    assert(failing_result.trials.size() == 2);
    assert(failing_result.trials[0].score.has_value());
    assert(!failing_result.trials[1].score.has_value());
    assert(!failing_result.trials[1].error.empty());

    std::cout << "Model selection tests passed.\n";
    return 0;
}
