/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#pragma once

#include "art/model_selection/cross_validate.h"

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace art::model_selection
{
    using ParameterMap = std::map<std::string, double>;
    using ParameterGrid = std::map<std::string, std::vector<double>>;
    using ParameterizedEstimatorFactory =
        std::function<std::unique_ptr<base::Estimator>(const ParameterMap&)>;

    struct GridSearchOptions
    {
        std::size_t folds = 5;
        std::optional<std::uint64_t> random_seed = std::nullopt;
        bool greater_is_better = true;
    };

    struct GridSearchTrial
    {
        ParameterMap parameters;
        std::optional<double> score;
        std::string error;
    };

    struct GridSearchResult
    {
        std::vector<GridSearchTrial> trials;
        std::optional<std::size_t> best_index;
        ParameterMap best_parameters;
        double best_score = 0.0;
    };

    class GridSearch final
    {
    public:
        GridSearch(
            ParameterizedEstimatorFactory estimator_factory,
            FitPredictFunction fit_predict,
            Scorer scorer
        );

        GridSearch(
            std::function<std::unique_ptr<base::Predictor>(const ParameterMap&)>
                predictor_factory,
            Scorer scorer
        );

        GridSearchResult search(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const ParameterGrid& parameter_grid,
            const GridSearchOptions& options = {}
        ) const;

        GridSearchResult fit(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const ParameterGrid& parameter_grid,
            const GridSearchOptions& options = {}
        ) const;

    private:
        ParameterizedEstimatorFactory estimator_factory_;
        FitPredictFunction fit_predict_;
        Scorer scorer_;
    };
}
