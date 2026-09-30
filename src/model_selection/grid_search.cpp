/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#include "art/model_selection/grid_search.h"

#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

namespace art::model_selection
{
    namespace
    {
        void enumerate_grid(
            const ParameterGrid& grid,
            const std::vector<std::string>& names,
            const std::size_t position,
            ParameterMap& current,
            std::vector<ParameterMap>& combinations)
        {
            if (position == names.size())
            {
                combinations.push_back(current);
                return;
            }
            const auto& values = grid.at(names[position]);
            for (const auto value : values)
            {
                current[names[position]] = value;
                enumerate_grid(grid, names, position + 1, current, combinations);
            }
        }

        std::vector<ParameterMap> combinations_for(const ParameterGrid& grid)
        {
            std::vector<std::string> names;
            names.reserve(grid.size());
            for (const auto& [name, values] : grid)
            {
                if (name.empty() || values.empty())
                {
                    throw std::invalid_argument(
                        "parameter names and value lists must be non-empty");
                }
                names.push_back(name);
            }
            std::vector<ParameterMap> combinations;
            ParameterMap current;
            enumerate_grid(grid, names, 0, current, combinations);
            if (grid.empty())
            {
                combinations.push_back({});
            }
            return combinations;
        }
    }

    GridSearch::GridSearch(
        ParameterizedEstimatorFactory estimator_factory,
        FitPredictFunction fit_predict,
        Scorer scorer)
        : estimator_factory_(std::move(estimator_factory)),
          fit_predict_(std::move(fit_predict)),
          scorer_(std::move(scorer))
    {
        if (!estimator_factory_ || !fit_predict_ || !scorer_)
        {
            throw std::invalid_argument(
                "GridSearch requires factory, fit/predict callback and scorer");
        }
    }

    GridSearch::GridSearch(
        std::function<std::unique_ptr<base::Predictor>(const ParameterMap&)>
            predictor_factory,
        Scorer scorer)
        : GridSearch(
              [predictor_factory](const ParameterMap& parameters)
              {
                  if (!predictor_factory)
                  {
                      return std::unique_ptr<base::Estimator>{};
                  }
                  return std::unique_ptr<base::Estimator>(
                      predictor_factory(parameters).release());
              },
              [](base::Estimator& estimator,
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
              },
              std::move(scorer))
    {
    }

    GridSearchResult GridSearch::search(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const ParameterGrid& parameter_grid,
        const GridSearchOptions& options) const
    {
        if (options.folds < 2)
        {
            throw std::invalid_argument("GridSearch folds must be at least 2");
        }
        const auto combinations = combinations_for(parameter_grid);
        GridSearchResult result;
        result.trials.reserve(combinations.size());

        for (const auto& parameters : combinations)
        {
            GridSearchTrial trial;
            trial.parameters = parameters;
            try
            {
                const auto factory = [this, parameters]()
                {
                    return estimator_factory_(parameters);
                };
                const auto cv = cross_validate(
                    features, targets, factory, fit_predict_, scorer_,
                    options.folds, options.random_seed);
                trial.score = cv.mean_score;
                if (!result.best_index.has_value()
                    || (options.greater_is_better
                        ? *trial.score > result.best_score
                        : *trial.score < result.best_score))
                {
                    result.best_index = result.trials.size();
                    result.best_parameters = parameters;
                    result.best_score = *trial.score;
                }
            }
            catch (const std::exception& error)
            {
                trial.error = error.what();
            }
            catch (...)
            {
                trial.error = "unknown exception";
            }
            result.trials.push_back(std::move(trial));
        }
        return result;
    }

    GridSearchResult GridSearch::fit(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const ParameterGrid& parameter_grid,
        const GridSearchOptions& options) const
    {
        return search(features, targets, parameter_grid, options);
    }
}
