/*
 * +----------------------------------------------------------------------------+
 * |                                                                            |
 * | .###.  ####.  #####  #####  #...#  #####  .####  #...#  #....              |
 * | #...#  #...#  ..#..  #....  ##.##  ..#..  #....  ##.##  #....              |
 * | #####  ####.  ..#..  ####.  #.#.#  ..#..  .###.  #.#.#  #....              |
 * | #...#  #.#..  ..#..  #....  #...#  ..#..  ....#  #...#  #....              |
 * | #...#  #..##  ..#..  #####  #...#  #####  ####.  #...#  #####              |
 * |                                                                            |
 * | Han Lu & Yihan Wang                                                        |
 * +----------------------------------------------------------------------------+
 */
#include "art/pipeline/pipeline.h"

#include <stdexcept>

namespace art::pipeline
{
namespace
{
    void validateStepList(const std::vector<Pipeline::Step>& steps)
    {
        if (steps.empty())
        {
            throw std::invalid_argument("Pipeline requires at least one step");
        }

        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            if (steps[index].first.empty())
            {
                throw std::invalid_argument("Pipeline step name cannot be empty");
            }
            if (!steps[index].second)
            {
                throw std::invalid_argument(
                    "Pipeline step '" + steps[index].first + "' is null"
                );
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (steps[previous].first == steps[index].first)
                {
                    throw std::invalid_argument(
                        "Pipeline step names must be unique: " + steps[index].first
                    );
                }
            }

            if (index + 1 < steps.size() &&
                !std::dynamic_pointer_cast<base::Transformer>(steps[index].second))
            {
                throw std::invalid_argument(
                    "Pipeline step '" + steps[index].first +
                    "' must be a Transformer unless it is the final step"
                );
            }
        }
    }

    void validateTargets(
        const base::FeatureInput& features,
        const base::TargetInput& targets
    )
    {
        if (targets.size() != 0 && targets.size() != features.rows())
        {
            throw std::invalid_argument(
                "Pipeline features and targets have different sample counts"
            );
        }
    }

    base::DataInput transformPrefix(
        const std::vector<Pipeline::Step>& steps,
        const base::DataInput& input,
        std::size_t count
    )
    {
        base::DataInput current = input;
        for (std::size_t index = 0; index < count; ++index)
        {
            const auto transformer =
                std::dynamic_pointer_cast<base::Transformer>(steps[index].second);
            if (!transformer)
            {
                throw std::logic_error(
                    "Pipeline step '" + steps[index].first +
                    "' is not a Transformer"
                );
            }
            const base::DataInput transformed = transformer->transform(current);
            if (transformed.rows() != current.rows())
            {
                throw std::invalid_argument(
                    "Pipeline transformer '" + steps[index].first +
                    "' changed the number of samples"
                );
            }
            current = transformed;
        }
        return current;
    }
}

Pipeline::Pipeline(std::vector<Step> steps)
    : steps_(std::move(steps))
{
    validateStepList(steps_);
}

void Pipeline::fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    validateTargets(features, targets);
    fitted_ = false;

    base::DataInput current = features;
    for (std::size_t index = 0; index + 1 < steps_.size(); ++index)
    {
        const auto transformer =
            std::dynamic_pointer_cast<base::Transformer>(steps_[index].second);
        transformer->fit(current, targets);
        const base::DataInput transformed = transformer->transform(current);
        if (transformed.rows() != current.rows())
        {
            throw std::invalid_argument(
                "Pipeline transformer '" + steps_[index].first +
                "' changed the number of samples"
            );
        }
        current = transformed;
    }

    steps_.back().second->fit(current, targets);
    fitted_ = true;
}

base::DataInput Pipeline::transform(const base::DataInput& input) const
{
    if (!fitted_)
    {
        throw base::NotFittedError("Pipeline::transform()");
    }
    const auto final_transformer =
        std::dynamic_pointer_cast<base::Transformer>(steps_.back().second);
    if (!final_transformer)
    {
        throw std::logic_error(
            "Pipeline::transform() requires a Transformer as the final step"
        );
    }

    base::DataInput current = transformPrefix(steps_, input, steps_.size() - 1);
    const base::DataInput transformed = final_transformer->transform(current);
    if (transformed.rows() != current.rows())
    {
        throw std::invalid_argument("Pipeline final Transformer changed the number of samples");
    }
    return transformed;
}

base::DataInput Pipeline::fit_transform(
    const base::DataInput& input,
    const base::TargetInput& targets
)
{
    const auto final_transformer =
        std::dynamic_pointer_cast<base::Transformer>(steps_.back().second);
    if (!final_transformer)
    {
        throw std::logic_error(
            "Pipeline::fit_transform() requires a Transformer as the final step"
        );
    }

    validateTargets(input, targets);
    fitted_ = false;

    // Fit and transform each step exactly once. Calling fit() followed by
    // transform() here would apply every prefix Transformer twice and would
    // make stateful transformers observe an artificial second pass.
    base::DataInput current = input;
    for (std::size_t index = 0; index + 1 < steps_.size(); ++index)
    {
        const auto transformer =
            std::dynamic_pointer_cast<base::Transformer>(steps_[index].second);
        transformer->fit(current, targets);
        const base::DataInput transformed = transformer->transform(current);
        if (transformed.rows() != current.rows())
        {
            throw std::invalid_argument(
                "Pipeline transformer '" + steps_[index].first +
                "' changed the number of samples"
            );
        }
        current = transformed;
    }

    final_transformer->fit(current, targets);
    const base::DataInput transformed = final_transformer->transform(current);
    if (transformed.rows() != current.rows())
    {
        throw std::invalid_argument(
            "Pipeline final Transformer changed the number of samples"
        );
    }
    fitted_ = true;
    return transformed;
}

base::PredictionOutput Pipeline::predict(
    const base::FeatureInput& features
) const
{
    if (!fitted_)
    {
        throw base::NotFittedError("Pipeline::predict()");
    }
    const auto predictor =
        std::dynamic_pointer_cast<base::Predictor>(steps_.back().second);
    if (!predictor)
    {
        throw std::logic_error(
            "Pipeline::predict() requires a Predictor as the final step"
        );
    }
    return predictor->predict(
        transformPrefix(steps_, features, steps_.size() - 1)
    );
}

double Pipeline::score(
    const base::FeatureInput& features,
    const base::TargetInput& targets
) const
{
    if (!fitted_)
    {
        throw base::NotFittedError("Pipeline::score()");
    }
    validateTargets(features, targets);
    const auto regressor =
        std::dynamic_pointer_cast<base::Regressor>(steps_.back().second);
    const auto classifier =
        std::dynamic_pointer_cast<base::Classifier>(steps_.back().second);
    const base::DataInput transformed =
        transformPrefix(steps_, features, steps_.size() - 1);
    if (regressor)
    {
        return regressor->score(transformed, targets);
    }
    if (classifier)
    {
        return classifier->score(transformed, targets);
    }
    throw std::logic_error(
        "Pipeline::score() requires a Regressor or Classifier as the final step"
    );
}

bool Pipeline::is_fitted() const noexcept { return fitted_; }
std::size_t Pipeline::size() const noexcept { return steps_.size(); }
const std::vector<Pipeline::Step>& Pipeline::steps() const noexcept { return steps_; }

const Pipeline::Step& Pipeline::step(std::size_t index) const
{
    if (index >= steps_.size())
    {
        throw std::out_of_range("Pipeline step index is out of range");
    }
    return steps_[index];
}

const Pipeline::Step& Pipeline::step(const std::string& name) const
{
    for (const Step& current : steps_)
    {
        if (current.first == name)
        {
            return current;
        }
    }
    throw std::out_of_range("Pipeline step not found: " + name);
}

const std::shared_ptr<base::Estimator>& Pipeline::final_estimator() const
{
    return steps_.back().second;
}
}
