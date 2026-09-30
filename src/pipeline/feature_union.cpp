/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#include "art/pipeline/feature_union.h"

#include <stdexcept>

namespace art::pipeline
{
namespace
{
    void validateSteps(const std::vector<FeatureUnion::Step>& steps)
    {
        if (steps.empty())
        {
            throw std::invalid_argument("FeatureUnion requires at least one Transformer");
        }
        for (std::size_t index = 0; index < steps.size(); ++index)
        {
            if (steps[index].first.empty())
            {
                throw std::invalid_argument("FeatureUnion step name cannot be empty");
            }
            if (!steps[index].second)
            {
                throw std::invalid_argument(
                    "FeatureUnion step '" + steps[index].first + "' is null"
                );
            }
            for (std::size_t previous = 0; previous < index; ++previous)
            {
                if (steps[previous].first == steps[index].first)
                {
                    throw std::invalid_argument(
                        "FeatureUnion step names must be unique: " + steps[index].first
                    );
                }
            }
        }
    }

    base::DataInput concatenate(
        const std::vector<base::DataInput>& outputs,
        Eigen::Index expected_rows
    )
    {
        Eigen::Index total_columns = 0;
        for (const base::DataInput& output : outputs)
        {
            if (output.rows() != expected_rows)
            {
                throw std::invalid_argument(
                    "FeatureUnion Transformer outputs have different row counts"
                );
            }
            total_columns += output.cols();
        }

        base::DataInput result(expected_rows, total_columns);
        Eigen::Index column = 0;
        for (const base::DataInput& output : outputs)
        {
            result.middleCols(column, output.cols()) = output;
            column += output.cols();
        }
        return result;
    }
}

FeatureUnion::FeatureUnion(std::vector<Step> steps)
    : steps_(std::move(steps))
{
    validateSteps(steps_);
}

std::size_t FeatureUnion::size() const noexcept { return steps_.size(); }
const std::vector<FeatureUnion::Step>& FeatureUnion::steps() const noexcept { return steps_; }

const FeatureUnion::Step& FeatureUnion::step(std::size_t index) const
{
    if (index >= steps_.size())
    {
        throw std::out_of_range("FeatureUnion step index is out of range");
    }
    return steps_[index];
}

const FeatureUnion::Step& FeatureUnion::step(const std::string& name) const
{
    for (const Step& current : steps_)
    {
        if (current.first == name)
        {
            return current;
        }
    }
    throw std::out_of_range("FeatureUnion step not found: " + name);
}

void FeatureUnion::do_fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    for (const Step& step : steps_)
    {
        step.second->fit(features, targets);
    }
}

base::DataInput FeatureUnion::do_transform(const base::DataInput& input) const
{
    std::vector<base::DataInput> outputs;
    outputs.reserve(steps_.size());
    for (const Step& step : steps_)
    {
        outputs.push_back(step.second->transform(input));
    }
    return concatenate(outputs, input.rows());
}
}
