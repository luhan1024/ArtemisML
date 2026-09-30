/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#include "art/preprocessing/label_encoder.h"

#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace art::preprocessing
{
namespace
{
    void validateLabels(const std::vector<std::string>& labels, const char* operation)
    {
        if (labels.empty())
        {
            throw std::invalid_argument(
                std::string("LabelEncoder ") + operation + " requires at least one label"
            );
        }
    }
}

LabelEncoder::LabelEncoder(
    LabelEncodingStrategy strategy,
    UnknownLabelPolicy unknown_policy
)
    : strategy_(strategy), unknown_policy_(unknown_policy)
{
}

void LabelEncoder::fit(const std::vector<std::string>& labels)
{
    validateLabels(labels, "fit()");
    fitted_ = false;
    classes_.clear();

    std::unordered_map<std::string, std::size_t> seen;
    for (const std::string& label : labels)
    {
        if (seen.emplace(label, classes_.size()).second)
        {
            classes_.push_back(label);
        }
    }

    const std::size_t class_count = classes_.size();
    if (strategy_ == LabelEncodingStrategy::Auto)
    {
        effective_strategy_ = class_count == 2
            ? LabelEncodingStrategy::Binary01
            : (class_count >= 3
                ? LabelEncodingStrategy::OneHot
                : LabelEncodingStrategy::Ordinal);
    }
    else
    {
        effective_strategy_ = strategy_;
    }

    if ((effective_strategy_ == LabelEncodingStrategy::Binary01 ||
         effective_strategy_ == LabelEncodingStrategy::BinarySigned) &&
        class_count != 2)
    {
        throw std::invalid_argument("Binary label encoding requires exactly two classes");
    }
    if (effective_strategy_ == LabelEncodingStrategy::SignedOrdinal && class_count != 3)
    {
        throw std::invalid_argument("SignedOrdinal label encoding requires exactly three classes");
    }
    if (effective_strategy_ == LabelEncodingStrategy::Auto)
    {
        throw std::logic_error("LabelEncoder failed to resolve its encoding strategy");
    }
    fitted_ = true;
}

std::size_t LabelEncoder::class_index(const std::string& label) const
{
    for (std::size_t index = 0; index < classes_.size(); ++index)
    {
        if (classes_[index] == label) return index;
    }
    return std::numeric_limits<std::size_t>::max();
}

Eigen::VectorXd LabelEncoder::transform_ordinal_values(
    const std::vector<std::string>& labels
) const
{
    validateLabels(labels, "transform()");
    Eigen::VectorXd result(static_cast<Eigen::Index>(labels.size()));
    for (std::size_t row = 0; row < labels.size(); ++row)
    {
        const std::size_t index = class_index(labels[row]);
        if (index == std::numeric_limits<std::size_t>::max())
        {
            if (unknown_policy_ == UnknownLabelPolicy::Error)
            {
                throw std::invalid_argument("LabelEncoder encountered an unknown label: " + labels[row]);
            }
            result(static_cast<Eigen::Index>(row)) =
                std::numeric_limits<double>::quiet_NaN();
            continue;
        }

        double value = static_cast<double>(index);
        if (effective_strategy_ == LabelEncodingStrategy::BinarySigned)
        {
            value = index == 0 ? -1.0 : 1.0;
        }
        else if (effective_strategy_ == LabelEncodingStrategy::SignedOrdinal)
        {
            value = static_cast<double>(index) - 1.0;
        }
        result(static_cast<Eigen::Index>(row)) = value;
    }
    return result;
}

Eigen::VectorXd LabelEncoder::transform_vector(
    const std::vector<std::string>& labels
) const
{
    if (!fitted_)
    {
        throw std::logic_error("LabelEncoder cannot transform before fit()");
    }
    if (effective_strategy_ == LabelEncodingStrategy::OneHot)
    {
        throw std::invalid_argument("LabelEncoder strategy produces a matrix, not a vector");
    }
    return transform_ordinal_values(labels);
}

Eigen::MatrixXd LabelEncoder::transform_matrix(
    const std::vector<std::string>& labels
) const
{
    if (!fitted_)
    {
        throw std::logic_error("LabelEncoder cannot transform before fit()");
    }
    if (effective_strategy_ != LabelEncodingStrategy::OneHot)
    {
        throw std::invalid_argument("LabelEncoder strategy produces a vector, not a matrix");
    }
    validateLabels(labels, "transform()");
    Eigen::MatrixXd result = Eigen::MatrixXd::Zero(
        static_cast<Eigen::Index>(labels.size()),
        static_cast<Eigen::Index>(classes_.size())
    );
    for (std::size_t row = 0; row < labels.size(); ++row)
    {
        const std::size_t index = class_index(labels[row]);
        if (index == std::numeric_limits<std::size_t>::max())
        {
            if (unknown_policy_ == UnknownLabelPolicy::Error)
            {
                throw std::invalid_argument("LabelEncoder encountered an unknown label: " + labels[row]);
            }
            continue;
        }
        result(
            static_cast<Eigen::Index>(row),
            static_cast<Eigen::Index>(index)
        ) = 1.0;
    }
    return result;
}

EncodedTarget LabelEncoder::transform(const std::vector<std::string>& labels) const
{
    if (!fitted_)
    {
        throw std::logic_error("LabelEncoder cannot transform before fit()");
    }
    if (effective_strategy_ == LabelEncodingStrategy::OneHot)
    {
        return transform_matrix(labels);
    }
    return transform_vector(labels);
}

EncodedTarget LabelEncoder::fit_transform(const std::vector<std::string>& labels)
{
    fit(labels);
    return transform(labels);
}

bool LabelEncoder::is_fitted() const noexcept { return fitted_; }
LabelEncodingStrategy LabelEncoder::strategy() const noexcept { return strategy_; }
LabelEncodingStrategy LabelEncoder::effective_strategy() const
{
    if (!fitted_) throw std::logic_error("LabelEncoder strategy is unavailable before fit()");
    return effective_strategy_;
}
UnknownLabelPolicy LabelEncoder::unknown_policy() const noexcept { return unknown_policy_; }
const std::vector<std::string>& LabelEncoder::classes() const { return classes_; }

std::vector<std::string> LabelEncoder::output_column_names(const std::string& prefix) const
{
    if (!fitted_)
    {
        throw std::logic_error("LabelEncoder column names are unavailable before fit()");
    }
    if (effective_strategy_ != LabelEncodingStrategy::OneHot)
    {
        return {prefix};
    }
    std::vector<std::string> names;
    names.reserve(classes_.size());
    for (const std::string& label : classes_)
    {
        names.push_back(prefix + "=" + label);
    }
    return names;
}
}
