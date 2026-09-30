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
#include "art/preprocessing/column_transformer.h"
#include "art/preprocessing/feature_generation.h"
#include "art/preprocessing/feature_selection.h"
#include "art/preprocessing/imputer.h"
#include "art/preprocessing/min_max_scaler.h"
#include "art/preprocessing/one_hot_encoder.h"
#include "art/preprocessing/standard_scaler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>

namespace art::preprocessing
{
namespace
{
    void validateNonEmpty(const base::DataInput& input, const char* name)
    {
        if (input.rows() == 0 || input.cols() == 0)
        {
            throw std::invalid_argument(
                std::string(name) + " requires a non-empty 2D input"
            );
        }
    }

    void validateFinite(const base::DataInput& input, const char* name)
    {
        validateNonEmpty(input, name);
        for (Eigen::Index row = 0; row < input.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < input.cols(); ++column)
            {
                if (!std::isfinite(input(row, column)))
                {
                    throw std::invalid_argument(
                        std::string(name) + " does not accept non-finite values"
                    );
                }
            }
        }
    }

    void validateWidth(const base::DataInput& input, std::size_t expected, const char* name)
    {
        validateNonEmpty(input, name);
        if (static_cast<std::size_t>(input.cols()) != expected)
        {
            throw std::invalid_argument(
                std::string(name) + " received " + std::to_string(input.cols()) +
                " features, expected " + std::to_string(expected)
            );
        }
    }

    base::DataInput select(const base::DataInput& input, const std::vector<std::size_t>& columns)
    {
        base::DataInput result(input.rows(), static_cast<Eigen::Index>(columns.size()));
        for (std::size_t output = 0; output < columns.size(); ++output)
        {
            result.col(static_cast<Eigen::Index>(output)) =
                input.col(static_cast<Eigen::Index>(columns[output]));
        }
        return result;
    }

    void validateColumns(const std::vector<std::size_t>& columns, std::size_t width, const char* name)
    {
        if (columns.empty())
        {
            throw std::invalid_argument(std::string(name) + " requires at least one column");
        }
        std::unordered_set<std::size_t> seen;
        for (const std::size_t column : columns)
        {
            if (column >= width)
            {
                throw std::invalid_argument(std::string(name) + " column index is out of range");
            }
            if (!seen.insert(column).second)
            {
                throw std::invalid_argument(std::string(name) + " column indices must be unique");
            }
        }
    }

    void addMonomials(
        const base::DataInput& input,
        Eigen::Index row,
        std::size_t next_column,
        std::size_t remaining_degree,
        double value,
        std::vector<double>& output
    )
    {
        if (remaining_degree == 0)
        {
            output.push_back(value);
            return;
        }
        for (std::size_t column = next_column;
             column < static_cast<std::size_t>(input.cols()); ++column)
        {
            addMonomials(
                input, row, column, remaining_degree - 1,
                value * input(row, static_cast<Eigen::Index>(column)), output
            );
        }
    }
}

StandardScaler::StandardScaler(bool with_mean, bool with_std)
    : with_mean_(with_mean), with_std_(with_std)
{
    if (!with_mean_ && !with_std_)
    {
        throw std::invalid_argument("StandardScaler must enable centering or scaling");
    }
}

const Eigen::VectorXd& StandardScaler::mean() const { return mean_; }
const Eigen::VectorXd& StandardScaler::scale() const { return scale_; }
std::size_t StandardScaler::feature_count() const noexcept { return feature_count_; }

void StandardScaler::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateFinite(features, "StandardScaler");
    feature_count_ = static_cast<std::size_t>(features.cols());
    mean_ = features.colwise().mean().transpose();
    scale_.resize(features.cols());
    for (Eigen::Index column = 0; column < features.cols(); ++column)
    {
        const Eigen::ArrayXd centered = features.col(column).array() - mean_(column);
        const double variance = centered.square().mean();
        scale_(column) = with_std_ ? std::sqrt(variance) : 1.0;
        if (scale_(column) == 0.0)
        {
            scale_(column) = 1.0;
        }
    }
    if (!with_mean_)
    {
        mean_.setZero();
    }
}

base::DataInput StandardScaler::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "StandardScaler");
    validateFinite(input, "StandardScaler");
    base::DataInput result(input.rows(), input.cols());
    for (Eigen::Index row = 0; row < input.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < input.cols(); ++column)
        {
            result(row, column) =
                (input(row, column) - mean_(column)) / scale_(column);
        }
    }
    return result;
}

MinMaxScaler::MinMaxScaler(double lower, double upper)
    : lower_(lower), upper_(upper)
{
    if (!std::isfinite(lower_) || !std::isfinite(upper_) || lower_ >= upper_)
    {
        throw std::invalid_argument("MinMaxScaler requires a finite lower bound below upper bound");
    }
}

double MinMaxScaler::lower() const noexcept { return lower_; }
double MinMaxScaler::upper() const noexcept { return upper_; }
const Eigen::VectorXd& MinMaxScaler::data_min() const { return data_min_; }
const Eigen::VectorXd& MinMaxScaler::data_max() const { return data_max_; }
std::size_t MinMaxScaler::feature_count() const noexcept { return feature_count_; }

void MinMaxScaler::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateFinite(features, "MinMaxScaler");
    feature_count_ = static_cast<std::size_t>(features.cols());
    data_min_ = features.colwise().minCoeff().transpose();
    data_max_ = features.colwise().maxCoeff().transpose();
    scale_.resize(features.cols());
    for (Eigen::Index column = 0; column < features.cols(); ++column)
    {
        const double span = data_max_(column) - data_min_(column);
        scale_(column) = span == 0.0 ? 0.0 : (upper_ - lower_) / span;
    }
}

base::DataInput MinMaxScaler::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "MinMaxScaler");
    validateFinite(input, "MinMaxScaler");
    base::DataInput result(input.rows(), input.cols());
    for (Eigen::Index column = 0; column < input.cols(); ++column)
    {
        result.col(column) =
            (input.col(column).array() - data_min_(column)) * scale_(column) +
            lower_;
    }
    return result;
}

Imputer::Imputer(ImputationStrategy strategy, double fill_value)
    : strategy_(strategy), fill_value_(fill_value)
{
    if (strategy_ == ImputationStrategy::Constant && !std::isfinite(fill_value_))
    {
        throw std::invalid_argument("Imputer constant fill value must be finite");
    }
}

ImputationStrategy Imputer::strategy() const noexcept { return strategy_; }
const Eigen::VectorXd& Imputer::statistics() const { return statistics_; }
std::size_t Imputer::feature_count() const noexcept { return feature_count_; }

void Imputer::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateNonEmpty(features, "Imputer");
    feature_count_ = static_cast<std::size_t>(features.cols());
    statistics_.resize(features.cols());
    for (Eigen::Index column = 0; column < features.cols(); ++column)
    {
        if (strategy_ == ImputationStrategy::Constant)
        {
            statistics_(column) = fill_value_;
            for (Eigen::Index row = 0; row < features.rows(); ++row)
            {
                if (!std::isfinite(features(row, column)) &&
                    !std::isnan(features(row, column)))
                {
                    throw std::invalid_argument(
                        "Imputer only treats NaN as missing; infinity is unsupported"
                    );
                }
            }
            continue;
        }
        double sum = 0.0;
        Eigen::Index count = 0;
        for (Eigen::Index row = 0; row < features.rows(); ++row)
        {
            if (std::isnan(features(row, column)))
            {
                continue;
            }
            if (!std::isfinite(features(row, column)))
            {
                throw std::invalid_argument(
                    "Imputer only treats NaN as missing; infinity is unsupported"
                );
            }
            if (std::isfinite(features(row, column)))
            {
                sum += features(row, column);
                ++count;
            }
        }
        if (count == 0)
        {
            throw std::invalid_argument("Imputer mean strategy cannot fit an all-missing column");
        }
        statistics_(column) = sum / static_cast<double>(count);
    }
}

base::DataInput Imputer::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "Imputer");
    base::DataInput result = input;
    for (Eigen::Index row = 0; row < result.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < result.cols(); ++column)
        {
            if (std::isnan(result(row, column)))
            {
                result(row, column) = statistics_(column);
            }
            else if (!std::isfinite(result(row, column)))
            {
                throw std::invalid_argument("Imputer only treats NaN as missing; infinity is unsupported");
            }
        }
    }
    return result;
}

OneHotEncoder::OneHotEncoder(UnknownCategoryPolicy policy) : policy_(policy) {}
UnknownCategoryPolicy OneHotEncoder::unknown_policy() const noexcept { return policy_; }
const std::vector<std::vector<double>>& OneHotEncoder::categories() const { return categories_; }
std::size_t OneHotEncoder::feature_count() const noexcept { return feature_count_; }
std::size_t OneHotEncoder::output_feature_count() const noexcept { return output_feature_count_; }

void OneHotEncoder::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateFinite(features, "OneHotEncoder");
    feature_count_ = static_cast<std::size_t>(features.cols());
    categories_.assign(feature_count_, {});
    output_feature_count_ = 0;
    for (Eigen::Index column = 0; column < features.cols(); ++column)
    {
        auto& categories = categories_[static_cast<std::size_t>(column)];
        for (Eigen::Index row = 0; row < features.rows(); ++row)
        {
            categories.push_back(features(row, column));
        }
        std::sort(categories.begin(), categories.end());
        categories.erase(std::unique(categories.begin(), categories.end()), categories.end());
        output_feature_count_ += categories.size();
    }
}

base::DataInput OneHotEncoder::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "OneHotEncoder");
    validateFinite(input, "OneHotEncoder");
    base::DataInput result = base::DataInput::Zero(
        input.rows(), static_cast<Eigen::Index>(output_feature_count_)
    );
    Eigen::Index output_column = 0;
    for (Eigen::Index column = 0; column < input.cols(); ++column)
    {
        const auto& categories = categories_[static_cast<std::size_t>(column)];
        for (Eigen::Index row = 0; row < input.rows(); ++row)
        {
            const auto iterator = std::lower_bound(
                categories.begin(), categories.end(), input(row, column)
            );
            if (iterator == categories.end() || *iterator != input(row, column))
            {
                if (policy_ == UnknownCategoryPolicy::Error)
                {
                    throw std::invalid_argument("OneHotEncoder encountered an unknown category");
                }
                continue;
            }
            result(row, output_column + static_cast<Eigen::Index>(iterator - categories.begin())) = 1.0;
        }
        output_column += static_cast<Eigen::Index>(categories.size());
    }
    return result;
}

ColumnTransformer::ColumnTransformer(std::vector<Specification> specifications)
    : specifications_(std::move(specifications))
{
    if (specifications_.empty())
    {
        throw std::invalid_argument("ColumnTransformer requires at least one specification");
    }
    std::unordered_set<std::string> names;
    for (const Specification& specification : specifications_)
    {
        if (specification.name.empty() || !names.insert(specification.name).second)
        {
            throw std::invalid_argument("ColumnTransformer specification names must be non-empty and unique");
        }
        if (!specification.transformer)
        {
            throw std::invalid_argument("ColumnTransformer specification Transformer cannot be null");
        }
        if (specification.columns.empty())
        {
            throw std::invalid_argument("ColumnTransformer specification requires columns");
        }
    }
}

const std::vector<ColumnTransformer::Specification>& ColumnTransformer::specifications() const noexcept { return specifications_; }
std::size_t ColumnTransformer::input_feature_count() const noexcept { return input_feature_count_; }
std::size_t ColumnTransformer::output_feature_count() const noexcept { return output_feature_count_; }

void ColumnTransformer::do_fit(const base::FeatureInput& features, const base::TargetInput& targets)
{
    validateNonEmpty(features, "ColumnTransformer");
    input_feature_count_ = static_cast<std::size_t>(features.cols());
    std::unordered_set<std::size_t> selected;
    output_feature_count_ = 0;
    for (Specification& specification : specifications_)
    {
        validateColumns(specification.columns, input_feature_count_, "ColumnTransformer");
        for (const std::size_t column : specification.columns)
        {
            if (!selected.insert(column).second)
            {
                throw std::invalid_argument("ColumnTransformer specifications cannot overlap");
            }
        }
        const base::DataInput subset = select(features, specification.columns);
        specification.transformer->fit(subset, targets);
        const base::DataInput transformed = specification.transformer->transform(subset);
        if (transformed.rows() != features.rows())
        {
            throw std::invalid_argument("ColumnTransformer child changed the number of samples");
        }
        output_feature_count_ += static_cast<std::size_t>(transformed.cols());
    }
}

base::DataInput ColumnTransformer::do_transform(const base::DataInput& input) const
{
    validateWidth(input, input_feature_count_, "ColumnTransformer");
    base::DataInput result(input.rows(), static_cast<Eigen::Index>(output_feature_count_));
    Eigen::Index output_column = 0;
    for (const Specification& specification : specifications_)
    {
        const base::DataInput transformed = specification.transformer->transform(
            select(input, specification.columns)
        );
        if (transformed.rows() != input.rows())
        {
            throw std::invalid_argument("ColumnTransformer child changed the number of samples");
        }
        result.middleCols(output_column, transformed.cols()) = transformed;
        output_column += transformed.cols();
    }
    return result;
}

SelectColumns::SelectColumns(std::vector<std::size_t> columns)
    : columns_(std::move(columns))
{
    if (columns_.empty())
    {
        throw std::invalid_argument("SelectColumns requires at least one column");
    }
}

const std::vector<std::size_t>& SelectColumns::columns() const noexcept { return columns_; }

void SelectColumns::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateNonEmpty(features, "SelectColumns");
    feature_count_ = static_cast<std::size_t>(features.cols());
    validateColumns(columns_, feature_count_, "SelectColumns");
}

base::DataInput SelectColumns::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "SelectColumns");
    return select(input, columns_);
}

PolynomialFeatures::PolynomialFeatures(std::size_t degree, bool include_bias)
    : degree_(degree), include_bias_(include_bias)
{
    if (degree_ == 0)
    {
        throw std::invalid_argument("PolynomialFeatures degree must be at least one");
    }
}

std::size_t PolynomialFeatures::degree() const noexcept { return degree_; }
bool PolynomialFeatures::include_bias() const noexcept { return include_bias_; }
std::size_t PolynomialFeatures::feature_count() const noexcept { return feature_count_; }
std::size_t PolynomialFeatures::output_feature_count() const noexcept { return output_feature_count_; }

void PolynomialFeatures::do_fit(const base::FeatureInput& features, const base::TargetInput&)
{
    validateFinite(features, "PolynomialFeatures");
    feature_count_ = static_cast<std::size_t>(features.cols());
    output_feature_count_ = include_bias_ ? 1 : 0;
    for (std::size_t current_degree = 1; current_degree <= degree_; ++current_degree)
    {
        std::size_t count = 1;
        for (std::size_t index = 0; index < current_degree; ++index)
        {
            count *= feature_count_ + index;
            count /= index + 1;
        }
        output_feature_count_ += count;
    }
}

base::DataInput PolynomialFeatures::do_transform(const base::DataInput& input) const
{
    validateWidth(input, feature_count_, "PolynomialFeatures");
    validateFinite(input, "PolynomialFeatures");
    base::DataInput result(input.rows(), static_cast<Eigen::Index>(output_feature_count_));
    for (Eigen::Index row = 0; row < input.rows(); ++row)
    {
        std::vector<double> values;
        values.reserve(output_feature_count_);
        if (include_bias_)
        {
            values.push_back(1.0);
        }
        for (std::size_t current_degree = 1; current_degree <= degree_; ++current_degree)
        {
            addMonomials(input, row, 0, current_degree, 1.0, values);
        }
        for (std::size_t column = 0; column < values.size(); ++column)
        {
            result(row, static_cast<Eigen::Index>(column)) = values[column];
        }
    }
    return result;
}
}
