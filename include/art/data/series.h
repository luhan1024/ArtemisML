/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/core/errors.h"
#include "art/data/dtype.h"
#include "art/data/index.h"
#include "art/data/missing.h"

#include <cstdlib>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace art::data
{
class Series
{
public:
    Series() = default;

    Series(
        std::vector<std::string> values,
        std::string name = {},
        Index index = {}
    )
        : values_(std::move(values)), name_(std::move(name)), index_(std::move(index))
    {
        if (index_.empty() && !values_.empty())
        {
            index_ = Index::default_index(values_.size());
        }
        if (index_.size() != values_.size())
        {
            throw art::core::DimensionError("Series values and index lengths differ");
        }
    }

    std::size_t size() const noexcept { return values_.size(); }
    bool empty() const noexcept { return values_.empty(); }
    const std::string& name() const noexcept { return name_; }
    const Index& index() const noexcept { return index_; }

    const std::string& at(std::size_t position) const
    {
        if (position >= values_.size())
        {
            throw art::core::IndexError("Series position out of range: " + std::to_string(position));
        }
        return values_[position];
    }

    bool is_missing(std::size_t position, const MissingPolicy& policy = {}) const
    {
        return data::is_missing(at(position), policy);
    }

    DType dtype(const MissingPolicy& policy = {}) const
    {
        return infer_dtype(values_, policy);
    }

    std::vector<double> to_numeric(const MissingPolicy& policy = {}) const
    {
        std::vector<double> result;
        result.reserve(values_.size());
        for (const std::string& value : values_)
        {
            if (data::is_missing(value, policy))
            {
                throw art::core::MissingValueError("Cannot convert missing Series value to numeric");
            }
            char* end = nullptr;
            const double number = std::strtod(value.c_str(), &end);
            if (end == value.c_str() || *end != '\0')
            {
                throw art::core::TypeError("Series value is not numeric: " + value);
            }
            if (!std::isfinite(number))
            {
                throw art::core::DataError("Series value is not finite: " + value);
            }
            result.push_back(number);
        }
        return result;
    }

    const std::vector<std::string>& values() const noexcept { return values_; }

private:
    std::vector<std::string> values_;
    std::string name_;
    Index index_;
};
}
