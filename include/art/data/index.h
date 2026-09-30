/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/core/errors.h"
#include "art/core/types.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace art::data
{
class Index
{
public:
    Index() = default;

    explicit Index(std::vector<std::string> labels)
        : labels_(std::move(labels))
    {
        rebuild_positions();
    }

    static Index default_index(std::size_t size)
    {
        std::vector<std::string> labels;
        labels.reserve(size);
        for (std::size_t position = 0; position < size; ++position)
        {
            labels.push_back(std::to_string(position));
        }
        return Index(std::move(labels));
    }

    std::size_t size() const noexcept { return labels_.size(); }
    bool empty() const noexcept { return labels_.empty(); }

    const std::string& at(std::size_t position) const
    {
        if (position >= labels_.size())
        {
            throw art::core::IndexError("Index position out of range: " + std::to_string(position));
        }
        return labels_[position];
    }

    std::size_t position_of(const std::string& label) const
    {
        const auto found = positions_.find(label);
        if (found == positions_.end())
        {
            throw art::core::IndexError("Index label not found: " + label);
        }
        return found->second;
    }

    bool contains(const std::string& label) const noexcept
    {
        return positions_.find(label) != positions_.end();
    }

    const std::vector<std::string>& labels() const noexcept { return labels_; }

private:
    void rebuild_positions()
    {
        for (std::size_t position = 0; position < labels_.size(); ++position)
        {
            if (!positions_.emplace(labels_[position], position).second)
            {
                throw art::core::DataError("Duplicate index label: " + labels_[position]);
            }
        }
    }

    std::vector<std::string> labels_;
    std::unordered_map<std::string, std::size_t> positions_;
};
}
