/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#pragma once

#include "art/core/errors.h"
#include "art/data/dataframe.h"

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace art::data
{
class LabelVocabulary
{
public:
    LabelVocabulary() = default;

    explicit LabelVocabulary(std::vector<std::string> values)
        : values_(std::move(values))
    {
        for (std::size_t index = 0; index < values_.size(); ++index)
        {
            if (!indices_.emplace(values_[index], index).second)
            {
                throw art::core::DataError("Duplicate label in vocabulary: " + values_[index]);
            }
        }
    }

    static LabelVocabulary from_series(const Series& labels)
    {
        std::vector<std::string> values;
        std::unordered_map<std::string, bool> seen;
        for (const std::string& label : labels.values())
        {
            if (seen.emplace(label, true).second)
            {
                values.push_back(label);
            }
        }
        return LabelVocabulary(std::move(values));
    }

    std::size_t size() const noexcept { return values_.size(); }
    const std::vector<std::string>& values() const noexcept { return values_; }

    bool contains(const std::string& label) const noexcept
    {
        return indices_.find(label) != indices_.end();
    }

    std::size_t index_of(const std::string& label) const
    {
        const auto found = indices_.find(label);
        if (found == indices_.end())
        {
            throw art::core::IndexError("Label not found in vocabulary: " + label);
        }
        return found->second;
    }

private:
    std::vector<std::string> values_;
    std::unordered_map<std::string, std::size_t> indices_;
};

struct TextLabelDataset
{
    DataFrame features;
    Series labels;
    std::vector<std::string> feature_names;
    std::string label_name;

    static TextLabelDataset from_dataframe(
        const DataFrame& data_frame,
        const std::string& label_column,
        const std::vector<std::string>& requested_feature_columns = {}
    )
    {
        data_frame.validate();
        const std::size_t label_index = data_frame.column_index(label_column);
        (void)label_index;

        std::vector<std::string> selected = requested_feature_columns;
        if (selected.empty())
        {
            for (const std::string& name : data_frame.column_names)
            {
                if (name != label_column) selected.push_back(name);
            }
        }
        if (selected.empty())
        {
            throw art::core::DataError("TextLabelDataset must contain at least one feature column");
        }
        for (const std::string& name : selected)
        {
            if (name == label_column)
            {
                throw art::core::DataError("Label column cannot also be a feature column: " + name);
            }
        }

        TextLabelDataset result;
        result.features = data_frame.select_columns(selected);
        std::vector<std::string> labels;
        labels.reserve(data_frame.rows.size());
        for (const auto& row : data_frame.rows)
        {
            labels.push_back(row[label_index]);
        }
        result.labels = Series(std::move(labels), label_column, data_frame.row_index());
        result.feature_names = std::move(selected);
        result.label_name = label_column;
        return result;
    }

    std::size_t sample_count() const noexcept { return labels.size(); }
    std::size_t feature_count() const noexcept { return feature_names.size(); }
    LabelVocabulary label_vocabulary() const { return LabelVocabulary::from_series(labels); }
};
}
