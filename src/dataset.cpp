#include "dataset.h"
#include "csvReader.h"

#include <stdexcept>

namespace
{
    std::size_t findColumn(
        const std::vector<std::string>& column_names,
        const std::string& name
    )
    {
        for (std::size_t index = 0; index < column_names.size(); ++index)
        {
            if (column_names[index] == name)
            {
                return index;
            }
        }

        throw std::runtime_error("Column not found: " + name);
    }

    double parseNumber(const std::string& value, const std::string& column_name)
    {
        std::size_t parsed_length = 0;
        double number = 0.0;

        try
        {
            number = std::stod(value, &parsed_length);
        }
        catch (const std::exception&)
        {
            throw std::runtime_error(
                "Non-numeric value in column " + column_name + ": " + value
            );
        }

        if (parsed_length != value.size())
        {
            throw std::runtime_error(
                "Non-numeric value in column " + column_name + ": " + value
            );
        }

        return number;
    }
}

namespace art::data
{
Dataset Dataset::from_csv(
    const std::string& filename,
    const std::string& label_column,
    const std::vector<std::string>& requested_feature_columns
)
{
    CsvReader reader;
    const DataFrame data_frame = reader.read(filename);
    return Dataset::from_dataframe(data_frame, label_column, requested_feature_columns);
}

Dataset Dataset::from_dataframe(
    const DataFrame& data_frame,
    const std::string& label_column,
    const std::vector<std::string>& requested_feature_columns
)
{
    const std::size_t label_index = findColumn(data_frame.column_names, label_column);

    std::vector<std::string> selected_features = requested_feature_columns;

    if (selected_features.empty())
    {
        for (const std::string& column_name : data_frame.column_names)
        {
            if (column_name != label_column)
            {
                selected_features.push_back(column_name);
            }
        }
    }

    if (selected_features.empty())
    {
        throw std::runtime_error("Dataset must contain at least one feature column");
    }

    std::vector<std::size_t> feature_indices;

    for (const std::string& feature_name : selected_features)
    {
        if (feature_name == label_column)
        {
            throw std::runtime_error(
                "Label column cannot also be a feature column: " + feature_name
            );
        }

        feature_indices.push_back(findColumn(data_frame.column_names, feature_name));
    }

    Dataset dataset;
    dataset.feature_names = selected_features;
    dataset.label_name = label_column;
    dataset.features.resize(
        static_cast<Eigen::Index>(data_frame.rows.size()),
        static_cast<Eigen::Index>(feature_indices.size())
    );
    dataset.labels.resize(static_cast<Eigen::Index>(data_frame.rows.size()));

    for (std::size_t row_index = 0; row_index < data_frame.rows.size(); ++row_index)
    {
        const std::vector<std::string>& row = data_frame.rows[row_index];

        for (std::size_t index = 0; index < feature_indices.size(); ++index)
        {
            dataset.features(
                static_cast<Eigen::Index>(row_index),
                static_cast<Eigen::Index>(index)
            ) = parseNumber(
                row[feature_indices[index]],
                selected_features[index]
            );
        }

        dataset.labels(static_cast<Eigen::Index>(row_index)) =
            parseNumber(row[label_index], label_column);
    }

    return dataset;
}

std::size_t Dataset::sample_count() const
{
    return static_cast<std::size_t>(features.rows());
}

std::size_t Dataset::feature_count() const
{
    return static_cast<std::size_t>(features.cols());
}
}
