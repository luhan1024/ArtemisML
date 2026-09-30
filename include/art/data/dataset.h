#pragma once

#include "art/data/dataframe.h"

#include <Eigen/Dense>

#include <string>
#include <vector>

namespace art::data
{
    struct Dataset
    {
        Eigen::MatrixXd features;
        Eigen::VectorXd labels;
        std::vector<std::string> feature_names;
        std::string label_name;

        static Dataset from_csv(
            const std::string& filename,
            const std::string& label_column,
            const std::vector<std::string>& feature_columns = {}
        );

        static Dataset from_dataframe(
            const DataFrame& data_frame,
            const std::string& label_column,
            const std::vector<std::string>& feature_columns = {}
        );

        std::size_t sample_count() const;
        std::size_t feature_count() const;
    };
}
