/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#include "art/io/csvReader.h"
#include "art/preprocessing/label_encoder.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace
{
    template <typename Function>
    void expect_invalid(Function&& function)
    {
        bool raised = false;
        try { function(); }
        catch (const std::invalid_argument&) { raised = true; }
        assert(raised);
    }

    template <typename Function>
    void expect_logic_error(Function&& function)
    {
        bool raised = false;
        try { function(); }
        catch (const std::logic_error&) { raised = true; }
        assert(raised);
    }
}

int main()
{
    const art::data::DataFrame frame = art::io::CsvReader().read("iris.csv");
    const std::size_t label_column = frame.column_index("species");
    std::vector<std::string> iris_labels;
    iris_labels.reserve(frame.rows.size());
    for (const auto& row : frame.rows)
    {
        iris_labels.push_back(row[label_column]);
    }

    art::preprocessing::LabelEncoder auto_encoder;
    expect_logic_error([&] { auto_encoder.transform(iris_labels); });
    const auto auto_result = auto_encoder.fit_transform(iris_labels);
    assert(auto_encoder.is_fitted());
    assert(auto_encoder.effective_strategy() ==
           art::preprocessing::LabelEncodingStrategy::OneHot);
    const std::vector<std::string> expected_classes =
        {"setosa", "versicolor", "virginica"};
    assert(auto_encoder.classes() == expected_classes);
    const Eigen::MatrixXd auto_matrix = std::get<Eigen::MatrixXd>(auto_result);
    const std::size_t class_size = iris_labels.size() / 3;
    assert(iris_labels.size() % 3 == 0);
    assert(auto_matrix.rows() == static_cast<Eigen::Index>(iris_labels.size()));
    assert(auto_matrix.cols() == 3);
    assert(auto_matrix(0, 0) == 1.0 && auto_matrix(0, 1) == 0.0);
    assert(auto_matrix(static_cast<Eigen::Index>(class_size), 1) == 1.0);
    assert(auto_matrix(static_cast<Eigen::Index>(2 * class_size), 2) == 1.0);
    const std::vector<std::string> expected_columns =
        {"species=setosa", "species=versicolor", "species=virginica"};
    assert(auto_encoder.output_column_names("species") == expected_columns);

    art::preprocessing::LabelEncoder ordinal(
        art::preprocessing::LabelEncodingStrategy::SignedOrdinal
    );
    const Eigen::VectorXd iris_ordinal = std::get<Eigen::VectorXd>(
        ordinal.fit_transform(iris_labels)
    );
    assert(iris_ordinal(0) == -1.0);
    assert(iris_ordinal(static_cast<Eigen::Index>(class_size)) == 0.0);
    assert(iris_ordinal(static_cast<Eigen::Index>(2 * class_size)) == 1.0);

    const std::vector<std::string> binary_labels = {"no", "yes", "yes"};
    art::preprocessing::LabelEncoder binary01(
        art::preprocessing::LabelEncodingStrategy::Binary01
    );
    const Eigen::VectorXd binary01_values = std::get<Eigen::VectorXd>(
        binary01.fit_transform(binary_labels)
    );
    assert(binary01_values(0) == 0.0);

    art::preprocessing::LabelEncoder signed_binary(
        art::preprocessing::LabelEncodingStrategy::BinarySigned
    );
    signed_binary.fit(binary_labels);
    assert(signed_binary.transform_vector(binary_labels)(0) == -1.0);
    assert(signed_binary.transform_vector(binary_labels)(1) == 1.0);

    art::preprocessing::LabelEncoder ignore_unknown(
        art::preprocessing::LabelEncodingStrategy::OneHot,
        art::preprocessing::UnknownLabelPolicy::Ignore
    );
    ignore_unknown.fit(iris_labels);
    assert(ignore_unknown.transform_matrix({"unknown"}).row(0).sum() == 0.0);
    expect_invalid([&] { auto_encoder.transform_matrix({"unknown"}); });

    return 0;
}
