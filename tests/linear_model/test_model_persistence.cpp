/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #   #  #    #      #   #  #  #  #
 *   #   #  #   #    #    #####  #   #  #####  ####   #   #  #####
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor  : Yihan Wang
 *
 * ============================================================================
 */
#include "art/io/model.h"
#include "art/io/csvReader.h"
#include "art/linear_model/linear_regression.h"
#include "art/preprocessing/label_encoder.h"

#include <Eigen/Dense>

#include <cassert>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <variant>

namespace
{
    void assert_same_predictions(
        const art::linear_model::LogisticRegression& original,
        const art::Model& loaded,
        const Eigen::MatrixXd& features,
        const Eigen::VectorXd& targets
    )
    {
        assert(loaded.is_fitted());
        assert(loaded.type_name() == original.type_name());
        assert(loaded.predict(features).isApprox(original.predict(features)));
        assert(loaded.predict_proba(features).isApprox(
            original.predict_proba(features)
        ));
        const auto metrics = std::get<art::linear_model::ClassificationMetrics>(
            loaded.evaluate(features, targets)
        );
        assert(metrics.accuracy == original.score(features, targets));
    }
}

int main()
{
    const Eigen::MatrixXd binary_features =
        (Eigen::MatrixXd(6, 1) << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0).finished();
    const Eigen::VectorXd binary_targets =
        (Eigen::VectorXd(6) << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished();
    art::linear_model::LogisticRegression binary_model;
    binary_model.fit(binary_features, binary_targets);
    binary_model.set_class_labels({"negative", "positive"});
    const std::filesystem::path binary_path = "model_binary.artemisml";
    binary_model.save(binary_path);
    const art::Model loaded_binary = art::load(binary_path.string());
    assert_same_predictions(
        binary_model, loaded_binary, binary_features, binary_targets
    );
    const std::filesystem::path binary_copy_path = "model_binary_copy.artemisml";
    loaded_binary.save(binary_copy_path);
    const art::Model loaded_binary_copy = art::load(binary_copy_path.string());
    assert_same_predictions(
        binary_model, loaded_binary_copy, binary_features, binary_targets
    );
    std::remove("model_binary.artemisml");
    std::remove("model_binary_copy.artemisml");

    const Eigen::MatrixXd multiclass_features =
        (Eigen::MatrixXd(9, 2) <<
            -3.0, -3.0, -2.0, -2.0, -1.0, -1.0,
             0.0, 3.0, 0.0, 2.0, 0.0, 1.0,
             1.0, 0.0, 2.0, 0.0, 3.0, 0.0).finished();
    const Eigen::VectorXd multiclass_targets =
        (Eigen::VectorXd(9) << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                               2.0, 2.0, 2.0).finished();
    art::linear_model::LogisticRegression multiclass_model;
    multiclass_model.fit(multiclass_features, multiclass_targets);
    multiclass_model.set_class_labels({"left", "middle", "right"});
    const std::filesystem::path multiclass_path = "model_multiclass.artemisml";
    art::save(multiclass_model, multiclass_path.string());
    const art::Model loaded_multiclass = art::load(multiclass_path.string());
    assert_same_predictions(
        multiclass_model, loaded_multiclass,
        multiclass_features, multiclass_targets
    );
    std::remove("model_multiclass.artemisml");

    const art::data::TextLabelDataset text_dataset =
        art::io::CsvReader().read_text_label_dataset("iris.csv", "species");
    Eigen::MatrixXd text_features(
        static_cast<Eigen::Index>(text_dataset.sample_count()),
        static_cast<Eigen::Index>(text_dataset.feature_count())
    );
    for (Eigen::Index row = 0; row < text_features.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < text_features.cols(); ++column)
        {
            text_features(row, column) = std::stod(
                text_dataset.features.at(
                    static_cast<std::size_t>(row),
                    static_cast<std::size_t>(column)
                )
            );
        }
    }
    art::preprocessing::LabelEncoder encoder(
        art::preprocessing::LabelEncodingStrategy::Ordinal
    );
    const Eigen::VectorXd text_targets = std::get<Eigen::VectorXd>(
        encoder.fit_transform(text_dataset.labels.values())
    );
    art::linear_model::LogisticRegression text_model;
    text_model.fit(text_features, text_targets);
    text_model.set_class_labels(encoder.classes());
    const std::filesystem::path text_path = "model_text.artemisml";
    art::save(text_model, text_path.string());
    const art::Model loaded_text = art::load(text_path.string());
    assert_same_predictions(text_model, loaded_text, text_features, text_targets);
    std::remove("model_text.artemisml");

    const art::data::Dataset regression_dataset =
        art::data::Dataset::from_csv("data.csv", "label");
    art::linear_model::LinearRegression regression_model;
    art::optim::NewtonOptimizer regression_optimizer;
    art::optim::OptimizerOptions regression_options;
    regression_options.max_iterations = 20;
    regression_options.tolerance = 1e-10;
    regression_model.fit(
        regression_dataset, regression_optimizer, regression_options
    );
    const std::filesystem::path regression_path = "model_regression.artemisml";
    regression_model.save(regression_path);
    const art::Model loaded_regression = art::load(regression_path.string());
    assert(loaded_regression.is_fitted());
    assert(loaded_regression.type_name() == regression_model.type_name());
    assert(loaded_regression.predict(regression_dataset.features).isApprox(
        regression_model.predict(regression_dataset.features)
    ));
    const auto loaded_regression_result =
        std::get<art::linear_model::RegressionMetrics>(
            loaded_regression.evaluate(
                regression_dataset.features, regression_dataset.labels
            )
        );
    assert(loaded_regression_result.r2_score == regression_model.r2_score(
        regression_dataset.features, regression_dataset.labels
    ));
    const art::linear_model::RegressionMetrics regression_metrics =
        loaded_regression.regression_metrics(
            regression_dataset.features, regression_dataset.labels
        );
    const art::linear_model::RegressionMetrics expected_metrics =
        regression_model.evaluate(
            regression_dataset.features, regression_dataset.labels
        );
    assert(regression_metrics.mean_squared_error ==
           expected_metrics.mean_squared_error);
    assert(regression_metrics.mean_absolute_error ==
           expected_metrics.mean_absolute_error);
    assert(regression_metrics.r2_score == expected_metrics.r2_score);
    const std::filesystem::path regression_copy_path =
        "model_regression_copy.artemisml";
    loaded_regression.save(regression_copy_path);
    const art::Model loaded_regression_copy =
        art::load(regression_copy_path.string());
    assert(loaded_regression_copy.predict(regression_dataset.features).isApprox(
        regression_model.predict(regression_dataset.features)
    ));
    std::remove("model_regression.artemisml");
    std::remove("model_regression_copy.artemisml");

    art::linear_model::Ridge ridge_model(0.1);
    ridge_model.fit(
        regression_dataset, regression_optimizer, regression_options
    );
    ridge_model.save("model_ridge.artemisml");
    const art::Model loaded_ridge = art::load("model_ridge.artemisml");
    assert(loaded_ridge.type_name() == ridge_model.type_name());
    assert(loaded_ridge.predict(regression_dataset.features).isApprox(
        ridge_model.predict(regression_dataset.features)
    ));
    const auto ridge_metrics = loaded_ridge.regression_metrics(
        regression_dataset.features, regression_dataset.labels
    );
    assert(ridge_metrics.r2_score == ridge_model.r2_score(
        regression_dataset.features, regression_dataset.labels
    ));
    const std::filesystem::path ridge_copy_path = "model_ridge_copy.artemisml";
    loaded_ridge.save(ridge_copy_path);
    std::remove("model_ridge.artemisml");
    std::remove("model_ridge_copy.artemisml");

    const Eigen::MatrixXd classifier_features =
        (Eigen::MatrixXd(6, 1) << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0).finished();
    const Eigen::VectorXd classifier_targets =
        (Eigen::VectorXd(6) << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished();
    art::linear_model::RidgeClassifier ridge_classifier;
    ridge_classifier.fit(classifier_features, classifier_targets);
    ridge_classifier.set_class_labels({"negative", "positive"});
    ridge_classifier.save("model_ridge_classifier.artemisml");
    const art::Model loaded_ridge_classifier =
        art::load("model_ridge_classifier.artemisml");
    assert(loaded_ridge_classifier.type_name() == ridge_classifier.type_name());
    assert(loaded_ridge_classifier.predict(classifier_features).isApprox(
        ridge_classifier.predict(classifier_features)
    ));
    const auto ridge_classifier_metrics =
        std::get<art::linear_model::ClassificationMetrics>(
            loaded_ridge_classifier.evaluate(
                classifier_features, classifier_targets
            )
        );
    assert(ridge_classifier_metrics.accuracy == ridge_classifier.score(
        classifier_features, classifier_targets
    ));
    std::remove("model_ridge_classifier.artemisml");

    std::cout << "Model persistence round-trip test passed.\n";
    return 0;
}
