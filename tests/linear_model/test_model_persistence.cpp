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
#include "art/preprocessing/label_encoder.h"

#include <Eigen/Dense>

#include <cassert>
#include <cstdio>
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
        assert(loaded.evaluate(features, targets) ==
               original.score(features, targets));
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
    art::save(binary_model, "model_binary.artemisml");
    const art::Model loaded_binary = art::load("model_binary.artemisml");
    assert_same_predictions(
        binary_model, loaded_binary, binary_features, binary_targets
    );
    std::remove("model_binary.artemisml");

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
    art::save(multiclass_model, "model_multiclass.artemisml");
    const art::Model loaded_multiclass = art::load("model_multiclass.artemisml");
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
    art::save(text_model, "model_text.artemisml");
    const art::Model loaded_text = art::load("model_text.artemisml");
    assert_same_predictions(text_model, loaded_text, text_features, text_targets);
    std::remove("model_text.artemisml");

    std::cout << "Model persistence round-trip test passed.\n";
    return 0;
}
