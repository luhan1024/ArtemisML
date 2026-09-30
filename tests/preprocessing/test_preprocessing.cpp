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

#include <cassert>
#include <cmath>
#include <memory>
#include <limits>
#include <stdexcept>

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
    void expect_not_fitted(Function&& function)
    {
        bool raised = false;
        try { function(); }
        catch (const art::base::NotFittedError&) { raised = true; }
        assert(raised);
    }
}

int main()
{
    const Eigen::MatrixXd train =
        (Eigen::MatrixXd(3, 2) << 1.0, 10.0, 2.0, 20.0, 3.0, 30.0).finished();

    art::preprocessing::StandardScaler scaler;
    expect_not_fitted([&] { scaler.transform(train); });
    scaler.fit(train);
    assert(scaler.is_fitted());
    const Eigen::MatrixXd scaled = scaler.transform(train);
    assert(std::abs(scaled.col(0).mean()) < 1e-12);
    assert(std::abs(scaled(0, 0) + 1.224744871391589) < 1e-10);
    const Eigen::MatrixXd test =
        (Eigen::MatrixXd(1, 2) << 4.0, 40.0).finished();
    assert(std::abs(scaler.transform(test)(0, 0) - 2.449489742783178) < 1e-10);
    scaler.fit((Eigen::MatrixXd(2, 2) << 100.0, 100.0, 102.0, 102.0).finished());
    assert(std::abs(scaler.transform(test)(0, 0) + 97.0) < 1e-12);
    expect_invalid([&] { scaler.transform(Eigen::MatrixXd::Zero(1, 1)); });

    art::preprocessing::MinMaxScaler minmax(-1.0, 1.0);
    const Eigen::MatrixXd minmax_output = minmax.fit_transform(train);
    assert(std::abs(minmax_output(0, 0) + 1.0) < 1e-12);
    assert(std::abs(minmax_output(2, 1) - 1.0) < 1e-12);
    assert(minmax_output.rows() == train.rows());

    const Eigen::MatrixXd missing =
        (Eigen::MatrixXd(3, 2) << 1.0, 10.0,
                                   std::numeric_limits<double>::quiet_NaN(), 20.0,
                                   3.0, 30.0).finished();
    art::preprocessing::Imputer imputer;
    const Eigen::MatrixXd filled = imputer.fit_transform(missing);
    assert(std::abs(filled(1, 0) - 2.0) < 1e-12);
    assert(std::abs(filled(1, 1) - 20.0) < 1e-12);
    art::preprocessing::Imputer constant(
        art::preprocessing::ImputationStrategy::Constant, -7.0
    );
    assert(constant.fit_transform(missing)(1, 0) == -7.0);

    const Eigen::MatrixXd categories =
        (Eigen::MatrixXd(3, 2) << 2.0, 0.0, 1.0, 1.0, 2.0, 1.0).finished();
    art::preprocessing::OneHotEncoder encoder(
        art::preprocessing::UnknownCategoryPolicy::Ignore
    );
    const Eigen::MatrixXd encoded = encoder.fit_transform(categories);
    assert(encoded.rows() == categories.rows());
    assert(encoded.cols() == 4);
    assert(encoded(0, 1) == 1.0);
    assert(encoded(0, 2) == 1.0);
    const Eigen::MatrixXd unknown = (Eigen::MatrixXd(1, 2) << 9.0, 9.0).finished();
    assert(encoder.transform(unknown).row(0).sum() == 0.0);
    art::preprocessing::OneHotEncoder strict;
    strict.fit(categories);
    expect_invalid([&] { strict.transform(unknown); });

    auto column_scaler = std::make_shared<art::preprocessing::StandardScaler>();
    auto column_encoder = std::make_shared<art::preprocessing::OneHotEncoder>(
        art::preprocessing::UnknownCategoryPolicy::Ignore
    );
    art::preprocessing::ColumnTransformer columns({
        {"numeric", {0}, column_scaler},
        {"categorical", {1}, column_encoder}
    });
    const Eigen::MatrixXd combined = columns.fit_transform(categories);
    assert(combined.rows() == categories.rows());
    assert(combined.cols() == 3);
    assert(columns.output_feature_count() == 3);

    art::preprocessing::SelectColumns select({1, 0});
    assert(select.fit_transform(train)(0, 0) == 10.0);

    art::preprocessing::PolynomialFeatures polynomial(2, true);
    const Eigen::MatrixXd generated = polynomial.fit_transform(
        (Eigen::MatrixXd(1, 2) << 2.0, 3.0).finished()
    );
    assert(generated.cols() == 6);
    assert(generated(0, 0) == 1.0);
    assert(generated(0, 1) == 2.0);
    assert(generated(0, 2) == 3.0);
    assert(generated(0, 5) == 9.0);

    return 0;
}
