/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #      #   #    #       #   #  #  #
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
#include "art/linear_model/elastic_net.h"
#include "art/linear_model/lasso.h"

#include <Eigen/Dense>

#include <cassert>
#include <cstdio>
#include <stdexcept>

int main()
{
    const Eigen::MatrixXd features =
        (Eigen::MatrixXd(6, 2) << -3.0, 1.0, -2.0, 1.0, -1.0, 1.0,
                                   1.0, 1.0, 2.0, 1.0, 3.0, 1.0).finished();
    const Eigen::VectorXd targets =
        (Eigen::VectorXd(6) << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0).finished();

    art::linear_model::Lasso lasso(0.01);
    lasso.fit(features, targets);
    const auto lasso_metrics = lasso.evaluate(features, targets);
    assert(lasso.is_fitted());
    assert(lasso_metrics.mean_squared_error >= 0.0);
    lasso.save("model_lasso.artemisml");
    const art::Model loaded_lasso = art::load("model_lasso.artemisml");
    assert(loaded_lasso.type_name() == lasso.type_name());
    assert(loaded_lasso.predict(features).isApprox(lasso.predict(features)));
    const auto loaded_lasso_metrics =
        loaded_lasso.regression_metrics(features, targets);
    assert(loaded_lasso_metrics.r2_score == lasso.score(features, targets));
    loaded_lasso.save("model_lasso_copy.artemisml");

    art::linear_model::ElasticNet elastic_net(0.01, 0.25);
    elastic_net.fit(features, targets);
    const auto elastic_metrics = elastic_net.evaluate(features, targets);
    assert(elastic_net.is_fitted());
    assert(elastic_metrics.mean_absolute_error >= 0.0);
    elastic_net.save("model_elastic_net.artemisml");
    const art::Model loaded_elastic_net = art::load("model_elastic_net.artemisml");
    assert(loaded_elastic_net.type_name() == elastic_net.type_name());
    assert(loaded_elastic_net.predict(features).isApprox(
        elastic_net.predict(features)
    ));
    loaded_elastic_net.save("model_elastic_net_copy.artemisml");

    bool invalid_alpha = false;
    try { art::linear_model::Lasso invalid(-1.0); }
    catch (const std::invalid_argument&) { invalid_alpha = true; }
    assert(invalid_alpha);

    bool invalid_ratio = false;
    try { art::linear_model::ElasticNet invalid(1.0, 1.5); }
    catch (const std::invalid_argument&) { invalid_ratio = true; }
    assert(invalid_ratio);

    std::remove("model_lasso.artemisml");
    std::remove("model_lasso_copy.artemisml");
    std::remove("model_elastic_net.artemisml");
    std::remove("model_elastic_net_copy.artemisml");
    return 0;
}
