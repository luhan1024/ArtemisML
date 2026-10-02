#include "art/io/model.h"
#include "art/linear_model/elastic_net.h"
#include "art/linear_model/linear_regression.h"
#include "art/linear_model/lasso.h"
#include "art/linear_model/ridge.h"
#include "art/linear_model/logistic_regression.h"
#include "art/optim/optimizer.h"
#include "art/preprocessing/label_encoder.h"

#include <Eigen/Dense>

#include <cassert>
#include <cmath>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace
{
    void assert_finite(const Eigen::VectorXd& values)
    {
        assert(values.allFinite());
    }

    void assert_finite(const Eigen::MatrixXd& values)
    {
        assert(values.allFinite());
    }

    Eigen::MatrixXd regression_features()
    {
        return (Eigen::MatrixXd(6, 2) <<
            -3.0, 1.0, -2.0, 1.0, -1.0, 1.0,
             1.0, 1.0,  2.0, 1.0,  3.0, 1.0).finished();
    }

    Eigen::VectorXd regression_targets()
    {
        return (Eigen::VectorXd(6) << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0).finished();
    }

    Eigen::MatrixXd binary_features()
    {
        return (Eigen::MatrixXd(6, 1) << -3.0, -2.0, -1.0, 1.0, 2.0, 3.0).finished();
    }

    Eigen::VectorXd binary_targets_01()
    {
        return (Eigen::VectorXd(6) << 0.0, 0.0, 0.0, 1.0, 1.0, 1.0).finished();
    }

    Eigen::VectorXd binary_targets_pm1()
    {
        return (Eigen::VectorXd(6) << -1.0, -1.0, -1.0, 1.0, 1.0, 1.0).finished();
    }

    Eigen::MatrixXd multiclass_one_hot()
    {
        Eigen::MatrixXd targets = Eigen::MatrixXd::Zero(6, 3);
        targets(0, 0) = 1.0;
        targets(1, 0) = 1.0;
        targets(2, 1) = 1.0;
        targets(3, 1) = 1.0;
        targets(4, 2) = 1.0;
        targets(5, 2) = 1.0;
        return targets;
    }
}

int main()
{
    const Eigen::MatrixXd x = regression_features();
    const Eigen::VectorXd y = regression_targets();
    const Eigen::MatrixXd binary_x = binary_features();
    const Eigen::VectorXd y01 = binary_targets_01();

    art::optim::GradientDescent gradient_descent(1e-3);
    art::optim::NewtonOptimizer newton;
    art::optim::OptimizerOptions options;
    options.max_iterations = 200;
    options.tolerance = 1e-8;

    // LinearRegression: both base fit(X,y) and explicit Dataset/optimizer fit.
    art::linear_model::LinearRegression linear;
    art::base::Regressor& linear_base = linear;
    linear_base.fit(x, y);
    assert(linear.is_fitted());
    assert_finite(linear.predict(x));
    assert(std::isfinite(linear.evaluate(x, y).r2_score));

    art::data::Dataset dataset;
    dataset.features = x;
    dataset.labels = y;
    art::linear_model::LinearRegression explicit_linear;
    const auto linear_result = explicit_linear.fit(dataset, newton, options);
    assert(linear_result.parameters.size() == x.cols() + 1);
    assert_finite(explicit_linear.predict(x));

    // Ridge: base API and explicit optimizer API must produce fitted models.
    art::linear_model::Ridge ridge(0.1);
    art::base::Regressor& ridge_base = ridge;
    ridge_base.fit(x, y);
    assert(ridge.is_fitted());
    assert(ridge.parameters().size() == x.cols() + 1);
    assert(std::isfinite(ridge.evaluate(x, y).mean_squared_error));

    art::linear_model::Ridge explicit_ridge(0.1);
    const auto ridge_result = explicit_ridge.fit(dataset, gradient_descent, options);
    assert(ridge_result.parameters.size() == x.cols() + 1);

    // Lasso and ElasticNet expose the same X/y plus explicit-optimizer path.
    art::linear_model::Lasso lasso(0.01);
    lasso.fit(x, y);
    assert(lasso.is_fitted());
    assert(std::isfinite(lasso.evaluate(x, y).mean_absolute_error));
    art::linear_model::Lasso explicit_lasso(0.01);
    explicit_lasso.fit(x, y, gradient_descent);
    assert(explicit_lasso.is_fitted());

    art::linear_model::ElasticNet elastic_net(0.01, 0.25);
    elastic_net.fit(x, y);
    assert(elastic_net.is_fitted());
    assert(std::isfinite(elastic_net.evaluate(x, y).mean_squared_error));
    art::linear_model::ElasticNet explicit_elastic_net(0.01, 0.25);
    explicit_elastic_net.fit(x, y, gradient_descent);
    assert(explicit_elastic_net.is_fitted());

    // RidgeClassifier: binary 0/1 and -1/1 plus one-hot multiclass.
    art::linear_model::RidgeClassifier ridge_classifier;
    ridge_classifier.fit(binary_x, y01);
    assert(ridge_classifier.is_fitted());
    assert(std::isfinite(ridge_classifier.score(binary_x, y01)));
    ridge_classifier.fit(binary_x, binary_targets_pm1());
    assert(ridge_classifier.is_fitted());
    ridge_classifier.fit(binary_x, multiclass_one_hot());
    assert(ridge_classifier.class_count() == 3);
    assert_finite(ridge_classifier.decision_function(binary_x));

    // LogisticRegression: binary 0/1 and -1/1, default GD and injected Newton.
    art::linear_model::LogisticRegression logistic;
    logistic.fit(binary_x, y01);
    assert(logistic.class_count() == 2);
    assert_finite(logistic.predict_proba(binary_x));
    assert(!logistic.training_history().empty());

    art::linear_model::LogisticRegression logistic_pm1;
    logistic_pm1.fit(binary_x, binary_targets_pm1());
    assert(logistic_pm1.class_count() == 2);

    art::linear_model::LogisticRegression logistic_newton;
    logistic_newton.fit(binary_x, y01, newton);
    assert(logistic_newton.class_count() == 2);
    assert(!logistic_newton.loss_history().empty());

    // Text labels are encoded at the preprocessing boundary, then preserved
    // through the model label mapping and persistence API.
    const std::vector<std::string> text_labels =
        {"negative", "negative", "negative", "positive", "positive", "positive"};
    art::preprocessing::LabelEncoder encoder(
        art::preprocessing::LabelEncodingStrategy::Ordinal
    );
    const Eigen::VectorXd encoded_text = std::get<Eigen::VectorXd>(
        encoder.fit_transform(text_labels)
    );
    art::linear_model::LogisticRegression text_logistic;
    text_logistic.fit(binary_x, encoded_text);
    text_logistic.set_class_labels(encoder.classes());
    assert(text_logistic.class_labels() == encoder.classes());

    // Save/load must preserve predictions and avoid string/path overload ambiguity.
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "artemisml_unified_logistic.artemisml";
    text_logistic.save(path);
    const art::Model loaded = art::load(path.string());
    assert(loaded.is_fitted());
    assert(loaded.predict(binary_x).isApprox(text_logistic.predict(binary_x)));
    assert(loaded.predict_proba(binary_x).isApprox(
        text_logistic.predict_proba(binary_x)
    ));
    std::filesystem::remove(path);

    return 0;
}
