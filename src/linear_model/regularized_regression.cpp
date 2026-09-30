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
#include "art/linear_model/lasso.h"
#include "art/linear_model/elastic_net.h"

#include "art/io/model.h"
#include "art/loss/loss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace art::linear_model
{
namespace
{
    void validate_data(
        const base::FeatureInput& features,
        const base::TargetInput& targets
    )
    {
        if (features.rows() == 0 || features.cols() == 0 ||
            targets.size() == 0 || features.rows() != targets.size() ||
            !features.allFinite() || !targets.allFinite())
        {
            throw std::invalid_argument(
                "Regularized regression features and targets are invalid"
            );
        }
    }

    Eigen::MatrixXd add_intercept(const Eigen::MatrixXd& features)
    {
        Eigen::MatrixXd result(features.rows(), features.cols() + 1);
        result.leftCols(features.cols()) = features;
        result.col(features.cols()).setOnes();
        return result;
    }

    double sign(double value)
    {
        return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
    }

    class RegularizedProblem final : public optim::OptimizationProblem
    {
    public:
        RegularizedProblem(
            Eigen::MatrixXd design,
            Eigen::MatrixXd targets,
            double alpha,
            double l1_ratio
        )
            : design_(std::move(design)),
              targets_(std::move(targets)),
              alpha_(alpha),
              l1_ratio_(l1_ratio) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), design_.cols(), 1
            );
            loss::MeanSquaredError loss;
            const double l1 = weights.topRows(weights.rows() - 1).array().abs().sum();
            const double l2 = weights.topRows(weights.rows() - 1).squaredNorm();
            return loss.value(design_ * weights, targets_) +
                alpha_ * l1_ratio_ * l1 +
                0.5 * alpha_ * (1.0 - l1_ratio_) * l2;
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), design_.cols(), 1
            );
            loss::MeanSquaredError loss;
            Eigen::MatrixXd result = design_.transpose() *
                loss.gradient(design_ * weights, targets_);
            for (Eigen::Index index = 0; index + 1 < result.rows(); ++index)
            {
                result(index, 0) += alpha_ * l1_ratio_ * sign(weights(index, 0)) +
                    alpha_ * (1.0 - l1_ratio_) * weights(index, 0);
            }
            return result;
        }

    private:
        Eigen::MatrixXd design_;
        Eigen::MatrixXd targets_;
        double alpha_;
        double l1_ratio_;
    };

    optim::ParameterVector fit_regularized(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        double alpha,
        double l1_ratio,
        const optim::Optimizer& optimizer
    )
    {
        validate_data(features, targets);
        if (!std::isfinite(alpha) || alpha < 0.0 ||
            !std::isfinite(l1_ratio) || l1_ratio < 0.0 || l1_ratio > 1.0)
        {
            throw std::invalid_argument("Regularized regression parameters are invalid");
        }
        const Eigen::MatrixXd design = add_intercept(features);
        const Eigen::MatrixXd matrix_targets = targets;
        RegularizedProblem problem(design, matrix_targets, alpha, l1_ratio);
        optim::OptimizerOptions options;
        options.max_iterations = 10000;
        options.tolerance = 1e-8;
        optim::ParameterVector parameters =
            optim::ParameterVector::Zero(design.cols());
        optim::OptimizerState state;
        for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
        {
            const optim::OptimizationStepResult step = optimizer.step(
                problem, parameters, state, options
            );
            parameters = step.parameters;
            if (step.step_norm <= options.tolerance) break;
        }
        return parameters;
    }

    RegressionMetrics regression_metrics(
        const base::Predictor& model,
        const Eigen::MatrixXd& features,
        const Eigen::VectorXd& targets
    )
    {
        const Eigen::VectorXd predictions = model.predict(features);
        if (targets.size() == 0 || predictions.size() != targets.size())
            throw std::invalid_argument("Prediction features and targets do not match");
        const Eigen::VectorXd residuals = predictions - targets;
        const double total = (targets.array() - targets.mean()).square().sum();
        return {
            residuals.squaredNorm() / static_cast<double>(targets.size()),
            residuals.array().abs().mean(),
            total == 0.0 ? (residuals.squaredNorm() == 0.0 ? 1.0 : 0.0) :
                1.0 - residuals.squaredNorm() / total
        };
    }
}

Lasso::Lasso(double alpha) : alpha_(alpha)
{
    if (!std::isfinite(alpha_) || alpha_ < 0.0)
        throw std::invalid_argument("Lasso alpha must be finite and non-negative");
}

void Lasso::fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets,
    const optim::Optimizer& optimizer
)
{
    active_optimizer_ = &optimizer;
    try { base::Predictor::fit(features, targets); }
    catch (...) { active_optimizer_ = nullptr; throw; }
    active_optimizer_ = nullptr;
}

void Lasso::do_fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    optim::GradientDescent default_optimizer;
    const optim::Optimizer& optimizer = active_optimizer_ == nullptr
        ? static_cast<const optim::Optimizer&>(default_optimizer)
        : *active_optimizer_;
    parameters_ = fit_regularized(features, targets, alpha_, 1.0, optimizer);
    feature_count_ = static_cast<std::size_t>(features.cols());
    mark_fitted();
}

base::PredictionOutput Lasso::do_predict(
    const base::FeatureInput& features
) const
{
    if (features.cols() != static_cast<Eigen::Index>(feature_count_))
        throw std::invalid_argument("Prediction feature dimension does not match model");
    return features * parameters_.head(features.cols()) +
        Eigen::VectorXd::Constant(features.rows(), parameters_(features.cols()));
}

double Lasso::do_score(
    const base::FeatureInput& features,
    const base::TargetInput& targets
) const
{
    return regression_metrics(*this, features, targets).r2_score;
}

double Lasso::alpha() const noexcept { return alpha_; }
std::size_t Lasso::feature_count() const noexcept { return feature_count_; }
const optim::ParameterVector& Lasso::parameters() const
{
    if (!is_fitted()) throw base::NotFittedError("parameters()");
    return parameters_;
}
RegressionMetrics Lasso::evaluate(
    const Eigen::MatrixXd& features, const Eigen::VectorXd& targets
) const { return regression_metrics(*this, features, targets); }
void Lasso::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}
std::string Lasso::type_name() const
{
    return "art::linear_model::Lasso";
}
void Lasso::restore_state(
    const optim::ParameterVector& parameters,
    std::size_t feature_count,
    double alpha
)
{
    if (feature_count == 0 || parameters.size() !=
        static_cast<Eigen::Index>(feature_count + 1) || !parameters.allFinite() ||
        !std::isfinite(alpha) || alpha < 0.0)
        throw std::invalid_argument("Invalid Lasso serialized state");
    alpha_ = alpha;
    parameters_ = parameters;
    feature_count_ = feature_count;
    mark_fitted();
}

ElasticNet::ElasticNet(double alpha, double l1_ratio)
    : alpha_(alpha), l1_ratio_(l1_ratio)
{
    if (!std::isfinite(alpha_) || alpha_ < 0.0 ||
        !std::isfinite(l1_ratio_) || l1_ratio_ < 0.0 || l1_ratio_ > 1.0)
        throw std::invalid_argument("ElasticNet parameters are invalid");
}

void ElasticNet::fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets,
    const optim::Optimizer& optimizer
)
{
    active_optimizer_ = &optimizer;
    try { base::Predictor::fit(features, targets); }
    catch (...) { active_optimizer_ = nullptr; throw; }
    active_optimizer_ = nullptr;
}

void ElasticNet::do_fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    optim::GradientDescent default_optimizer;
    const optim::Optimizer& optimizer = active_optimizer_ == nullptr
        ? static_cast<const optim::Optimizer&>(default_optimizer)
        : *active_optimizer_;
    parameters_ = fit_regularized(
        features, targets, alpha_, l1_ratio_, optimizer
    );
    feature_count_ = static_cast<std::size_t>(features.cols());
    mark_fitted();
}

base::PredictionOutput ElasticNet::do_predict(
    const base::FeatureInput& features
) const
{
    if (features.cols() != static_cast<Eigen::Index>(feature_count_))
        throw std::invalid_argument("Prediction feature dimension does not match model");
    return features * parameters_.head(features.cols()) +
        Eigen::VectorXd::Constant(features.rows(), parameters_(features.cols()));
}

double ElasticNet::do_score(
    const base::FeatureInput& features,
    const base::TargetInput& targets
) const
{
    return regression_metrics(*this, features, targets).r2_score;
}

double ElasticNet::alpha() const noexcept { return alpha_; }
double ElasticNet::l1_ratio() const noexcept { return l1_ratio_; }
std::size_t ElasticNet::feature_count() const noexcept { return feature_count_; }
const optim::ParameterVector& ElasticNet::parameters() const
{
    if (!is_fitted()) throw base::NotFittedError("parameters()");
    return parameters_;
}
RegressionMetrics ElasticNet::evaluate(
    const Eigen::MatrixXd& features, const Eigen::VectorXd& targets
) const { return regression_metrics(*this, features, targets); }
void ElasticNet::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}
std::string ElasticNet::type_name() const
{
    return "art::linear_model::ElasticNet";
}
void ElasticNet::restore_state(
    const optim::ParameterVector& parameters,
    std::size_t feature_count,
    double alpha,
    double l1_ratio
)
{
    if (feature_count == 0 || parameters.size() !=
        static_cast<Eigen::Index>(feature_count + 1) || !parameters.allFinite() ||
        !std::isfinite(alpha) || alpha < 0.0 || !std::isfinite(l1_ratio) ||
        l1_ratio < 0.0 || l1_ratio > 1.0)
        throw std::invalid_argument("Invalid ElasticNet serialized state");
    alpha_ = alpha;
    l1_ratio_ = l1_ratio;
    parameters_ = parameters;
    feature_count_ = feature_count;
    mark_fitted();
}
}
