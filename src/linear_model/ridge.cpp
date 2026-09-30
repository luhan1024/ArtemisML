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
#include "art/linear_model/ridge.h"

#include "art/base/errors.h"
#include "art/io/model.h"
#include "art/loss/loss.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace art::linear_model
{
namespace
{
    void validateDataset(const data::Dataset& dataset)
    {
        if (dataset.sample_count() == 0 || dataset.feature_count() == 0)
            throw std::invalid_argument("Ridge requires non-empty features and labels");
        if (dataset.features.rows() != dataset.labels.size())
            throw std::invalid_argument("Feature and label sample counts do not match");
        if (!dataset.features.allFinite() || !dataset.labels.allFinite())
            throw std::invalid_argument("Ridge data contains a non-finite value");
    }

    void validateParameters(const optim::ParameterVector& parameters, std::size_t count)
    {
        if (parameters.size() != static_cast<Eigen::Index>(count))
            throw std::invalid_argument("Ridge parameter dimension is invalid");
    }

    double r2(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets)
    {
        if (predictions.size() != targets.size() || targets.size() == 0)
            throw std::invalid_argument("Prediction and target sample counts do not match");
        const double total = (targets.array() - targets.mean()).square().sum();
        if (total == 0.0)
            return predictions.isApprox(targets) ? 1.0 : 0.0;
        return 1.0 - (predictions - targets).squaredNorm() / total;
    }
}

RidgeProblem::RidgeProblem(const data::Dataset& dataset, double alpha)
    : dataset_(dataset), alpha_(alpha)
{
    validateDataset(dataset_);
    if (!std::isfinite(alpha_) || alpha_ < 0.0)
        throw std::invalid_argument("Ridge alpha must be finite and non-negative");
}

double RidgeProblem::value(const optim::ParameterVector& parameters) const
{
    validateParameters(parameters, parameter_count());
    const auto residuals = dataset_.features * parameters.head(dataset_.feature_count()) +
        Eigen::VectorXd::Constant(dataset_.sample_count(), parameters(dataset_.feature_count())) - dataset_.labels;
    return 0.5 * residuals.squaredNorm() / dataset_.sample_count() +
        0.5 * alpha_ * parameters.head(dataset_.feature_count()).squaredNorm();
}

optim::ParameterVector RidgeProblem::gradient(const optim::ParameterVector& parameters) const
{
    validateParameters(parameters, parameter_count());
    const auto residuals = dataset_.features * parameters.head(dataset_.feature_count()) +
        Eigen::VectorXd::Constant(dataset_.sample_count(), parameters(dataset_.feature_count())) - dataset_.labels;
    optim::ParameterVector result(parameter_count());
    result.head(dataset_.feature_count()) = dataset_.features.transpose() * residuals / dataset_.sample_count() +
        alpha_ * parameters.head(dataset_.feature_count());
    result(dataset_.feature_count()) = residuals.sum() / dataset_.sample_count();
    return result;
}

optim::HessianMatrix RidgeProblem::hessian(const optim::ParameterVector& parameters) const
{
    validateParameters(parameters, parameter_count());
    const auto p = static_cast<Eigen::Index>(dataset_.feature_count());
    const auto n = static_cast<double>(dataset_.sample_count());
    Eigen::MatrixXd design(dataset_.features.rows(), p + 1);
    design.leftCols(p) = dataset_.features;
    design.col(p).setOnes();
    Eigen::MatrixXd result = design.transpose() * design / n;
    result.topLeftCorner(p, p).diagonal().array() += alpha_;
    return result;
}

std::size_t RidgeProblem::parameter_count() const { return dataset_.feature_count() + 1; }

Ridge::Ridge(double alpha) : alpha_(alpha)
{
    if (!std::isfinite(alpha_) || alpha_ < 0.0)
        throw std::invalid_argument("Ridge alpha must be finite and non-negative");
}

optim::OptimizationResult Ridge::fit(const data::Dataset& dataset, const optim::Optimizer& optimizer, const optim::OptimizerOptions& options)
{
    mark_unfitted();
    parameters_.resize(0);
    feature_count_ = 0;
    last_result_ = {};

    RidgeProblem problem(dataset, alpha_);
    if (options.max_iterations == 0)
        throw std::invalid_argument("Ridge requires at least one optimization iteration");
    const auto initial = optim::ParameterVector::Zero(static_cast<Eigen::Index>(problem.parameter_count()));
    optim::OptimizerState state;
    optim::ParameterVector parameters = initial;
    last_result_ = {};

    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        const optim::OptimizationStepResult step =
            optimizer.step(problem, parameters, state, options);
        parameters = step.parameters;
        const optim::ParameterVector next_gradient = problem.gradient(parameters);
        if (next_gradient.size() != parameters.size() || !next_gradient.allFinite())
        {
            throw std::runtime_error("Ridge produced a non-finite gradient");
        }
        const double final_value = problem.value(parameters);
        if (!std::isfinite(final_value))
        {
            throw std::runtime_error("Ridge produced a non-finite objective value");
        }
        const bool converged =
            next_gradient.norm() <= options.tolerance ||
            step.step_norm <= options.tolerance;
        last_result_ = {parameters, final_value, state.step, converged};
        if (converged)
            break;
    }

    parameters_ = last_result_.parameters;
    feature_count_ = dataset.feature_count();
    mark_fitted();
    return last_result_;
}

double Ridge::alpha() const noexcept { return alpha_; }
const optim::ParameterVector& Ridge::parameters() const
{
    if (!is_fitted()) throw base::NotFittedError("parameters()");
    return parameters_;
}
std::size_t Ridge::feature_count() const noexcept { return feature_count_; }

double Ridge::mean_squared_error(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& labels
) const
{
    if (labels.size() == 0 || features.rows() != labels.size())
        throw std::invalid_argument("Prediction features and labels do not match");
    return (predict(features) - labels).squaredNorm() /
        static_cast<double>(labels.size());
}

double Ridge::mean_absolute_error(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& labels
) const
{
    if (labels.size() == 0 || features.rows() != labels.size())
        throw std::invalid_argument("Prediction features and labels do not match");
    return (predict(features) - labels).array().abs().mean();
}

double Ridge::r2_score(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& labels
) const
{
    return r2(predict(features), labels);
}

void Ridge::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}

const char* Ridge::type_name() const noexcept
{
    return "art::linear_model::Ridge";
}

void Ridge::restore_state(
    const optim::ParameterVector& parameters,
    std::size_t feature_count,
    double alpha,
    std::size_t iterations,
    bool converged
)
{
    if (feature_count == 0 || parameters.size() !=
        static_cast<Eigen::Index>(feature_count + 1) ||
        !parameters.allFinite() || !std::isfinite(alpha) || alpha < 0.0 ||
        iterations == 0)
    {
        throw std::invalid_argument("Invalid Ridge serialized state");
    }
    alpha_ = alpha;
    parameters_ = parameters;
    feature_count_ = feature_count;
    last_result_ = {parameters, 0.0, iterations, converged};
    mark_fitted();
}

std::size_t Ridge::iterations() const noexcept
{
    return last_result_.iterations;
}

bool Ridge::converged() const noexcept
{
    return last_result_.converged;
}

void Ridge::do_fit(const base::FeatureInput& features, const base::TargetInput& targets)
{
    data::Dataset dataset;
    dataset.features = features;
    dataset.labels = targets;
    optim::NewtonOptimizer optimizer;
    optim::OptimizerOptions options;
    options.max_iterations = 100;
    options.tolerance = 1e-10;
    fit(dataset, optimizer, options);
}

base::PredictionOutput Ridge::do_predict(const base::FeatureInput& features) const
{
    if (features.cols() != static_cast<Eigen::Index>(feature_count_))
        throw std::invalid_argument("Prediction feature dimension does not match model");
    const auto p = static_cast<Eigen::Index>(feature_count_);
    return features * parameters_.head(p) + Eigen::VectorXd::Constant(features.rows(), parameters_(p));
}

double Ridge::do_score(const base::FeatureInput& features, const base::TargetInput& targets) const
{
    return r2(do_predict(features), targets);
}

namespace
{
    class RidgeClassifierProblem final : public optim::OptimizationProblem
    {
    public:
        RidgeClassifierProblem(
            Eigen::MatrixXd design,
            Eigen::MatrixXd targets,
            double alpha
        )
            : design_(std::move(design)),
              targets_(std::move(targets)),
              alpha_(alpha) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), design_.cols(), targets_.cols()
            );
            const Eigen::MatrixXd prediction = design_ * weights;
            loss::MeanSquaredError loss;
            return loss.value(prediction, targets_) +
                alpha_ * weights.topRows(weights.rows() - 1).squaredNorm() / 2.0;
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), design_.cols(), targets_.cols()
            );
            const Eigen::MatrixXd prediction = design_ * weights;
            loss::MeanSquaredError loss;
            Eigen::MatrixXd result = design_.transpose() *
                loss.gradient(prediction, targets_);
            result.topRows(result.rows() - 1) +=
                alpha_ * weights.topRows(weights.rows() - 1);
            return Eigen::Map<const optim::ParameterVector>(
                result.data(), result.size()
            );
        }

        optim::HessianMatrix hessian(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), design_.cols(), targets_.cols()
            );
            (void)weights;
            const double scale = 2.0 /
                static_cast<double>(targets_.size());
            Eigen::MatrixXd result = scale * design_.transpose() * design_;
            result.topLeftCorner(result.rows() - 1, result.cols() - 1).diagonal().array() += alpha_;
            return result;
        }

    private:
        Eigen::MatrixXd design_;
        Eigen::MatrixXd targets_;
        double alpha_;
    };

    Eigen::MatrixXd classifier_design(const Eigen::MatrixXd& features)
    {
        Eigen::MatrixXd result(features.rows(), features.cols() + 1);
        result.leftCols(features.cols()) = features;
        result.col(features.cols()).setOnes();
        return result;
    }

    Eigen::MatrixXd normalize_classifier_targets(
        const Eigen::VectorXd& targets,
        std::size_t& class_count
    )
    {
        if (targets.size() == 0 || !targets.allFinite())
            throw std::invalid_argument("RidgeClassifier targets are invalid");
        bool signed_binary = true;
        bool binary = true;
        double maximum = -1.0;
        for (Eigen::Index index = 0; index < targets.size(); ++index)
        {
            const double value = targets(index);
            if (value != -1.0 && value != 1.0) signed_binary = false;
            if (value != 0.0 && value != 1.0) binary = false;
            if (std::abs(value - std::round(value)) > 1e-12 || value < 0.0)
                continue;
            maximum = std::max(maximum, value);
        }
        if (signed_binary || binary)
        {
            class_count = 2;
            Eigen::MatrixXd result(targets.size(), 1);
            for (Eigen::Index index = 0; index < targets.size(); ++index)
                result(index, 0) = signed_binary ? (targets(index) + 1.0) / 2.0 : targets(index);
            return result;
        }
        if (maximum < 1.0)
            throw std::invalid_argument("RidgeClassifier requires at least two classes");
        class_count = static_cast<std::size_t>(maximum + 1.0);
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(targets.size(), class_count);
        for (Eigen::Index row = 0; row < targets.size(); ++row)
        {
            const double rounded = std::round(targets(row));
            if (std::abs(targets(row) - rounded) > 1e-12 ||
                rounded >= static_cast<double>(class_count))
                throw std::invalid_argument("RidgeClassifier class indices are invalid");
            result(row, static_cast<Eigen::Index>(rounded)) = 1.0;
        }
        return result;
    }
}

RidgeClassifier::RidgeClassifier(double alpha) : alpha_(alpha)
{
    if (!std::isfinite(alpha_) || alpha_ < 0.0)
        throw std::invalid_argument("RidgeClassifier alpha must be finite and non-negative");
}

void RidgeClassifier::fit(
    const base::FeatureInput& features,
    const Eigen::MatrixXd& one_hot_targets
)
{
    if (features.rows() == 0 || features.cols() == 0 ||
        !features.allFinite() || one_hot_targets.rows() != features.rows() ||
        one_hot_targets.cols() < 2 || !one_hot_targets.allFinite())
        throw std::invalid_argument("RidgeClassifier feature or one-hot target matrix is invalid");
    if (one_hot_targets.cols() == 2)
    {
        fit_encoded(features, one_hot_targets.col(1));
    }
    else
    {
        fit_encoded(features, one_hot_targets);
    }
}

void RidgeClassifier::fit_encoded(
    const base::FeatureInput& features,
    const Eigen::MatrixXd& targets
)
{
    mark_unfitted();
    if (features.rows() == 0 || features.cols() == 0 ||
        !features.allFinite() || targets.rows() != features.rows() ||
        targets.cols() == 0 || !targets.allFinite())
        throw std::invalid_argument("RidgeClassifier feature or target matrix is invalid");
    const Eigen::MatrixXd design = classifier_design(features);
    RidgeClassifierProblem problem(design, targets, alpha_);
    optim::NewtonOptimizer optimizer;
    optim::OptimizerOptions options;
    options.max_iterations = 20;
    options.tolerance = 1e-10;
    optim::ParameterVector parameters =
        optim::ParameterVector::Zero(design.cols() * targets.cols());
    optim::OptimizerState state;
    for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
    {
        const auto step = optimizer.step(problem, parameters, state, options);
        parameters = step.parameters;
        if (step.step_norm <= options.tolerance) break;
    }
    coefficients_ = Eigen::Map<const Eigen::MatrixXd>(
        parameters.data(), design.cols(), targets.cols()
    );
    feature_count_ = static_cast<std::size_t>(features.cols());
    class_count_ = static_cast<std::size_t>(targets.cols() == 1 ? 2 : targets.cols());
    class_labels_.clear();
    for (std::size_t index = 0; index < class_count_; ++index)
        class_labels_.push_back(std::to_string(index));
    mark_fitted();
}

void RidgeClassifier::do_fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    std::size_t class_count = 0;
    const Eigen::MatrixXd encoded = normalize_classifier_targets(targets, class_count);
    (void)class_count;
    fit_encoded(features, encoded);
}

Eigen::MatrixXd RidgeClassifier::predict_proba(
    const base::FeatureInput& features
) const
{
    if (!is_fitted()) throw base::NotFittedError("predict_proba()");
    if (features.cols() != static_cast<Eigen::Index>(feature_count_))
        throw std::invalid_argument("Prediction feature dimension does not match model");
    const Eigen::MatrixXd scores = classifier_design(features) * coefficients_;
    if (class_count_ == 2)
    {
        Eigen::MatrixXd result(features.rows(), 2);
        for (Eigen::Index row = 0; row < features.rows(); ++row)
        {
            const double positive = std::clamp(scores(row, 0), 0.0, 1.0);
            result(row, 0) = 1.0 - positive;
            result(row, 1) = positive;
        }
        return result;
    }
    Eigen::MatrixXd result = scores;
    for (Eigen::Index row = 0; row < result.rows(); ++row)
    {
        const double sum = result.row(row).cwiseMax(0.0).sum();
        if (sum > 0.0) result.row(row) = result.row(row).cwiseMax(0.0) / sum;
    }
    return result;
}

base::PredictionOutput RidgeClassifier::do_predict(
    const base::FeatureInput& features
) const
{
    const Eigen::MatrixXd scores = classifier_design(features) * coefficients_;
    base::PredictionOutput result(features.rows());
    for (Eigen::Index row = 0; row < scores.rows(); ++row)
    {
        Eigen::Index index = 0;
        scores.row(row).maxCoeff(&index);
        result(row) = static_cast<double>(index);
    }
    return result;
}

double RidgeClassifier::do_score(
    const base::FeatureInput& features,
    const base::TargetInput& targets
) const
{
    if (features.rows() != targets.size() || targets.size() == 0)
        throw std::invalid_argument("Prediction features and targets do not match");
    const auto predictions = do_predict(features);
    Eigen::Index correct = 0;
    for (Eigen::Index index = 0; index < targets.size(); ++index)
    {
        double expected = targets(index);
        if (class_count_ == 2 && expected == -1.0) expected = 0.0;
        if (predictions(index) == expected) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(targets.size());
}

double RidgeClassifier::alpha() const noexcept { return alpha_; }
std::size_t RidgeClassifier::class_count() const noexcept { return class_count_; }
std::size_t RidgeClassifier::feature_count() const noexcept { return feature_count_; }
const Eigen::MatrixXd& RidgeClassifier::coefficients() const
{
    if (!is_fitted()) throw base::NotFittedError("coefficients()");
    return coefficients_;
}
const std::vector<std::string>& RidgeClassifier::class_labels() const noexcept
{
    return class_labels_;
}
void RidgeClassifier::set_class_labels(const std::vector<std::string>& labels)
{
    if (!is_fitted() || labels.size() != class_count_)
        throw std::invalid_argument("RidgeClassifier class labels are invalid");
    class_labels_ = labels;
}
void RidgeClassifier::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}
const char* RidgeClassifier::type_name() const noexcept
{
    return "art::linear_model::RidgeClassifier";
}

void RidgeClassifier::restore_state(
    const Eigen::MatrixXd& coefficients,
    std::size_t feature_count,
    std::size_t class_count,
    double alpha,
    const std::vector<std::string>& labels
)
{
    if (feature_count == 0 || class_count < 2 || !std::isfinite(alpha) || alpha < 0.0 ||
        coefficients.rows() != static_cast<Eigen::Index>(feature_count + 1) ||
        coefficients.cols() != static_cast<Eigen::Index>(class_count == 2 ? 1 : class_count) ||
        !coefficients.allFinite() || labels.size() != class_count)
        throw std::invalid_argument("Invalid RidgeClassifier serialized state");
    alpha_ = alpha;
    coefficients_ = coefficients;
    feature_count_ = feature_count;
    class_count_ = class_count;
    class_labels_ = labels;
    mark_fitted();
}
}
