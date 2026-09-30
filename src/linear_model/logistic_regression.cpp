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
#include "art/linear_model/logistic_regression.h"
#include "art/io/model.h"

#include "art/loss/loss.h"
#include "art/optim/optimizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace art::linear_model
{
namespace
{
    void validate_features(const base::FeatureInput& features)
    {
        if (features.rows() == 0 || features.cols() == 0 || !features.allFinite())
        {
            throw std::invalid_argument(
                "LogisticRegression requires a non-empty finite feature matrix"
            );
        }
    }

    Eigen::Index class_count_from_targets(const base::TargetInput& targets)
    {
        if (targets.size() == 0 || !targets.allFinite())
        {
            throw std::invalid_argument(
                "LogisticRegression targets must be non-empty and finite"
            );
        }
        double maximum = -1.0;
        for (Eigen::Index index = 0; index < targets.size(); ++index)
        {
            const double value = targets(index);
            const double rounded = std::round(value);
            if (value < 0.0 || std::abs(value - rounded) > 1e-12)
            {
                throw std::invalid_argument(
                    "LogisticRegression targets must be integer class indices"
                );
            }
            maximum = std::max(maximum, rounded);
        }
        const double count = maximum + 1.0;
        if (count < 2.0 || count > static_cast<double>(std::numeric_limits<Eigen::Index>::max()))
        {
            throw std::invalid_argument(
                "LogisticRegression requires at least two classes"
            );
        }
        return static_cast<Eigen::Index>(count);
    }

    Eigen::MatrixXd add_intercept(const base::FeatureInput& features)
    {
        Eigen::MatrixXd result(features.rows(), features.cols() + 1);
        result.leftCols(features.cols()) = features;
        result.col(features.cols()).setOnes();
        return result;
    }

    Eigen::MatrixXd one_hot(const base::TargetInput& targets, Eigen::Index class_count)
    {
        Eigen::MatrixXd result = Eigen::MatrixXd::Zero(targets.size(), class_count);
        for (Eigen::Index row = 0; row < targets.size(); ++row)
        {
            result(row, static_cast<Eigen::Index>(targets(row))) = 1.0;
        }
        return result;
    }

    double sigmoid(double value)
    {
        if (value >= 0.0) return 1.0 / (1.0 + std::exp(-value));
        const double exponent = std::exp(value);
        return exponent / (1.0 + exponent);
    }

    class BinaryProblem final : public optim::OptimizationProblem
    {
    public:
        BinaryProblem(
            Eigen::MatrixXd features,
            Eigen::MatrixXd targets,
            double l2_penalty
        )
            : features_(std::move(features)),
              targets_(std::move(targets)),
              l2_penalty_(l2_penalty) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), 1
            );
            const double data_value = loss_.value(features_ * weights, targets_);
            return data_value + regularization_value(weights);
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), 1
            );
            const Eigen::MatrixXd logits = features_ * weights;
            Eigen::MatrixXd result =
                features_.transpose() * loss_.gradient(logits, targets_);
            add_regularization_gradient(result, weights);
            return result;
        }

        optim::HessianMatrix hessian(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), 1
            );
            const Eigen::VectorXd logits = features_ * weights;
            Eigen::VectorXd curvature(logits.size());
            for (Eigen::Index row = 0; row < logits.size(); ++row)
            {
                const double probability = sigmoid(logits(row));
                curvature(row) = probability * (1.0 - probability) /
                    static_cast<double>(features_.rows());
            }
            Eigen::MatrixXd result = features_.transpose() *
                curvature.asDiagonal() * features_;
            for (Eigen::Index index = 0; index + 1 < result.rows(); ++index)
            {
                result(index, index) += l2_penalty_ /
                    static_cast<double>(features_.rows());
            }
            return result;
        }

    private:
        double regularization_value(
            const Eigen::Map<const Eigen::MatrixXd>& weights
        ) const
        {
            return l2_penalty_ * weights.topRows(weights.rows() - 1).squaredNorm() /
                (2.0 * static_cast<double>(features_.rows()));
        }

        void add_regularization_gradient(
            Eigen::MatrixXd& gradient,
            const Eigen::Map<const Eigen::MatrixXd>& weights
        ) const
        {
            gradient.topRows(gradient.rows() - 1) +=
                l2_penalty_ * weights.topRows(weights.rows() - 1) /
                static_cast<double>(features_.rows());
        }

        Eigen::MatrixXd features_;
        Eigen::MatrixXd targets_;
        double l2_penalty_;
        loss::BinaryCrossEntropy loss_;
    };

    class MulticlassProblem final : public optim::OptimizationProblem
    {
    public:
        MulticlassProblem(
            Eigen::MatrixXd features,
            Eigen::MatrixXd targets,
            double l2_penalty
        )
            : features_(std::move(features)),
              targets_(std::move(targets)),
              l2_penalty_(l2_penalty) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), targets_.cols()
            );
            return loss_.value(features_ * weights, targets_) +
                l2_penalty_ * weights.topRows(weights.rows() - 1).squaredNorm() /
                (2.0 * static_cast<double>(features_.rows()));
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), targets_.cols()
            );
            const Eigen::MatrixXd logits = features_ * weights;
            Eigen::MatrixXd matrix_gradient = features_.transpose() *
                loss_.gradient(logits, targets_);
            matrix_gradient.topRows(matrix_gradient.rows() - 1) +=
                l2_penalty_ * weights.topRows(weights.rows() - 1) /
                static_cast<double>(features_.rows());
            return Eigen::Map<const optim::ParameterVector>(
                matrix_gradient.data(), matrix_gradient.size()
            );
        }

    private:
        Eigen::MatrixXd features_;
        Eigen::MatrixXd targets_;
        double l2_penalty_;
        loss::SoftmaxCrossEntropy loss_;
    };

    Eigen::MatrixXd softmax(const Eigen::MatrixXd& logits)
    {
        Eigen::MatrixXd result(logits.rows(), logits.cols());
        for (Eigen::Index row = 0; row < logits.rows(); ++row)
        {
            const double maximum = logits.row(row).maxCoeff();
            const Eigen::RowVectorXd exponentials =
                (logits.row(row).array() - maximum).exp();
            result.row(row) = exponentials / exponentials.sum();
        }
        return result;
    }

    optim::ParameterVector optimize(
        const optim::OptimizationProblem& problem,
        Eigen::Index parameter_count,
        const optim::Optimizer& optimizer,
        const LogisticRegressionOptions& options,
        std::vector<TrainingRecord>& training_history,
        std::vector<double>& loss_history
    )
    {
        if (!std::isfinite(options.tolerance) || options.tolerance < 0.0 ||
            options.max_iterations == 0 ||
            !std::isfinite(options.l2_penalty) || options.l2_penalty < 0.0 ||
            !std::isfinite(options.decision_threshold) ||
            options.decision_threshold <= 0.0 ||
            options.decision_threshold >= 1.0)
        {
            throw std::invalid_argument(
                "LogisticRegression optimization options are invalid"
            );
        }
        optim::OptimizerOptions optimizer_options;
        optimizer_options.max_iterations = options.max_iterations;
        optimizer_options.tolerance = options.tolerance;
        optim::ParameterVector parameters =
            optim::ParameterVector::Zero(parameter_count);
        optim::OptimizerState state;
        loss_history.clear();
        training_history.clear();
        loss_history.reserve(options.max_iterations);
        training_history.reserve(options.max_iterations);
        for (std::size_t iteration = 0; iteration < options.max_iterations; ++iteration)
        {
            const optim::OptimizationStepResult step = optimizer.step(
                problem, parameters, state, optimizer_options
            );
            loss_history.push_back(step.objective_value);
            training_history.push_back({
                step.objective_value,
                step.gradient.norm(),
                step.step_norm
            });
            parameters = step.parameters;
            if (step.step_norm <= options.tolerance) break;
        }
        return parameters;
    }
}

LogisticRegression::LogisticRegression(
    double learning_rate, std::size_t max_iterations, double tolerance
)
    : options_{max_iterations, tolerance, 0.0},
      default_optimizer_(learning_rate) {}

LogisticRegression::LogisticRegression(
    const LogisticRegressionOptions& options
)
    : options_(options) {}

void LogisticRegression::fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets,
    const optim::Optimizer& optimizer
)
{
    active_optimizer_ = &optimizer;
    try
    {
        base::Predictor::fit(features, targets);
    }
    catch (...)
    {
        active_optimizer_ = nullptr;
        throw;
    }
    active_optimizer_ = nullptr;
}

void LogisticRegression::do_fit(
    const base::FeatureInput& features,
    const base::TargetInput& targets
)
{
    validate_features(features);
    if (features.rows() != targets.size())
    {
        throw std::invalid_argument(
            "LogisticRegression features and targets have different sample counts"
        );
    }
    const Eigen::Index classes = class_count_from_targets(targets);
    const Eigen::MatrixXd design = add_intercept(features);
    const optim::Optimizer& optimizer = active_optimizer_ == nullptr
        ? static_cast<const optim::Optimizer&>(default_optimizer_)
        : *active_optimizer_;
    optim::ParameterVector parameters;
    std::vector<TrainingRecord> training_history;
    std::vector<double> loss_history;
    if (classes == 2)
    {
        BinaryProblem problem(design, targets, options_.l2_penalty);
        parameters = optimize(
            problem, design.cols(), optimizer, options_, training_history,
            loss_history
        );
    }
    else
    {
        if (dynamic_cast<const optim::NewtonOptimizer*>(&optimizer) != nullptr)
        {
            throw std::invalid_argument(
                "NewtonOptimizer is currently supported only for binary LogisticRegression"
            );
        }
        MulticlassProblem problem(
            design, one_hot(targets, classes), options_.l2_penalty
        );
        parameters = optimize(
            problem, design.cols() * classes, optimizer, options_,
            training_history, loss_history
        );
    }

    const Eigen::Index coefficient_columns = classes == 2 ? 1 : classes;
    coefficients_ = Eigen::Map<const Eigen::MatrixXd>(
        parameters.data(), design.cols(), coefficient_columns
    );
    feature_count_ = static_cast<std::size_t>(features.cols());
    class_count_ = static_cast<std::size_t>(classes);
    class_labels_.clear();
    class_labels_.reserve(class_count_);
    for (std::size_t index = 0; index < class_count_; ++index)
    {
        class_labels_.push_back(std::to_string(index));
    }
    loss_history_ = std::move(loss_history);
    training_history_ = std::move(training_history);
}

Eigen::MatrixXd LogisticRegression::predict_proba(
    const base::FeatureInput& features
) const
{
    if (!is_fitted()) throw base::NotFittedError("predict_proba()");
    if (features.cols() != static_cast<Eigen::Index>(feature_count_) ||
        !features.allFinite())
    {
        throw std::invalid_argument("LogisticRegression prediction features are invalid");
    }
    const Eigen::MatrixXd logits = add_intercept(features) * coefficients_;
    if (class_count_ == 2)
    {
        Eigen::MatrixXd result(features.rows(), 2);
        for (Eigen::Index row = 0; row < features.rows(); ++row)
        {
            const double positive = sigmoid(logits(row, 0));
            result(row, 0) = 1.0 - positive;
            result(row, 1) = positive;
        }
        return result;
    }
    return softmax(logits);
}

base::PredictionOutput LogisticRegression::do_predict(
    const base::FeatureInput& features
) const
{
    const Eigen::MatrixXd probabilities = predict_proba(features);
    base::PredictionOutput result(probabilities.rows());
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row)
    {
        Eigen::Index index = 0;
        probabilities.row(row).maxCoeff(&index);
        if (class_count_ == 2 &&
            probabilities(row, 1) < options_.decision_threshold)
        {
            index = 0;
        }
        else if (class_count_ == 2)
        {
            index = 1;
        }
        result(row) = static_cast<double>(index);
    }
    return result;
}

double LogisticRegression::do_score(
    const base::FeatureInput& features,
    const base::TargetInput& targets
) const
{
    if (targets.size() == 0 || features.rows() != targets.size())
    {
        throw std::invalid_argument(
            "LogisticRegression features and targets have different sample counts"
        );
    }
    const base::PredictionOutput predictions = do_predict(features);
    Eigen::Index correct = 0;
    for (Eigen::Index row = 0; row < targets.size(); ++row)
    {
        const double rounded = std::round(targets(row));
        if (!std::isfinite(targets(row)) || targets(row) < 0.0 ||
            std::abs(targets(row) - rounded) > 1e-12 ||
            rounded >= static_cast<double>(class_count_))
        {
            throw std::invalid_argument(
                "LogisticRegression score targets must be valid class indices"
            );
        }
        if (predictions(row) == targets(row)) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(targets.size());
}

std::size_t LogisticRegression::class_count() const noexcept
{
    return class_count_;
}

const std::vector<double>& LogisticRegression::loss_history() const noexcept
{
    return loss_history_;
}

const std::vector<TrainingRecord>& LogisticRegression::training_history() const noexcept
{
    return training_history_;
}

const std::vector<std::string>& LogisticRegression::class_labels() const noexcept
{
    return class_labels_;
}

void LogisticRegression::set_class_labels(
    const std::vector<std::string>& labels
)
{
    if (!is_fitted()) throw base::NotFittedError("set_class_labels()");
    if (labels.size() != class_count_)
    {
        throw std::invalid_argument(
            "LogisticRegression class label count does not match model"
        );
    }
    for (std::size_t index = 0; index < labels.size(); ++index)
    {
        if (labels[index].empty() ||
            std::find(labels.begin(), labels.begin() + index, labels[index]) !=
                labels.begin() + index)
        {
            throw std::invalid_argument(
                "LogisticRegression class labels must be non-empty and unique"
            );
        }
    }
    class_labels_ = labels;
}

double LogisticRegression::decision_threshold() const noexcept
{
    return options_.decision_threshold;
}

const char* LogisticRegression::type_name() const noexcept
{
    return "art::linear_model::LogisticRegression";
}

const LogisticRegressionOptions& LogisticRegression::options() const noexcept
{
    return options_;
}

void LogisticRegression::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}

void LogisticRegression::restore_state(
    const Eigen::MatrixXd& coefficients,
    std::size_t feature_count,
    std::size_t class_count,
    double decision_threshold,
    const LogisticRegressionOptions& options,
    const std::vector<std::string>& class_labels
)
{
    if (feature_count == 0 || class_count < 2 ||
        coefficients.rows() != static_cast<Eigen::Index>(feature_count + 1) ||
        coefficients.cols() != static_cast<Eigen::Index>(
            class_count == 2 ? 1 : class_count
        ) || !coefficients.allFinite() ||
        !std::isfinite(decision_threshold) || decision_threshold <= 0.0 ||
        decision_threshold >= 1.0 || options.max_iterations == 0 ||
        !std::isfinite(options.tolerance) || options.tolerance < 0.0 ||
        !std::isfinite(options.l2_penalty) || options.l2_penalty < 0.0 ||
        class_labels.size() != class_count)
    {
        throw std::invalid_argument(
            "Invalid LogisticRegression serialized state"
        );
    }
    for (const std::string& label : class_labels)
    {
        if (label.empty())
        {
            throw std::invalid_argument(
                "Serialized LogisticRegression class labels must not be empty"
            );
        }
    }

    mark_unfitted();
    options_ = options;
    options_.decision_threshold = decision_threshold;
    coefficients_ = coefficients;
    feature_count_ = feature_count;
    class_count_ = class_count;
    class_labels_ = class_labels;
    loss_history_.clear();
    training_history_.clear();
    active_optimizer_ = nullptr;
    mark_fitted();
}

const Eigen::MatrixXd& LogisticRegression::coefficients() const
{
    if (!is_fitted()) throw base::NotFittedError("coefficients()");
    return coefficients_;
}
}
