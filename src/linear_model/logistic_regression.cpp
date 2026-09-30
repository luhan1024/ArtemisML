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
        BinaryProblem(Eigen::MatrixXd features, Eigen::MatrixXd targets)
            : features_(std::move(features)), targets_(std::move(targets)) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), 1
            );
            return loss_.value(features_ * weights, targets_);
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), 1
            );
            const Eigen::MatrixXd logits = features_ * weights;
            return features_.transpose() * loss_.gradient(logits, targets_);
        }

    private:
        Eigen::MatrixXd features_;
        Eigen::MatrixXd targets_;
        loss::BinaryCrossEntropy loss_;
    };

    class MulticlassProblem final : public optim::OptimizationProblem
    {
    public:
        MulticlassProblem(Eigen::MatrixXd features, Eigen::MatrixXd targets)
            : features_(std::move(features)), targets_(std::move(targets)) {}

        double value(const optim::ParameterVector& parameters) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), targets_.cols()
            );
            return loss_.value(features_ * weights, targets_);
        }

        optim::ParameterVector gradient(
            const optim::ParameterVector& parameters
        ) const override
        {
            const Eigen::Map<const Eigen::MatrixXd> weights(
                parameters.data(), features_.cols(), targets_.cols()
            );
            const Eigen::MatrixXd logits = features_ * weights;
            const Eigen::MatrixXd matrix_gradient = features_.transpose() *
                loss_.gradient(logits, targets_);
            return Eigen::Map<const optim::ParameterVector>(
                matrix_gradient.data(), matrix_gradient.size()
            );
        }

    private:
        Eigen::MatrixXd features_;
        Eigen::MatrixXd targets_;
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
        double learning_rate,
        std::size_t max_iterations,
        double tolerance
    )
    {
        if (!std::isfinite(learning_rate) || learning_rate <= 0.0 ||
            !std::isfinite(tolerance) || tolerance < 0.0 || max_iterations == 0)
        {
            throw std::invalid_argument(
                "LogisticRegression optimization options are invalid"
            );
        }
        optim::GradientDescent optimizer;
        optim::OptimizerOptions options;
        options.learning_rate = learning_rate;
        options.max_iterations = max_iterations;
        options.tolerance = tolerance;
        optim::ParameterVector parameters =
            optim::ParameterVector::Zero(parameter_count);
        optim::OptimizerState state;
        for (std::size_t iteration = 0; iteration < max_iterations; ++iteration)
        {
            const optim::OptimizationStepResult step = optimizer.step(
                problem, parameters, state, options
            );
            parameters = step.parameters;
            if (step.step_norm <= tolerance) break;
        }
        return parameters;
    }
}

LogisticRegression::LogisticRegression(
    double learning_rate, std::size_t max_iterations, double tolerance
)
    : learning_rate_(learning_rate),
      max_iterations_(max_iterations),
      tolerance_(tolerance) {}

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
    optim::ParameterVector parameters;
    if (classes == 2)
    {
        BinaryProblem problem(design, targets);
        parameters = optimize(
            problem, design.cols(), learning_rate_, max_iterations_, tolerance_
        );
    }
    else
    {
        MulticlassProblem problem(design, one_hot(targets, classes));
        parameters = optimize(
            problem, design.cols() * classes,
            learning_rate_, max_iterations_, tolerance_
        );
    }

    const Eigen::Index coefficient_columns = classes == 2 ? 1 : classes;
    coefficients_ = Eigen::Map<const Eigen::MatrixXd>(
        parameters.data(), design.cols(), coefficient_columns
    );
    feature_count_ = static_cast<std::size_t>(features.cols());
    class_count_ = static_cast<std::size_t>(classes);
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

const Eigen::MatrixXd& LogisticRegression::coefficients() const
{
    if (!is_fitted()) throw base::NotFittedError("coefficients()");
    return coefficients_;
}
}
