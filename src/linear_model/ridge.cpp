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
}
