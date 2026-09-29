#include "linear_regression.h"

#include <stdexcept>

namespace art::linear_model
{
namespace
{
    void validateDataset(const Dataset& dataset)
    {
        if (dataset.sample_count() == 0)
        {
            throw std::invalid_argument("Linear regression requires at least one sample");
        }

        if (dataset.feature_count() == 0)
        {
            throw std::invalid_argument("Linear regression requires at least one feature");
        }

        if (dataset.features.rows() != dataset.labels.size())
        {
            throw std::invalid_argument("Feature and label sample counts do not match");
        }
    }

    void validateParameters(
        const ParameterVector& parameters,
        std::size_t expected_count
    )
    {
        if (parameters.size() != static_cast<Eigen::Index>(expected_count))
        {
            throw std::invalid_argument("Linear regression parameter dimension is invalid");
        }
    }
}

LinearRegressionProblem::LinearRegressionProblem(const Dataset& dataset)
    : dataset_(dataset)
{
    validateDataset(dataset_);
}

double LinearRegressionProblem::value(const ParameterVector& parameters) const
{
    validateParameters(parameters, parameter_count());

    const Eigen::Index feature_count = dataset_.features.cols();
    const ParameterVector weights = parameters.head(feature_count);
    const double bias = parameters(feature_count);
    const Eigen::VectorXd residuals =
        dataset_.features * weights +
        Eigen::VectorXd::Constant(dataset_.sample_count(), bias) -
        dataset_.labels;

    return 0.5 * residuals.squaredNorm() /
        static_cast<double>(dataset_.sample_count());
}

ParameterVector LinearRegressionProblem::gradient(
    const ParameterVector& parameters
) const
{
    validateParameters(parameters, parameter_count());

    const Eigen::Index feature_count = dataset_.features.cols();
    const ParameterVector weights = parameters.head(feature_count);
    const double bias = parameters(feature_count);
    const Eigen::VectorXd residuals =
        dataset_.features * weights +
        Eigen::VectorXd::Constant(dataset_.sample_count(), bias) -
        dataset_.labels;

    ParameterVector result(parameter_count());
    result.head(feature_count) =
        dataset_.features.transpose() * residuals /
        static_cast<double>(dataset_.sample_count());
    result(feature_count) = residuals.sum() /
        static_cast<double>(dataset_.sample_count());

    return result;
}

HessianMatrix LinearRegressionProblem::hessian(
    const ParameterVector& parameters
) const
{
    validateParameters(parameters, parameter_count());

    const Eigen::Index feature_count = dataset_.features.cols();
    HessianMatrix result(parameter_count(), parameter_count());
    result.setZero();

    const Eigen::VectorXd ones =
        Eigen::VectorXd::Ones(dataset_.sample_count());
    const double sample_count = static_cast<double>(dataset_.sample_count());

    result.topLeftCorner(feature_count, feature_count) =
        dataset_.features.transpose() * dataset_.features / sample_count;
    result.topRightCorner(feature_count, 1) =
        dataset_.features.transpose() * ones / sample_count;
    result.bottomLeftCorner(1, feature_count) =
        ones.transpose() * dataset_.features / sample_count;
    result(feature_count, feature_count) = ones.squaredNorm() / sample_count;

    return result;
}

std::size_t LinearRegressionProblem::parameter_count() const
{
    return dataset_.feature_count() + 1;
}

OptimizationResult LinearRegression::fit(
    const Dataset& dataset,
    const Optimizer& optimizer,
    const OptimizerOptions& options
)
{
    validateDataset(dataset);

    LinearRegressionProblem problem(dataset);
    const ParameterVector initial_parameters =
        ParameterVector::Zero(static_cast<Eigen::Index>(problem.parameter_count()));
    const OptimizationResult result = optimizer.optimize(
        problem,
        initial_parameters,
        options
    );

    parameters_ = result.parameters;
    feature_count_ = dataset.feature_count();
    fitted_ = true;

    return result;
}

Eigen::VectorXd LinearRegression::predict(
    const Eigen::MatrixXd& features
) const
{
    if (!fitted_)
    {
        throw std::logic_error("Linear regression model has not been fitted");
    }

    if (features.cols() != static_cast<Eigen::Index>(feature_count_))
    {
        throw std::invalid_argument("Prediction feature dimension does not match model");
    }

    const Eigen::Index feature_count = static_cast<Eigen::Index>(feature_count_);
    return features * parameters_.head(feature_count) +
        Eigen::VectorXd::Constant(features.rows(), parameters_(feature_count));
}

double LinearRegression::mean_squared_error(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& labels
) const
{
    if (features.rows() != labels.size())
    {
        throw std::invalid_argument("Prediction features and labels have different sample counts");
    }

    const Eigen::VectorXd residuals = predict(features) - labels;
    return residuals.squaredNorm() / static_cast<double>(labels.size());
}

const ParameterVector& LinearRegression::parameters() const
{
    if (!fitted_)
    {
        throw std::logic_error("Linear regression model has not been fitted");
    }

    return parameters_;
}

std::size_t LinearRegression::feature_count() const
{
    return feature_count_;
}
}
