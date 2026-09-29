#pragma once

#include "dataset.h"
#include "optimizer.h"

namespace art::linear_model
{
    using data::Dataset;
    using optim::HessianMatrix;
    using optim::OptimizationProblem;
    using optim::OptimizationResult;
    using optim::Optimizer;
    using optim::OptimizerOptions;
    using optim::ParameterVector;

    class LinearRegressionProblem final : public OptimizationProblem
    {
    public:
        explicit LinearRegressionProblem(const Dataset& dataset);

        double value(const ParameterVector& parameters) const override;
        ParameterVector gradient(const ParameterVector& parameters) const override;
        HessianMatrix hessian(const ParameterVector& parameters) const override;

        std::size_t parameter_count() const;

    private:
        const Dataset& dataset_;
    };

    class LinearRegression
    {
    public:
        OptimizationResult fit(
            const Dataset& dataset,
            const Optimizer& optimizer,
            const OptimizerOptions& options
        );

        Eigen::VectorXd predict(const Eigen::MatrixXd& features) const;
        double mean_squared_error(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& labels
        ) const;

        const ParameterVector& parameters() const;
        std::size_t feature_count() const;

    private:
        ParameterVector parameters_;
        std::size_t feature_count_ = 0;
        bool fitted_ = false;
    };
}
