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
#pragma once

#include "art/data/dataset.h"
#include "art/base/regressor.h"
#include "art/linear_model/metrics.h"
#include "art/optim/optimizer.h"

#include <filesystem>
#include <string>

namespace art
{
    class Model;
}

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

    class LinearRegression final : public base::Regressor
    {
    public:
        using base::Regressor::fit;

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
        double mean_absolute_error(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& labels
        ) const;
        double r2_score(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& labels
        ) const;
        RegressionMetrics evaluate(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& labels
        ) const;

        void save(const std::filesystem::path& path) const;
        std::string type_name() const;
        std::size_t max_iterations() const noexcept;
        double tolerance() const noexcept;

        const ParameterVector& parameters() const;
        std::size_t feature_count() const;

    protected:
        void do_fit(const base::FeatureInput& features, const base::TargetInput& targets) override;
        base::PredictionOutput do_predict(const base::FeatureInput& features) const override;
        double do_score(const base::FeatureInput& features, const base::TargetInput& targets) const override;

    private:
        friend class ::art::Model;

        void restore_state(
            const ParameterVector& parameters,
            std::size_t feature_count,
            std::size_t max_iterations,
            double tolerance
        );

        ParameterVector parameters_;
        std::size_t feature_count_ = 0;
        std::size_t max_iterations_ = 0;
        double tolerance_ = 0.0;
        bool fitted_ = false;
    };
}
