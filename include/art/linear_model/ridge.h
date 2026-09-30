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

#include "art/base/classifier.h"
#include "art/base/regressor.h"
#include "art/data/dataset.h"
#include "art/optim/optimizer.h"

#include <filesystem>
#include <string>
#include <vector>

namespace art
{
    class Model;
}

namespace art::linear_model
{
    class RidgeProblem final : public optim::OptimizationProblem
    {
    public:
        RidgeProblem(const data::Dataset& dataset, double alpha);

        double value(const optim::ParameterVector& parameters) const override;
        optim::ParameterVector gradient(const optim::ParameterVector& parameters) const override;
        optim::HessianMatrix hessian(const optim::ParameterVector& parameters) const override;
        std::size_t parameter_count() const;

    private:
        const data::Dataset& dataset_;
        double alpha_;
    };

    class Ridge final : public base::Regressor
    {
    public:
        using base::Regressor::fit;

        explicit Ridge(double alpha = 1.0);

        optim::OptimizationResult fit(
            const data::Dataset& dataset,
            const optim::Optimizer& optimizer,
            const optim::OptimizerOptions& options
        );

        double alpha() const noexcept;
        const optim::ParameterVector& parameters() const;
        std::size_t feature_count() const noexcept;
        void save(const std::filesystem::path& path) const;
        const char* type_name() const noexcept;
        std::size_t iterations() const noexcept;
        bool converged() const noexcept;
        double mean_squared_error(
            const Eigen::MatrixXd&, const Eigen::VectorXd& labels
        ) const;
        double mean_absolute_error(
            const Eigen::MatrixXd&, const Eigen::VectorXd& labels
        ) const;
        double r2_score(
            const Eigen::MatrixXd&, const Eigen::VectorXd& labels
        ) const;

    protected:
        void do_fit(const base::FeatureInput& features, const base::TargetInput& targets) override;
        base::PredictionOutput do_predict(const base::FeatureInput& features) const override;
        double do_score(const base::FeatureInput& features, const base::TargetInput& targets) const override;

    private:
        friend class ::art::Model;

        void restore_state(
            const optim::ParameterVector& parameters,
            std::size_t feature_count,
            double alpha,
            std::size_t iterations,
            bool converged
        );

        double alpha_;
        optim::ParameterVector parameters_;
        std::size_t feature_count_ = 0;
        optim::OptimizationResult last_result_;
    };

    class RidgeClassifier final : public base::Classifier
    {
    public:
        using base::Classifier::fit;

        explicit RidgeClassifier(double alpha = 1.0);

        void fit(
            const base::FeatureInput& features,
            const Eigen::MatrixXd& one_hot_targets
        );

        Eigen::MatrixXd predict_proba(
            const base::FeatureInput& features
        ) const;
        double alpha() const noexcept;
        std::size_t class_count() const noexcept;
        std::size_t feature_count() const noexcept;
        const Eigen::MatrixXd& coefficients() const;
        const std::vector<std::string>& class_labels() const noexcept;
        void set_class_labels(const std::vector<std::string>& labels);
        void save(const std::filesystem::path& path) const;
        const char* type_name() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::PredictionOutput do_predict(const base::FeatureInput&) const override;
        double do_score(const base::FeatureInput&, const base::TargetInput&) const override;

    private:
        friend class ::art::Model;

        void fit_encoded(
            const base::FeatureInput& features,
            const Eigen::MatrixXd& targets
        );
        void restore_state(
            const Eigen::MatrixXd& coefficients,
            std::size_t feature_count,
            std::size_t class_count,
            double alpha,
            const std::vector<std::string>& labels
        );

        double alpha_;
        Eigen::MatrixXd coefficients_;
        std::size_t feature_count_ = 0;
        std::size_t class_count_ = 0;
        std::vector<std::string> class_labels_;
    };
}
