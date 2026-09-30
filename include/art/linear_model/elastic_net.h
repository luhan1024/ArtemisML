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
#pragma once

#include "art/base/regressor.h"
#include "art/linear_model/linear_regression.h"
#include "art/optim/optimizer.h"

#include <filesystem>

namespace art
{
    class Model;
}

namespace art::linear_model
{
    class ElasticNet final : public base::Regressor
    {
    public:
        using base::Regressor::fit;

        explicit ElasticNet(
            double alpha = 1.0,
            double l1_ratio = 0.5
        );

        void fit(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const optim::Optimizer& optimizer
        );

        double alpha() const noexcept;
        double l1_ratio() const noexcept;
        std::size_t feature_count() const noexcept;
        const optim::ParameterVector& parameters() const;
        RegressionMetrics evaluate(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& targets
        ) const;
        void save(const std::filesystem::path& path) const;
        const char* type_name() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::PredictionOutput do_predict(const base::FeatureInput&) const override;
        double do_score(const base::FeatureInput&, const base::TargetInput&) const override;

    private:
        friend class ::art::Model;

        void restore_state(
            const optim::ParameterVector& parameters,
            std::size_t feature_count,
            double alpha,
            double l1_ratio
        );

        double alpha_;
        double l1_ratio_;
        optim::ParameterVector parameters_;
        std::size_t feature_count_ = 0;
        const optim::Optimizer* active_optimizer_ = nullptr;
    };
}
