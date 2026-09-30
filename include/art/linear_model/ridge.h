/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#pragma once

#include "art/base/regressor.h"
#include "art/data/dataset.h"
#include "art/optim/optimizer.h"

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

    protected:
        void do_fit(const base::FeatureInput& features, const base::TargetInput& targets) override;
        base::PredictionOutput do_predict(const base::FeatureInput& features) const override;
        double do_score(const base::FeatureInput& features, const base::TargetInput& targets) const override;

    private:
        double alpha_;
        optim::ParameterVector parameters_;
        std::size_t feature_count_ = 0;
        optim::OptimizationResult last_result_;
    };
}
