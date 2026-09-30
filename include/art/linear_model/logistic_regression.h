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

#include <Eigen/Dense>

#include <cstddef>

namespace art::linear_model
{
    class LogisticRegression final : public base::Classifier
    {
    public:
        LogisticRegression() = default;

        LogisticRegression(
            double learning_rate,
            std::size_t max_iterations,
            double tolerance
        );

        Eigen::MatrixXd predict_proba(
            const base::FeatureInput& features
        ) const;

        std::size_t class_count() const noexcept;
        // Rows include the intercept; binary models use one logit column,
        // while multiclass models use one column per class.
        const Eigen::MatrixXd& coefficients() const;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::PredictionOutput do_predict(const base::FeatureInput&) const override;
        double do_score(const base::FeatureInput&, const base::TargetInput&) const override;

    private:
        double learning_rate_ = 0.05;
        std::size_t max_iterations_ = 10000;
        double tolerance_ = 1e-7;
        Eigen::MatrixXd coefficients_;
        std::size_t feature_count_ = 0;
        std::size_t class_count_ = 0;
    };
}
