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
#include "art/optim/optimizer.h"

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <vector>

namespace art
{
    class Model;
}

namespace art::linear_model
{
    struct LogisticRegressionOptions
    {
        std::size_t max_iterations = 10000;
        double tolerance = 1e-7;
        double l2_penalty = 0.0;
        double decision_threshold = 0.5;
    };

    struct TrainingRecord
    {
        double objective = 0.0;
        double gradient_norm = 0.0;
        double step_norm = 0.0;
    };

    class LogisticRegression final : public base::Classifier
    {
    public:
        using base::Classifier::fit;

        LogisticRegression() = default;

        LogisticRegression(
            double learning_rate,
            std::size_t max_iterations,
            double tolerance
        );

        explicit LogisticRegression(
            const LogisticRegressionOptions& options
        );

        void fit(
            const base::FeatureInput& features,
            const base::TargetInput& targets,
            const optim::Optimizer& optimizer
        );

        Eigen::MatrixXd predict_proba(
            const base::FeatureInput& features
        ) const;

        std::size_t class_count() const noexcept;
        const std::vector<double>& loss_history() const noexcept;
        const std::vector<TrainingRecord>& training_history() const noexcept;
        const std::vector<std::string>& class_labels() const noexcept;
        void set_class_labels(const std::vector<std::string>& labels);
        double decision_threshold() const noexcept;
        const char* type_name() const noexcept;
        const LogisticRegressionOptions& options() const noexcept;
        // Rows include the intercept; binary models use one logit column,
        // while multiclass models use one column per class.
        const Eigen::MatrixXd& coefficients() const;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::PredictionOutput do_predict(const base::FeatureInput&) const override;
        double do_score(const base::FeatureInput&, const base::TargetInput&) const override;

    private:
        friend class ::art::Model;

        void restore_state(
            const Eigen::MatrixXd& coefficients,
            std::size_t feature_count,
            std::size_t class_count,
            double decision_threshold,
            const LogisticRegressionOptions& options,
            const std::vector<std::string>& class_labels
        );

        LogisticRegressionOptions options_;
        optim::GradientDescent default_optimizer_;
        const optim::Optimizer* active_optimizer_ = nullptr;
        Eigen::MatrixXd coefficients_;
        std::size_t feature_count_ = 0;
        std::size_t class_count_ = 0;
        std::vector<double> loss_history_;
        std::vector<TrainingRecord> training_history_;
        std::vector<std::string> class_labels_;
    };
}
