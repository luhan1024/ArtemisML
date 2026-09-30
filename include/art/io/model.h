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

#include "art/linear_model/logistic_regression.h"

#include <Eigen/Dense>

#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace art
{
    class Model
    {
    public:
        std::string type_name() const;
        bool is_fitted() const noexcept;
        Eigen::VectorXd predict(const Eigen::MatrixXd& features) const;
        Eigen::MatrixXd predict_proba(const Eigen::MatrixXd& features) const;
        double evaluate(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& targets
        ) const;
        void save(const std::filesystem::path& path) const;

    private:
        friend void save(
            const linear_model::LogisticRegression& model,
            const std::string& path
        );
        friend void save(const Model& model, const std::string& path);
        friend Model load(const std::string& path);

        explicit Model(
            std::shared_ptr<linear_model::LogisticRegression> model
        );

        void restore_state(
            const Eigen::MatrixXd& coefficients,
            std::size_t feature_count,
            std::size_t class_count,
            double decision_threshold,
            const linear_model::LogisticRegressionOptions& options,
            const std::vector<std::string>& class_labels
        );
        std::shared_ptr<linear_model::LogisticRegression> model_;
    };

    void save(
        const linear_model::LogisticRegression& model,
        const std::string& path
    );
    void save(const Model& model, const std::string& path);
    Model load(const std::string& path);
}
