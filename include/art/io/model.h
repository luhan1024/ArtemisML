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

#include "art/linear_model/linear_regression.h"
#include "art/linear_model/logistic_regression.h"
#include "art/linear_model/ridge.h"
#include "art/linear_model/lasso.h"
#include "art/linear_model/elastic_net.h"

#include <Eigen/Dense>

#include <filesystem>
#include <memory>
#include <string>
#include <variant>
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
        Eigen::MatrixXd decision_function(const Eigen::MatrixXd& features) const;
        linear_model::EvaluationResult evaluate(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& targets
        ) const;
        linear_model::RegressionMetrics regression_metrics(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& targets
        ) const;
        linear_model::ClassificationMetrics classification_metrics(
            const Eigen::MatrixXd& features,
            const Eigen::VectorXd& targets
        ) const;
        void save(const std::filesystem::path& path) const;

    private:
        friend void save(
            const linear_model::LogisticRegression& model,
            const std::string& path
        );
        friend void save(
            const linear_model::LinearRegression& model,
            const std::string& path
        );
        friend void save(
            const linear_model::Ridge& model,
            const std::string& path
        );
        friend void save(
            const linear_model::RidgeClassifier& model,
            const std::string& path
        );
        friend void save(const linear_model::Lasso& model, const std::string& path);
        friend void save(const linear_model::ElasticNet& model, const std::string& path);
        friend void save(const Model& model, const std::string& path);
        friend Model load(const std::string& path);

        using Storage = std::variant<
            std::shared_ptr<linear_model::LogisticRegression>,
            std::shared_ptr<linear_model::LinearRegression>,
            std::shared_ptr<linear_model::Ridge>,
            std::shared_ptr<linear_model::RidgeClassifier>,
            std::shared_ptr<linear_model::Lasso>,
            std::shared_ptr<linear_model::ElasticNet>
        >;

        explicit Model(Storage model);

        void restore_state(
            const Eigen::MatrixXd& coefficients,
            std::size_t feature_count,
            std::size_t class_count,
            double decision_threshold,
            const linear_model::LogisticRegressionOptions& options,
            const std::vector<std::string>& class_labels
        );
        void restore_linear_state(
            const Eigen::VectorXd& parameters,
            std::size_t feature_count,
            std::size_t max_iterations,
            double tolerance
        );
        void restore_ridge_state(
            const Eigen::VectorXd& parameters,
            std::size_t feature_count,
            double alpha,
            std::size_t iterations,
            bool converged
        );
        void restore_ridge_classifier_state(
            const Eigen::MatrixXd& coefficients,
            std::size_t feature_count,
            std::size_t class_count,
            double alpha,
            const std::vector<std::string>& labels
        );
        void restore_lasso_state(
            const Eigen::VectorXd& parameters,
            std::size_t feature_count,
            double alpha
        );
        void restore_elastic_net_state(
            const Eigen::VectorXd& parameters,
            std::size_t feature_count,
            double alpha,
            double l1_ratio
        );
        Storage model_;
    };

    void save(
        const linear_model::LogisticRegression& model,
        const std::string& path
    );
    void save(
        const linear_model::LinearRegression& model,
        const std::string& path
    );
    void save(const linear_model::Ridge& model, const std::string& path);
    void save(
        const linear_model::RidgeClassifier& model,
        const std::string& path
    );
    void save(const linear_model::Lasso& model, const std::string& path);
    void save(const linear_model::ElasticNet& model, const std::string& path);
    void save(const Model& model, const std::string& path);
    Model load(const std::string& path);

    inline void save(
        const linear_model::LogisticRegression& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const linear_model::LinearRegression& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const linear_model::Ridge& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const linear_model::RidgeClassifier& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const linear_model::Lasso& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const linear_model::ElasticNet& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(
        const Model& model,
        const std::filesystem::path& path
    ) { save(model, path.string()); }

    inline void save(const linear_model::LogisticRegression& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const linear_model::LinearRegression& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const linear_model::Ridge& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const linear_model::RidgeClassifier& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const linear_model::Lasso& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const linear_model::ElasticNet& model, const char* path)
    { save(model, std::string(path)); }

    inline void save(const Model& model, const char* path)
    { save(model, std::string(path)); }

    inline Model load(const char* path)
    { return load(std::string(path)); }

    inline Model load(const std::filesystem::path& path)
    { return load(path.string()); }
}
