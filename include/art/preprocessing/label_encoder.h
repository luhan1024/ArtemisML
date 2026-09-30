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

#include <Eigen/Dense>

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

namespace art::preprocessing
{
    enum class LabelEncodingStrategy
    {
        Auto,
        Binary01,
        BinarySigned,
        SignedOrdinal,
        Ordinal,
        OneHot
    };

    enum class UnknownLabelPolicy
    {
        Error,
        Ignore
    };

    using EncodedTarget = std::variant<Eigen::VectorXd, Eigen::MatrixXd>;

    class LabelEncoder final
    {
    public:
        explicit LabelEncoder(
            LabelEncodingStrategy strategy = LabelEncodingStrategy::Auto,
            UnknownLabelPolicy unknown_policy = UnknownLabelPolicy::Error
        );

        void fit(const std::vector<std::string>& labels);
        EncodedTarget transform(const std::vector<std::string>& labels) const;
        EncodedTarget fit_transform(const std::vector<std::string>& labels);

        Eigen::VectorXd transform_vector(const std::vector<std::string>& labels) const;
        Eigen::MatrixXd transform_matrix(const std::vector<std::string>& labels) const;

        bool is_fitted() const noexcept;
        LabelEncodingStrategy strategy() const noexcept;
        LabelEncodingStrategy effective_strategy() const;
        UnknownLabelPolicy unknown_policy() const noexcept;
        const std::vector<std::string>& classes() const;
        std::vector<std::string> output_column_names(
            const std::string& prefix = "label"
        ) const;

    private:
        std::size_t class_index(const std::string& label) const;
        Eigen::VectorXd transform_ordinal_values(
            const std::vector<std::string>& labels
        ) const;

        LabelEncodingStrategy strategy_;
        UnknownLabelPolicy unknown_policy_;
        LabelEncodingStrategy effective_strategy_ = LabelEncodingStrategy::Auto;
        std::vector<std::string> classes_;
        bool fitted_ = false;
    };
}
