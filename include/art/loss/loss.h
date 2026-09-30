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

namespace art::loss
{
enum class Reduction
{
    Mean,
    Sum
};

class LossFunction
{
public:
    explicit LossFunction(Reduction reduction = Reduction::Mean) noexcept;
    virtual ~LossFunction() = default;

    Reduction reduction() const noexcept;

    // The returned gradient is with respect to prediction/logits. Targets are
    // constants and no target gradient is exposed by this interface.
    virtual double value(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const = 0;

    virtual Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const = 0;

protected:
    static void validateSameShape(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target,
        const char* loss_name
    );
    static void validateNonEmpty(
        const Eigen::MatrixXd& prediction,
        const char* loss_name
    );
    static void validateFinite(
        const Eigen::MatrixXd& values,
        const char* name,
        const char* loss_name
    );
    static double scale(Reduction reduction, std::size_t count);

    Reduction reduction_;
};

class MeanSquaredError final : public LossFunction
{
public:
    using LossFunction::LossFunction;

    double value(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const override;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const override;
};

class MeanAbsoluteError final : public LossFunction
{
public:
    using LossFunction::LossFunction;

    double value(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const override;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& prediction,
        const Eigen::MatrixXd& target
    ) const override;
};

// BinaryCrossEntropy consumes logits, not probabilities. Targets must be in
// [0, 1], and the gradient is sigmoid(logit) - target.
class BinaryCrossEntropy final : public LossFunction
{
public:
    using LossFunction::LossFunction;

    double value(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;
};

// SoftmaxCrossEntropy consumes an N x C logits matrix and an N x C one-hot
// target matrix. Its unscaled gradient is softmax(logits) - target.
class SoftmaxCrossEntropy final : public LossFunction
{
public:
    using LossFunction::LossFunction;

    double value(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;
};

// SparseCrossEntropy consumes an N x C logits matrix and integer class labels.
// The MatrixXd overload expects an N x 1 column containing integral values;
// VectorXi is the preferred representation.
class SparseCrossEntropy final : public LossFunction
{
public:
    using LossFunction::LossFunction;

    double value(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& logits,
        const Eigen::MatrixXd& target
    ) const override;

    double value(
        const Eigen::MatrixXd& logits,
        const Eigen::VectorXi& target
    ) const;

    Eigen::MatrixXd gradient(
        const Eigen::MatrixXd& logits,
        const Eigen::VectorXi& target
    ) const;
};
}
