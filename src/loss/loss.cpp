/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#include "art/loss/loss.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace art::loss
{
namespace
{
    std::string prefix(const char* loss_name)
    {
        return std::string("art::loss::") + loss_name + ": ";
    }

    double reductionScale(Reduction reduction, std::size_t count)
    {
        if (count == 0)
        {
            throw std::invalid_argument("art::loss: reduction count must be positive");
        }
        return reduction == Reduction::Mean ? 1.0 / static_cast<double>(count) : 1.0;
    }

    void validateBinaryTargets(
        const Eigen::MatrixXd& target,
        const char* loss_name
    )
    {
        for (Eigen::Index row = 0; row < target.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < target.cols(); ++column)
            {
                const double value = target(row, column);
                if (value < 0.0 || value > 1.0)
                {
                    throw std::invalid_argument(
                        prefix(loss_name) + "target must be in [0, 1]"
                    );
                }
            }
        }
    }

    void validateOneHotTargets(
        const Eigen::MatrixXd& target,
        const char* loss_name
    )
    {
        constexpr double tolerance = 1e-12;
        for (Eigen::Index row = 0; row < target.rows(); ++row)
        {
            double row_sum = 0.0;
            for (Eigen::Index column = 0; column < target.cols(); ++column)
            {
                const double value = target(row, column);
                if (std::abs(value) > tolerance &&
                    std::abs(value - 1.0) > tolerance)
                {
                    throw std::invalid_argument(
                        prefix(loss_name) +
                        "target must contain only one-hot values"
                    );
                }
                row_sum += value;
            }
            if (std::abs(row_sum - 1.0) > tolerance)
            {
                throw std::invalid_argument(
                    prefix(loss_name) + "each target row must sum to one"
                );
            }
        }
    }

    double sigmoid(double value)
    {
        if (value >= 0.0)
        {
            return 1.0 / (1.0 + std::exp(-value));
        }
        const double exponent = std::exp(value);
        return exponent / (1.0 + exponent);
    }

    double logSumExp(const Eigen::MatrixXd& logits, Eigen::Index row)
    {
        const double maximum = logits.row(row).maxCoeff();
        double sum = 0.0;
        for (Eigen::Index column = 0; column < logits.cols(); ++column)
        {
            sum += std::exp(logits(row, column) - maximum);
        }
        return maximum + std::log(sum);
    }

    void validateSparseVector(
        const Eigen::MatrixXd& logits,
        const Eigen::VectorXi& target,
        const char* loss_name
    )
    {
        if (target.size() != logits.rows())
        {
            throw std::invalid_argument(
                prefix(loss_name) + "target length must equal sample count"
            );
        }
        for (Eigen::Index row = 0; row < target.size(); ++row)
        {
            if (target(row) < 0 || target(row) >= logits.cols())
            {
                throw std::invalid_argument(
                    prefix(loss_name) + "class index is out of range"
                );
            }
        }
    }

    Eigen::VectorXi sparseVectorFromMatrix(
        const Eigen::MatrixXd& target,
        const char* loss_name
    )
    {
        if (target.cols() != 1)
        {
            throw std::invalid_argument(
                prefix(loss_name) + "sparse target must have shape N x 1"
            );
        }

        Eigen::VectorXi result(target.rows());
        constexpr double tolerance = 1e-12;
        for (Eigen::Index row = 0; row < target.rows(); ++row)
        {
            const double value = target(row, 0);
            const double rounded = std::round(value);
            if (std::abs(value - rounded) > tolerance ||
                rounded < static_cast<double>(std::numeric_limits<int>::min()) ||
                rounded > static_cast<double>(std::numeric_limits<int>::max()))
            {
                throw std::invalid_argument(
                    prefix(loss_name) +
                    "sparse target must contain integer class indices"
                );
            }
            result(row) = static_cast<int>(rounded);
        }
        return result;
    }

    double sparseValue(
        const Eigen::MatrixXd& logits,
        const Eigen::VectorXi& target,
        Reduction reduction
    )
    {
        double raw = 0.0;
        for (Eigen::Index row = 0; row < logits.rows(); ++row)
        {
            raw += logSumExp(logits, row) - logits(row, target(row));
        }
        return raw * reductionScale(
            reduction,
            static_cast<std::size_t>(logits.rows())
        );
    }

    Eigen::MatrixXd sparseGradient(
        const Eigen::MatrixXd& logits,
        const Eigen::VectorXi& target,
        Reduction reduction
    )
    {
        Eigen::MatrixXd result(logits.rows(), logits.cols());
        for (Eigen::Index row = 0; row < logits.rows(); ++row)
        {
            const double log_sum_exp = logSumExp(logits, row);
            for (Eigen::Index column = 0; column < logits.cols(); ++column)
            {
                result(row, column) =
                    std::exp(logits(row, column) - log_sum_exp);
            }
            result(row, target(row)) -= 1.0;
        }
        return result * reductionScale(
            reduction,
            static_cast<std::size_t>(logits.rows())
        );
    }
}

LossFunction::LossFunction(Reduction reduction) noexcept
    : reduction_(reduction)
{
}

Reduction LossFunction::reduction() const noexcept
{
    return reduction_;
}

void LossFunction::validateSameShape(
    const Eigen::MatrixXd& prediction,
    const Eigen::MatrixXd& target,
    const char* loss_name
)
{
    if (prediction.rows() != target.rows() ||
        prediction.cols() != target.cols())
    {
        throw std::invalid_argument(
            prefix(loss_name) + "prediction and target shapes must match"
        );
    }
}

void LossFunction::validateNonEmpty(
    const Eigen::MatrixXd& prediction,
    const char* loss_name
)
{
    if (prediction.rows() == 0 || prediction.cols() == 0)
    {
        throw std::invalid_argument(
            prefix(loss_name) + "prediction must not be empty"
        );
    }
}

void LossFunction::validateFinite(
    const Eigen::MatrixXd& values,
    const char* name,
    const char* loss_name
)
{
    if (!values.allFinite())
    {
        throw std::invalid_argument(
            prefix(loss_name) + std::string(name) +
            " contains a non-finite value"
        );
    }
}

double LossFunction::scale(Reduction reduction, std::size_t count)
{
    return reductionScale(reduction, count);
}

double MeanSquaredError::value(
    const Eigen::MatrixXd& prediction,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(prediction, "MeanSquaredError");
    validateSameShape(prediction, target, "MeanSquaredError");
    validateFinite(prediction, "prediction", "MeanSquaredError");
    validateFinite(target, "target", "MeanSquaredError");
    const Eigen::MatrixXd difference = prediction - target;
    return difference.squaredNorm() * scale(
        reduction_,
        static_cast<std::size_t>(prediction.size())
    );
}

Eigen::MatrixXd MeanSquaredError::gradient(
    const Eigen::MatrixXd& prediction,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(prediction, "MeanSquaredError");
    validateSameShape(prediction, target, "MeanSquaredError");
    validateFinite(prediction, "prediction", "MeanSquaredError");
    validateFinite(target, "target", "MeanSquaredError");
    return 2.0 * (prediction - target) * scale(
        reduction_,
        static_cast<std::size_t>(prediction.size())
    );
}

double MeanAbsoluteError::value(
    const Eigen::MatrixXd& prediction,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(prediction, "MeanAbsoluteError");
    validateSameShape(prediction, target, "MeanAbsoluteError");
    validateFinite(prediction, "prediction", "MeanAbsoluteError");
    validateFinite(target, "target", "MeanAbsoluteError");
    return (prediction - target).array().abs().sum() * scale(
        reduction_,
        static_cast<std::size_t>(prediction.size())
    );
}

Eigen::MatrixXd MeanAbsoluteError::gradient(
    const Eigen::MatrixXd& prediction,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(prediction, "MeanAbsoluteError");
    validateSameShape(prediction, target, "MeanAbsoluteError");
    validateFinite(prediction, "prediction", "MeanAbsoluteError");
    validateFinite(target, "target", "MeanAbsoluteError");
    const Eigen::MatrixXd difference = prediction - target;
    Eigen::MatrixXd result = difference.unaryExpr([](double value) {
        return value > 0.0 ? 1.0 : (value < 0.0 ? -1.0 : 0.0);
    });
    return result * scale(
        reduction_,
        static_cast<std::size_t>(prediction.size())
    );
}

double BinaryCrossEntropy::value(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "BinaryCrossEntropy");
    validateSameShape(logits, target, "BinaryCrossEntropy");
    validateFinite(logits, "logits", "BinaryCrossEntropy");
    validateFinite(target, "target", "BinaryCrossEntropy");
    validateBinaryTargets(target, "BinaryCrossEntropy");

    double raw = 0.0;
    for (Eigen::Index row = 0; row < logits.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < logits.cols(); ++column)
        {
            const double logit = logits(row, column);
            raw += std::max(logit, 0.0) - logit * target(row, column) +
                std::log1p(std::exp(-std::abs(logit)));
        }
    }
    return raw * scale(
        reduction_,
        static_cast<std::size_t>(logits.size())
    );
}

Eigen::MatrixXd BinaryCrossEntropy::gradient(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "BinaryCrossEntropy");
    validateSameShape(logits, target, "BinaryCrossEntropy");
    validateFinite(logits, "logits", "BinaryCrossEntropy");
    validateFinite(target, "target", "BinaryCrossEntropy");
    validateBinaryTargets(target, "BinaryCrossEntropy");

    Eigen::MatrixXd result(logits.rows(), logits.cols());
    for (Eigen::Index row = 0; row < logits.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < logits.cols(); ++column)
        {
            result(row, column) = sigmoid(logits(row, column)) -
                target(row, column);
        }
    }
    return result * scale(
        reduction_,
        static_cast<std::size_t>(logits.size())
    );
}

double SoftmaxCrossEntropy::value(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "SoftmaxCrossEntropy");
    validateSameShape(logits, target, "SoftmaxCrossEntropy");
    validateFinite(logits, "logits", "SoftmaxCrossEntropy");
    validateFinite(target, "target", "SoftmaxCrossEntropy");
    validateOneHotTargets(target, "SoftmaxCrossEntropy");

    double raw = 0.0;
    for (Eigen::Index row = 0; row < logits.rows(); ++row)
    {
        const double log_sum_exp = logSumExp(logits, row);
        for (Eigen::Index column = 0; column < logits.cols(); ++column)
        {
            raw += target(row, column) *
                (log_sum_exp - logits(row, column));
        }
    }
    return raw * scale(
        reduction_,
        static_cast<std::size_t>(logits.rows())
    );
}

Eigen::MatrixXd SoftmaxCrossEntropy::gradient(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "SoftmaxCrossEntropy");
    validateSameShape(logits, target, "SoftmaxCrossEntropy");
    validateFinite(logits, "logits", "SoftmaxCrossEntropy");
    validateFinite(target, "target", "SoftmaxCrossEntropy");
    validateOneHotTargets(target, "SoftmaxCrossEntropy");

    Eigen::MatrixXd result(logits.rows(), logits.cols());
    for (Eigen::Index row = 0; row < logits.rows(); ++row)
    {
        const double log_sum_exp = logSumExp(logits, row);
        for (Eigen::Index column = 0; column < logits.cols(); ++column)
        {
            result(row, column) =
                std::exp(logits(row, column) - log_sum_exp) -
                target(row, column);
        }
    }
    return result * scale(
        reduction_,
        static_cast<std::size_t>(logits.rows())
    );
}

double SparseCrossEntropy::value(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "SparseCrossEntropy");
    validateFinite(logits, "logits", "SparseCrossEntropy");
    validateFinite(target, "target", "SparseCrossEntropy");
    const Eigen::VectorXi labels = sparseVectorFromMatrix(
        target,
        "SparseCrossEntropy"
    );
    validateSparseVector(logits, labels, "SparseCrossEntropy");
    return sparseValue(logits, labels, reduction_);
}

Eigen::MatrixXd SparseCrossEntropy::gradient(
    const Eigen::MatrixXd& logits,
    const Eigen::MatrixXd& target
) const
{
    validateNonEmpty(logits, "SparseCrossEntropy");
    validateFinite(logits, "logits", "SparseCrossEntropy");
    validateFinite(target, "target", "SparseCrossEntropy");
    const Eigen::VectorXi labels = sparseVectorFromMatrix(
        target,
        "SparseCrossEntropy"
    );
    validateSparseVector(logits, labels, "SparseCrossEntropy");
    return sparseGradient(logits, labels, reduction_);
}

double SparseCrossEntropy::value(
    const Eigen::MatrixXd& logits,
    const Eigen::VectorXi& target
) const
{
    validateNonEmpty(logits, "SparseCrossEntropy");
    validateFinite(logits, "logits", "SparseCrossEntropy");
    validateSparseVector(logits, target, "SparseCrossEntropy");
    return sparseValue(logits, target, reduction_);
}

Eigen::MatrixXd SparseCrossEntropy::gradient(
    const Eigen::MatrixXd& logits,
    const Eigen::VectorXi& target
) const
{
    validateNonEmpty(logits, "SparseCrossEntropy");
    validateFinite(logits, "logits", "SparseCrossEntropy");
    validateSparseVector(logits, target, "SparseCrossEntropy");
    return sparseGradient(logits, target, reduction_);
}
}
