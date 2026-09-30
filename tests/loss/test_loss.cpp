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
#include "art/loss/loss.h"

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace
{
    constexpr double tolerance = 1e-10;

    void expectThrow(const std::function<void()>& operation)
    {
        bool thrown = false;
        try
        {
            operation();
        }
        catch (const std::invalid_argument&)
        {
            thrown = true;
        }
        assert(thrown);
    }

    Eigen::MatrixXd oneHotTargets()
    {
        Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2, 3);
        target(0, 1) = 1.0;
        target(1, 2) = 1.0;
        return target;
    }
}

int main()
{
    using art::loss::BinaryCrossEntropy;
    using art::loss::MeanAbsoluteError;
    using art::loss::MeanSquaredError;
    using art::loss::Reduction;
    using art::loss::SoftmaxCrossEntropy;
    using art::loss::SparseCrossEntropy;

    const Eigen::MatrixXd prediction =
        (Eigen::MatrixXd(2, 2) << 1.0, 3.0, 2.0, -1.0).finished();
    const Eigen::MatrixXd target =
        (Eigen::MatrixXd(2, 2) << 0.0, 1.0, 2.0, 1.0).finished();

    MeanSquaredError mse;
    assert(std::abs(mse.value(prediction, target) - 2.25) < tolerance);
    const Eigen::MatrixXd mse_gradient = mse.gradient(prediction, target);
    assert((mse_gradient -
            (Eigen::MatrixXd(2, 2) << 0.5, 1.0, 0.0, -1.0).finished()).norm() <
           tolerance);

    MeanSquaredError mse_sum(Reduction::Sum);
    assert(std::abs(mse_sum.value(prediction, target) - 9.0) < tolerance);
    assert((mse_sum.gradient(prediction, target) -
            (Eigen::MatrixXd(2, 2) << 2.0, 4.0, 0.0, -4.0).finished()).norm() <
           tolerance);

    MeanAbsoluteError mae;
    assert(std::abs(mae.value(prediction, target) - 1.25) < tolerance);
    assert((mae.gradient(prediction, target) -
            (Eigen::MatrixXd(2, 2) << 0.25, 0.25, 0.0, -0.25).finished()).norm() <
           tolerance);

    const Eigen::MatrixXd logits =
        (Eigen::MatrixXd(2, 2) << 1000.0, -1000.0, -2.0, 2.0).finished();
    const Eigen::MatrixXd binary_target =
        (Eigen::MatrixXd(2, 2) << 1.0, 0.0, 0.0, 1.0).finished();
    BinaryCrossEntropy bce;
    assert(std::isfinite(bce.value(logits, binary_target)));
    const Eigen::MatrixXd bce_gradient = bce.gradient(logits, binary_target);
    assert(std::abs(bce_gradient(0, 0)) < tolerance);
    assert(std::abs(bce_gradient(0, 1)) < tolerance);
    assert(std::abs(bce_gradient(1, 0) - 0.11920292 / 4.0) < 1e-8);
    assert(std::abs(bce_gradient(1, 1) + 0.11920292 / 4.0) < 1e-8);

    const Eigen::MatrixXd multiclass_logits =
        (Eigen::MatrixXd(2, 3) << 1000.0, 1001.0, 999.0,
                                  0.0, -1000.0, 1000.0).finished();
    const Eigen::MatrixXd multiclass_target = oneHotTargets();
    SoftmaxCrossEntropy softmax;
    assert(std::isfinite(softmax.value(
        multiclass_logits,
        multiclass_target
    )));
    const Eigen::MatrixXd softmax_gradient = softmax.gradient(
        multiclass_logits,
        multiclass_target
    );
    assert(softmax_gradient.rows() == 2 && softmax_gradient.cols() == 3);
    assert(std::abs(softmax_gradient.row(0).sum()) < tolerance);
    const double first_row_probability = 1.0 /
        (1.0 + std::exp(-1.0) + std::exp(-2.0));
    assert(std::abs(
        softmax_gradient(0, 1) - (first_row_probability / 2.0 - 0.5)
    ) < tolerance);

    const Eigen::VectorXi sparse_target =
        (Eigen::VectorXi(2) << 1, 2).finished();
    SparseCrossEntropy sparse;
    assert(std::abs(
        sparse.value(multiclass_logits, sparse_target) -
        softmax.value(multiclass_logits, multiclass_target)
    ) < tolerance);
    assert((sparse.gradient(multiclass_logits, sparse_target) -
            softmax_gradient).norm() < tolerance);

    Eigen::MatrixXd sparse_matrix_target(2, 1);
    sparse_matrix_target << 1.0, 2.0;
    assert(std::abs(
        sparse.value(multiclass_logits, sparse_matrix_target) -
        sparse.value(multiclass_logits, sparse_target)
    ) < tolerance);

    const double epsilon = 1e-6;
    Eigen::MatrixXd perturbed = prediction;
    for (Eigen::Index row = 0; row < prediction.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < prediction.cols(); ++column)
        {
            perturbed(row, column) += epsilon;
            const double plus = mse.value(perturbed, target);
            perturbed(row, column) -= 2.0 * epsilon;
            const double minus = mse.value(perturbed, target);
            perturbed(row, column) += epsilon;
            assert(std::abs(
                (plus - minus) / (2.0 * epsilon) - mse_gradient(row, column)
            ) < 1e-7);
        }
    }

    const Eigen::MatrixXd empty(0, 2);
    const Eigen::MatrixXd nonfinite =
        (Eigen::MatrixXd(1, 1) << std::numeric_limits<double>::infinity())
            .finished();
    expectThrow([&] { mse.value(empty, empty); });
    expectThrow([&] { mse.value(prediction, Eigen::MatrixXd(1, 1)); });
    expectThrow([&] { mse.value(nonfinite, nonfinite); });
    expectThrow([&] { mae.value(nonfinite, nonfinite); });
    expectThrow([&] {
        bce.value(nonfinite, Eigen::MatrixXd::Zero(1, 1));
    });
    expectThrow([&] { bce.value(logits, Eigen::MatrixXd::Constant(2, 2, 2.0)); });
    expectThrow([&] {
        softmax.value(nonfinite, Eigen::MatrixXd::Ones(1, 1));
    });

    Eigen::MatrixXd invalid_one_hot = multiclass_target;
    invalid_one_hot(0, 0) = 0.5;
    expectThrow([&] { softmax.value(multiclass_logits, invalid_one_hot); });
    Eigen::MatrixXd wrong_row_sum = multiclass_target;
    wrong_row_sum(0, 1) = 0.0;
    expectThrow([&] { softmax.value(multiclass_logits, wrong_row_sum); });
    expectThrow([&] {
        sparse.value(multiclass_logits, (Eigen::VectorXi(2) << 0, 3).finished());
    });
    expectThrow([&] {
        const Eigen::VectorXi invalid_logits_labels = Eigen::VectorXi::Zero(1);
        sparse.value(nonfinite, invalid_logits_labels);
    });
    Eigen::MatrixXd noninteger_label(2, 1);
    noninteger_label << 0.0, 1.5;
    expectThrow([&] { sparse.value(multiclass_logits, noninteger_label); });

    std::cout << "Loss test passed.\n";
    return 0;
}
