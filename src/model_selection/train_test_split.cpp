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
#include "art/model_selection/train_test_split.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

namespace art::model_selection
{
    TrainTestSplit train_test_split(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        const double test_size,
        const std::optional<std::uint64_t> random_seed
    )
    {
        if (features.rows() <= 0 || features.cols() <= 0 || targets.size() <= 0)
        {
            throw std::invalid_argument("train_test_split requires non-empty input");
        }
        if (features.rows() != targets.size())
        {
            throw std::invalid_argument(
                "train_test_split features and targets have different sample counts");
        }
        if (!std::isfinite(test_size) || test_size <= 0.0 || test_size >= 1.0)
        {
            throw std::invalid_argument("test_size must be finite and in (0, 1)");
        }

        const auto sample_count = static_cast<std::size_t>(features.rows());
        const auto test_count = std::max<std::size_t>(
            1, static_cast<std::size_t>(std::ceil(test_size * sample_count)));
        if (test_count >= sample_count)
        {
            throw std::invalid_argument(
                "test_size leaves no samples in the training partition");
        }

        std::vector<Eigen::Index> indices(features.rows());
        std::iota(indices.begin(), indices.end(), Eigen::Index{0});
        std::mt19937_64 generator(
            random_seed.value_or(std::random_device{}()));
        std::shuffle(indices.begin(), indices.end(), generator);

        const auto train_count = sample_count - test_count;
        TrainTestSplit result{
            base::FeatureInput(train_count, features.cols()),
            base::FeatureInput(test_count, features.cols()),
            base::TargetInput(train_count),
            base::TargetInput(test_count)};

        for (std::size_t row = 0; row < train_count; ++row)
        {
            result.features_train.row(static_cast<Eigen::Index>(row)) =
                features.row(indices[row]);
            result.targets_train(static_cast<Eigen::Index>(row)) =
                targets(indices[row]);
        }
        for (std::size_t row = 0; row < test_count; ++row)
        {
            result.features_test.row(static_cast<Eigen::Index>(row)) =
                features.row(indices[train_count + row]);
            result.targets_test(static_cast<Eigen::Index>(row)) =
                targets(indices[train_count + row]);
        }
        return result;
    }
}
