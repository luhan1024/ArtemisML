/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/base/types.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace art::model_selection
{
    struct TrainTestSplit
    {
        base::FeatureInput features_train;
        base::FeatureInput features_test;
        base::TargetInput targets_train;
        base::TargetInput targets_test;
    };

    // Split rows into disjoint train and test partitions. test_size is the
    // fraction assigned to the test partition; a supplied seed makes the
    // permutation reproducible.
    TrainTestSplit train_test_split(
        const base::FeatureInput& features,
        const base::TargetInput& targets,
        double test_size = 0.25,
        std::optional<std::uint64_t> random_seed = std::nullopt
    );
}
