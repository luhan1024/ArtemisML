/*
 * ============================================================================
 *                         A R T E M I S M L
 *                         A R T E M I S M L
 * ============================================================================
 * Project: ArtemisML - C++ machine learning library
 * Main contributors: Han Lu & Yihan Wang
 * ============================================================================
 */
#include "art/base/regressor.h"

namespace art::base
{
    double Regressor::score(
        const FeatureInput& features,
        const TargetInput& targets
    ) const
    {
        if (!is_fitted())
        {
            throw NotFittedError("score()");
        }
        return do_score(features, targets);
    }
}
