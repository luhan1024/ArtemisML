/*
 * ============================================================================
 *                         A R T E M I S M L
 *                         A R T E M I S M L
 * ============================================================================
 * Project: ArtemisML - C++ machine learning library
 * Main contributors: Han Lu & Yihan Wang
 * ============================================================================
 */
#pragma once

#include "art/base/predictor.h"

namespace art::base
{
    class Regressor : public Predictor
    {
    public:
        double score(
            const FeatureInput& features,
            const TargetInput& targets
        ) const;

    protected:
        virtual double do_score(
            const FeatureInput& features,
            const TargetInput& targets
        ) const = 0;
    };
}
