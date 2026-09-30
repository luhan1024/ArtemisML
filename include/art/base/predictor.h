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

#include "art/base/errors.h"
#include "art/base/estimator.h"

namespace art::base
{
    class Predictor : public Estimator
    {
    public:
        void fit(
            const FeatureInput& features,
            const TargetInput& targets = TargetInput{}
        ) final;

        bool is_fitted() const noexcept final;

        PredictionOutput predict(const FeatureInput& features) const;

    protected:
        virtual void do_fit(
            const FeatureInput& features,
            const TargetInput& targets
        ) = 0;

        virtual PredictionOutput do_predict(
            const FeatureInput& features
        ) const = 0;

        void mark_unfitted() noexcept;
        void mark_fitted() noexcept;

    private:
        bool fitted_ = false;
    };
}
