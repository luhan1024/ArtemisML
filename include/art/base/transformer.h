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
    class Transformer : public Estimator
    {
    public:
        void fit(
            const FeatureInput& features,
            const TargetInput& targets = TargetInput{}
        ) final;

        bool is_fitted() const noexcept final;

        DataInput transform(const DataInput& input) const;

        DataInput fit_transform(const DataInput& input);

        DataInput fit_transform(
            const DataInput& input,
            const TargetInput& targets
        );

    protected:
        virtual void do_fit(
            const FeatureInput& features,
            const TargetInput& targets
        ) = 0;

        virtual DataInput do_transform(const DataInput& input) const = 0;

        void mark_unfitted() noexcept;

    private:
        bool fitted_ = false;
    };
}
