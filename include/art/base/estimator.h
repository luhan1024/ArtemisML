#pragma once

#include "art/base/types.h"

namespace art::base
{
    class Estimator
    {
    public:
        virtual ~Estimator() = default;

        virtual void fit(
            const FeatureInput& features,
            const TargetInput& targets = TargetInput{}
        ) = 0;

        virtual bool is_fitted() const noexcept = 0;
    };
}
