/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#include "art/base/transformer.h"

namespace art::base
{
    void Transformer::fit(
        const FeatureInput& features,
        const TargetInput& targets
    )
    {
        // A failed refit invalidates the estimator. This makes the public
        // fitted state deterministic even when do_fit mutates before throwing.
        fitted_ = false;
        do_fit(features, targets);
        fitted_ = true;
    }

    bool Transformer::is_fitted() const noexcept
    {
        return fitted_;
    }

    DataInput Transformer::transform(const DataInput& input) const
    {
        if (!fitted_)
        {
            throw NotFittedError("transform()");
        }
        return do_transform(input);
    }

    DataInput Transformer::fit_transform(const DataInput& input)
    {
        fit(input);
        return transform(input);
    }

    DataInput Transformer::fit_transform(
        const DataInput& input,
        const TargetInput& targets
    )
    {
        fit(input, targets);
        return transform(input);
    }

    void Transformer::mark_unfitted() noexcept
    {
        fitted_ = false;
    }
}
