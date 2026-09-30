/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#include "art/base/predictor.h"

namespace art::base
{
    void Predictor::fit(
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

    bool Predictor::is_fitted() const noexcept
    {
        return fitted_;
    }

    PredictionOutput Predictor::predict(const FeatureInput& features) const
    {
        if (!fitted_)
        {
            throw NotFittedError("predict()");
        }
        return do_predict(features);
    }

    void Predictor::mark_unfitted() noexcept
    {
        fitted_ = false;
    }

    void Predictor::mark_fitted() noexcept
    {
        fitted_ = true;
    }
}
