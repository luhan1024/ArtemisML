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
