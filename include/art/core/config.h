/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#pragma once

#include "art/core/errors.h"

namespace art::core
{
// Shared numerical policy only. Algorithm-specific options belong to their
// owning layer and must not be added here.
struct NumericConfig
{
    bool require_finite = true;

    void validate() const
    {
        // Reserved for future backend-wide checks. The current policy has no
        // invalid range, so validation is intentionally a no-op.
    }
};
}
