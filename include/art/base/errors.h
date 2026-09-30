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
#pragma once

#include <stdexcept>
#include <string>

namespace art::base
{
    class NotFittedError : public std::logic_error
    {
    public:
        explicit NotFittedError(const std::string& operation)
            : std::logic_error(
                  "art::base: cannot call " + operation
                  + " before fit() has completed")
        {
        }
    };
}
