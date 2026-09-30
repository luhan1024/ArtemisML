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
#include "art/core/autodiff.h"

#include <cassert>
#include <cmath>
#include <iostream>

int main()
{
    art::core::autodiff::GradientTape tape;
    const auto x = tape.variable(2.0);
    const auto y = tape.variable(3.0);
    const auto unused = tape.variable(7.0);

    const auto output = x * x + x * y + art::core::autodiff::exp(y);
    const auto gradient = tape.gradient(output, {x, y, unused});

    assert(gradient.size() == 3);
    assert(std::abs(gradient(0) - 7.0) < 1e-12);
    assert(std::abs(gradient(1) - (2.0 + std::exp(3.0))) < 1e-12);
    assert(std::abs(gradient(2)) < 1e-12);

    const auto logarithm = art::core::autodiff::log(x);
    const auto logarithm_gradient = tape.gradient(logarithm, {x});
    assert(std::abs(logarithm_gradient(0) - 0.5) < 1e-12);

    std::cout << "Autodiff test passed.\n";
    return 0;
}
