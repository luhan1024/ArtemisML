/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#include "art/core/config.h"
#include "art/core/errors.h"
#include "art/core/types.h"

#include <cassert>

int main()
{
    const art::core::Shape shape{2, 3};
    assert(shape.rows == 2);
    assert(shape.columns == 3);
    assert(!shape.empty());
    assert(art::core::Shape{}.empty());

    art::core::NumericConfig{}.validate();

    bool caught = false;
    try
    {
        throw art::core::IndexError("out of range");
    }
    catch (const art::core::DataError&)
    {
        caught = true;
    }
    assert(caught);
    return 0;
}
