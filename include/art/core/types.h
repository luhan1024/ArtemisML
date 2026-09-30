/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#pragma once

#include <cstddef>

namespace art::core
{
using Size = std::size_t;
using RowCount = std::size_t;
using ColumnCount = std::size_t;
using Position = std::size_t;

struct Shape
{
    RowCount rows = 0;
    ColumnCount columns = 0;

    bool empty() const noexcept
    {
        return rows == 0 || columns == 0;
    }
};
}
