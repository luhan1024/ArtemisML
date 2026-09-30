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

#include <string>
#include <string_view>

namespace art::data
{
enum class MissingKind
{
    none,
    empty,
    null_literal,
    nan_literal
};

struct MissingPolicy
{
    bool empty_is_missing = false;
    bool null_is_missing = true;
    bool nan_is_missing = true;
};

inline MissingKind missing_kind(
    std::string_view value,
    const MissingPolicy& policy = {}
)
{
    if (value.empty() && policy.empty_is_missing)
    {
        return MissingKind::empty;
    }
    if ((value == "null" || value == "NULL") && policy.null_is_missing)
    {
        return MissingKind::null_literal;
    }
    if ((value == "nan" || value == "NaN" || value == "NAN") && policy.nan_is_missing)
    {
        return MissingKind::nan_literal;
    }
    return MissingKind::none;
}

inline bool is_missing(
    std::string_view value,
    const MissingPolicy& policy = {}
)
{
    return missing_kind(value, policy) != MissingKind::none;
}
}
