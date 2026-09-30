/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/core/errors.h"
#include "art/data/missing.h"

#include <cerrno>
#include <cstdlib>
#include <string>
#include <vector>

namespace art::data
{
enum class DTypeKind
{
    string,
    integer,
    floating,
    boolean,
    unknown
};

class DType
{
public:
    explicit DType(DTypeKind kind = DTypeKind::unknown) : kind_(kind) {}

    DTypeKind kind() const noexcept { return kind_; }
    bool is_numeric() const noexcept
    {
        return kind_ == DTypeKind::integer || kind_ == DTypeKind::floating;
    }

    bool operator==(const DType& other) const noexcept { return kind_ == other.kind_; }
    bool operator!=(const DType& other) const noexcept { return !(*this == other); }

private:
    DTypeKind kind_;
};

inline DType infer_dtype(
    const std::vector<std::string>& values,
    const MissingPolicy& policy = {}
)
{
    bool has_value = false;
    bool all_integer = true;
    bool all_numeric = true;
    bool all_boolean = true;

    for (const std::string& value : values)
    {
        if (is_missing(value, policy))
        {
            continue;
        }
        has_value = true;
        if (value != "true" && value != "false")
        {
            all_boolean = false;
        }

        char* end = nullptr;
        errno = 0;
        std::strtod(value.c_str(), &end);
        const bool numeric = errno == 0 && end != value.c_str() && *end == '\0';
        if (!numeric)
        {
            all_numeric = false;
            all_integer = false;
            continue;
        }
        if (value.find_first_of(".eE") != std::string::npos)
        {
            all_integer = false;
        }
    }

    if (!has_value) return DType(DTypeKind::unknown);
    if (all_boolean) return DType(DTypeKind::boolean);
    if (all_numeric && all_integer) return DType(DTypeKind::integer);
    if (all_numeric) return DType(DTypeKind::floating);
    return DType(DTypeKind::string);
}
}
