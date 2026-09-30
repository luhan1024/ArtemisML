/*
 * ============================================================================
 *                         A R T E M I S M L
 *                         A R T E M I S M L
 * ============================================================================
 * Project: ArtemisML - C++ machine learning library
 * Main contributors: Han Lu & Yihan Wang
 * ============================================================================
 */
#pragma once

#include <string>
#include <vector>

namespace art::data
{
    struct DataFrame
    {
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> rows;
    };
}
