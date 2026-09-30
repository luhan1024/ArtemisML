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

#include "art/data/dataframe.h"

#include <string>

namespace art::io
{
    // CSV is an external representation of data::DataFrame. Other formats
    // (for example JSON) should add their own adapter in art::io without
    // adding format-specific behavior to data::DataFrame.
    class CsvReader
    {
    public:
        art::data::DataFrame read(const std::string& filename) const;
        void write(
            const std::string& filename,
            const art::data::DataFrame& data_frame
        ) const;
    };
}
