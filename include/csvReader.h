#pragma once

#include "dataframe.h"

#include <string>

namespace art::data
{
    class CsvReader
    {
    public:
        DataFrame read(const std::string& filename) const;
        void write(const std::string& filename, const DataFrame& data_frame) const;
    };
}
