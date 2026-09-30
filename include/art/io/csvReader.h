/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#pragma once

#include "art/data/dataframe.h"
#include "art/data/text_label_dataset.h"

#include <string>
#include <vector>

namespace art::io
{
    // CSV is an external representation of data::DataFrame. Other formats
    // (for example JSON) should add their own adapter in art::io without
    // adding format-specific behavior to data::DataFrame.
    class CsvReader
    {
    public:
        art::data::DataFrame read(const std::string& filename) const;
        art::data::TextLabelDataset read_text_label_dataset(
            const std::string& filename,
            const std::string& label_column,
            const std::vector<std::string>& feature_columns = {}
        ) const;
        void write(
            const std::string& filename,
            const art::data::DataFrame& data_frame
        ) const;
    };
}
