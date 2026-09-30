/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/core/errors.h"
#include "art/core/types.h"
#include "art/data/index.h"
#include "art/data/series.h"

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace art::data
{
    struct DataFrame
    {
        std::vector<std::string> column_names;
        std::vector<std::vector<std::string>> rows;

        art::core::Shape shape() const noexcept
        {
            return {rows.size(), column_names.size()};
        }

        bool empty() const noexcept
        {
            return rows.empty() || column_names.empty();
        }

        void validate() const
        {
            std::unordered_set<std::string> seen;
            for (const std::string& name : column_names)
            {
                if (!seen.insert(name).second)
                {
                    throw art::core::DataError("Duplicate DataFrame column name: " + name);
                }
            }
            for (std::size_t row = 0; row < rows.size(); ++row)
            {
                if (rows[row].size() != column_names.size())
                {
                    throw art::core::DimensionError(
                        "DataFrame row " + std::to_string(row) +
                        " has " + std::to_string(rows[row].size()) +
                        " values; expected " + std::to_string(column_names.size())
                    );
                }
            }
        }

        std::size_t column_index(const std::string& name) const
        {
            validate();
            for (std::size_t index = 0; index < column_names.size(); ++index)
            {
                if (column_names[index] == name) return index;
            }
            throw art::core::IndexError("DataFrame column not found: " + name);
        }

        const std::string& at(std::size_t row, std::size_t column) const
        {
            validate();
            if (row >= rows.size() || column >= column_names.size())
            {
                throw art::core::IndexError("DataFrame position out of range");
            }
            return rows[row][column];
        }

        const std::string& at(std::size_t row, const std::string& column) const
        {
            return at(row, column_index(column));
        }

        Series column(std::size_t position) const
        {
            validate();
            if (position >= column_names.size())
            {
                throw art::core::IndexError("DataFrame column position out of range");
            }
            std::vector<std::string> values;
            values.reserve(rows.size());
            for (const auto& row : rows) values.push_back(row[position]);
            return Series(std::move(values), column_names[position], Index::default_index(rows.size()));
        }

        Series column(const std::string& name) const
        {
            const std::size_t position = column_index(name);
            return column(position);
        }

        DataFrame select_columns(const std::vector<std::string>& names) const
        {
            validate();
            DataFrame result;
            result.column_names = names;
            std::vector<std::size_t> source_positions;
            source_positions.reserve(names.size());
            for (const std::string& name : names)
            {
                source_positions.push_back(column_index(name));
            }
            result.rows.reserve(rows.size());
            for (const auto& row : rows)
            {
                std::vector<std::string> selected;
                selected.reserve(names.size());
                for (const std::size_t position : source_positions)
                {
                    selected.push_back(row[position]);
                }
                result.rows.push_back(std::move(selected));
            }
            result.validate();
            return result;
        }

        std::vector<std::string> row(std::size_t position) const
        {
            validate();
            if (position >= rows.size())
            {
                throw art::core::IndexError("DataFrame row position out of range");
            }
            return rows[position];
        }

        DataFrame select_rows(const std::vector<std::size_t>& positions) const
        {
            validate();
            DataFrame result{column_names, {}};
            result.rows.reserve(positions.size());
            for (const std::size_t position : positions) result.rows.push_back(row(position));
            return result;
        }

        Index row_index() const
        {
            return Index::default_index(rows.size());
        }
    };
}
