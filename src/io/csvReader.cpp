/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#include "art/io/csvReader.h"

#include <fstream>
#include <stdexcept>
#include <utility>

namespace
{
    bool readCsvRecord(
        std::istream& input,
        std::vector<std::string>& record,
        std::size_t& lineNumber
    )
    {
        record.clear();
        std::string cell;
        bool insideQuotes = false;
        bool readAnyCharacter = false;

        char character = '\0';
        while (input.get(character))
        {
            readAnyCharacter = true;

            if (insideQuotes)
            {
                if (character == '"')
                {
                    if (input.peek() == '"')
                    {
                        cell += '"';
                        input.get();
                    }
                    else
                    {
                        insideQuotes = false;
                    }
                }
                else
                {
                    cell += character;
                }
            }
            else if (character == '"')
            {
                insideQuotes = true;
            }
            else if (character == ',')
            {
                record.push_back(cell);
                cell.clear();
            }
            else if (character == '\n')
            {
                ++lineNumber;
                record.push_back(cell);
                return true;
            }
            else if (character == '\r')
            {
                if (input.peek() == '\n')
                {
                    input.get();
                }

                ++lineNumber;
                record.push_back(cell);
                return true;
            }
            else
            {
                cell += character;
            }
        }

        if (insideQuotes)
        {
            throw std::runtime_error("Unclosed quote in CSV line");
        }

        if (!readAnyCharacter && record.empty() && cell.empty())
        {
            return false;
        }

        record.push_back(cell);
        return true;
    }

    std::string escapeCsvCell(const std::string& cell)
    {
        const bool needsQuotes = cell.find_first_of(",\"\r\n") != std::string::npos;

        if (!needsQuotes)
        {
            return cell;
        }

        std::string escaped = "\"";

        for (const char character : cell)
        {
            if (character == '"')
            {
                escaped += "\"\"";
            }
            else
            {
                escaped += character;
            }
        }

        escaped += '"';
        return escaped;
    }
}

namespace art::io
{
art::data::DataFrame CsvReader::read(const std::string& filename) const
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::vector<std::string> header;
    std::size_t lineNumber = 1;

    if (!readCsvRecord(file, header, lineNumber))
    {
        throw std::runtime_error("Failed to read header record from file: " + filename);
    }

    art::data::DataFrame data_frame;
    data_frame.column_names = std::move(header);

    if (data_frame.column_names.empty() ||
        (data_frame.column_names.size() == 1 && data_frame.column_names.front().empty()))
    {
        throw std::runtime_error("CSV header is empty: " + filename);
    }

    std::vector<std::string> row;
    while (readCsvRecord(file, row, lineNumber))
    {
        if (row.size() == 1 && row.front().empty())
        {
            continue;
        }

        if (row.size() != data_frame.column_names.size())
        {
            throw std::runtime_error(
                "Column count mismatch at line " +
                std::to_string(lineNumber)
            );
        }

        data_frame.rows.push_back(row);
    }

    return data_frame;
}

art::data::TextLabelDataset CsvReader::read_text_label_dataset(
    const std::string& filename,
    const std::string& label_column,
    const std::vector<std::string>& feature_columns
) const
{
    return art::data::TextLabelDataset::from_dataframe(
        read(filename), label_column, feature_columns
    );
}

void CsvReader::write(
    const std::string& filename,
    const art::data::DataFrame& data_frame
) const
{
    if (data_frame.column_names.empty())
    {
        throw std::runtime_error("Cannot write CSV with an empty header");
    }

    for (const std::vector<std::string>& row : data_frame.rows)
    {
        if (row.size() != data_frame.column_names.size())
        {
            throw std::runtime_error("Cannot write CSV row with an invalid column count");
        }
    }

    std::ofstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    for (std::size_t index = 0; index < data_frame.column_names.size(); ++index)
    {
        if (index > 0)
        {
            file << ',';
        }

        file << escapeCsvCell(data_frame.column_names[index]);
    }

    file << '\n';

    for (const std::vector<std::string>& row : data_frame.rows)
    {
        for (std::size_t index = 0; index < row.size(); ++index)
        {
            if (index > 0)
            {
                file << ',';
            }

            file << escapeCsvCell(row[index]);
        }

        file << '\n';
    }

    if (!file)
    {
        throw std::runtime_error("Failed while writing CSV file: " + filename);
    }
}
}
