#include "csvReader.h"

#include <fstream>
#include <sstream>
#include <stdexcept>

namespace
{
    std::vector<std::string> parseCsvLine(const std::string& line)
    {
        std::vector<std::string> cells;
        std::string cell;
        bool insideQuotes = false;

        for (std::size_t index = 0; index < line.size(); ++index)
        {
            const char character = line[index];

            if (character == '"')
            {
                if (insideQuotes && index + 1 < line.size() && line[index + 1] == '"')
                {
                    cell += '"';
                    ++index;
                }
                else
                {
                    insideQuotes = !insideQuotes;
                }
            }
            else if (character == ',' && !insideQuotes)
            {
                cells.push_back(cell);
                cell.clear();
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

        cells.push_back(cell);
        return cells;
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

namespace art::data
{
DataFrame CsvReader::read(const std::string& filename) const
{
    std::ifstream file(filename);

    if (!file.is_open())
    {
        throw std::runtime_error("Failed to open file: " + filename);
    }

    std::string headerLine;

    if (!std::getline(file, headerLine))
    {
        throw std::runtime_error("Failed to read header line from file: " + filename);
    }

    DataFrame data_frame;
    data_frame.column_names = parseCsvLine(headerLine);

    if (headerLine.empty() || data_frame.column_names.empty())
    {
        throw std::runtime_error("CSV header is empty: " + filename);
    }

    std::string dataLine;
    std::size_t lineNumber = 1;

    while (std::getline(file, dataLine))
    {
        ++lineNumber;

        if (dataLine.empty())
        {
            continue;
        }

        std::vector<std::string> row = parseCsvLine(dataLine);

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

void CsvReader::write(const std::string& filename, const DataFrame& data_frame) const
{
    if (data_frame.column_names.empty())
    {
        throw std::runtime_error("Cannot write CSV with an empty header");
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
        if (row.size() != data_frame.column_names.size())
        {
            throw std::runtime_error("Cannot write CSV row with an invalid column count");
        }

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
}
}
