/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include <stdexcept>
#include <string>

namespace art::core
{
class Error : public std::runtime_error
{
public:
    explicit Error(const std::string& message) : std::runtime_error(message) {}
};

class ParameterError : public Error
{
public:
    explicit ParameterError(const std::string& message) : Error(message) {}
};

class DataError : public Error
{
public:
    explicit DataError(const std::string& message) : Error(message) {}
};

class DimensionError : public DataError
{
public:
    explicit DimensionError(const std::string& message) : DataError(message) {}
};

class TypeError : public DataError
{
public:
    explicit TypeError(const std::string& message) : DataError(message) {}
};

class MissingValueError : public DataError
{
public:
    explicit MissingValueError(const std::string& message) : DataError(message) {}
};

class IndexError : public DataError
{
public:
    explicit IndexError(const std::string& message) : DataError(message) {}
};

class UnsupportedOperationError : public Error
{
public:
    explicit UnsupportedOperationError(const std::string& message) : Error(message) {}
};
}
