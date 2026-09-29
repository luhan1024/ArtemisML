cmake_minimum_required(VERSION 3.20)

project(ArtemisML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(artemisml
    src/csvReader.cpp
    src/dataset.cpp
    src/optimizer.cpp
    src/linear_regression.cpp
)

target_include_directories(artemisml
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/eigen
)

if (MSVC)
    target_compile_options(artemisml PRIVATE /W4)
else()
    target_compile_options(artemisml PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(test_csvReader tests/test_csvReader.cpp)
target_link_libraries(test_csvReader PRIVATE artemisml)
add_test(NAME csv_reader COMMAND test_csvReader WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_optimizer tests/test_optimizer.cpp)
target_link_libraries(test_optimizer PRIVATE artemisml)
add_test(NAME optimizer COMMAND test_optimizer WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_linear_regression tests/test_linear_regression.cpp)
target_link_libraries(test_linear_regression PRIVATE artemisml)
add_test(NAME linear_regression COMMAND test_linear_regression WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
