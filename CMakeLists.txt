# ============================================================================
#                         A R T E M I S M L
#                         A R T E M I S M L
# ============================================================================
# Project: ArtemisML - C++ machine learning library
# Main contributors: Han Lu & Yihan Wang
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(ArtemisML LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(artemisml
    src/base/classifier.cpp
    src/base/predictor.cpp
    src/base/regressor.cpp
    src/base/transformer.cpp
    src/io/csvReader.cpp
    src/data/dataset.cpp
    src/optim/optimizer.cpp
    src/linear_model/linear_regression.cpp
    src/pipeline/pipeline.cpp
    src/pipeline/feature_union.cpp
    src/linear_model/ridge.cpp
    src/linear_model/logistic_regression.cpp
)

target_include_directories(artemisml
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
        ${PROJECT_SOURCE_DIR}/third_party/eigen
)

if (MSVC)
    target_compile_options(artemisml
        PUBLIC
            /utf-8
        PRIVATE
            /W4
    )
else()
    target_compile_options(artemisml PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()

add_executable(test_csvReader tests/io/test_csvReader.cpp)
target_link_libraries(test_csvReader PRIVATE artemisml)
add_test(NAME csv_reader COMMAND test_csvReader WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_optimizer tests/optim/test_optimizer.cpp)
target_link_libraries(test_optimizer PRIVATE artemisml)
add_test(NAME optimizer COMMAND test_optimizer WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_linear_regression tests/linear_model/test_linear_regression.cpp)
target_link_libraries(test_linear_regression PRIVATE artemisml)
add_test(NAME linear_regression COMMAND test_linear_regression WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_base tests/base/test_base.cpp)
target_link_libraries(test_base PRIVATE artemisml)
add_test(NAME base COMMAND test_base WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_dataset tests/data/test_dataset.cpp)
target_link_libraries(test_dataset PRIVATE artemisml)
add_test(NAME dataset COMMAND test_dataset WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_executable(test_pipeline tests/pipeline/test_pipeline.cpp)
target_link_libraries(test_pipeline PRIVATE artemisml)
add_test(NAME pipeline COMMAND test_pipeline WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})
