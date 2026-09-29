#include "csvReader.h"
#include "dataset.h"
#include <stdexcept>
#include <cassert>
#include <iostream>

using namespace art::data;

int main()
{
    CsvReader reader;
    DataFrame data_frame = reader.read("data.csv");

    assert(data_frame.column_names.size() == 3);
    assert(data_frame.rows.size() == 3);

    assert(data_frame.column_names[0] == "age");
    assert(data_frame.column_names[1] == "height");
    assert(data_frame.column_names[2] == "label");

    assert(data_frame.rows[0][0] == "20");
    assert(data_frame.rows[0][1] == "175");
    assert(data_frame.rows[0][2] == "1");
bool exception_caught = false;

try
{
    reader.read("not_exist.csv");
}
catch (const std::runtime_error& error)
{
    exception_caught = true;
    std::cout << "Caught expected exception: "
              << error.what() << '\n';
}

assert(exception_caught);
exception_caught = false;

try
{
    reader.read("bad.csv");
}
catch (const std::runtime_error& error)
{
    exception_caught = true;
    std::cout << "Caught expected exception: "
              << error.what() << '\n';
}

assert(exception_caught);
exception_caught = false;

try
{
    reader.read("empty.csv");
}
catch (const std::runtime_error& error)
{
    exception_caught = true;
    std::cout << "Caught expected exception: "
              << error.what() << '\n';
}

assert(exception_caught);

    DataFrame data_frame_to_write;
    data_frame_to_write.column_names = {"name", "description", "value"};
    data_frame_to_write.rows = {
        {"sample", "contains,comma", "1.5"},
        {"quoted", "contains \"quote\"", "2.5"}
    };

    reader.write("roundtrip.csv", data_frame_to_write);
    DataFrame roundtrip_data_frame = reader.read("roundtrip.csv");

    assert(roundtrip_data_frame.column_names == data_frame_to_write.column_names);
    assert(roundtrip_data_frame.rows == data_frame_to_write.rows);

    Dataset dataset = Dataset::from_csv("data.csv", "label");

    assert(dataset.sample_count() == 3);
    assert(dataset.feature_count() == 2);
    assert(dataset.feature_names[0] == "age");
    assert(dataset.feature_names[1] == "height");
    assert(dataset.labels(0) == 1.0);
    assert(dataset.features(0, 0) == 20.0);
    assert(dataset.features(0, 1) == 175.0);

    Dataset selected_dataset = Dataset::from_csv(
        "data.csv",
        "label",
        {"height"}
    );

    assert(selected_dataset.feature_count() == 1);
    assert(selected_dataset.feature_names[0] == "height");
    assert(selected_dataset.features(0, 0) == 175.0);

std::cout << "CSV test passed.\n";
    return 0;
}
