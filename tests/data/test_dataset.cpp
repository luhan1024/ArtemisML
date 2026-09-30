/*
 * ============================================================================
 *                         A R T E M I S M L
 *                         A R T E M I S M L
 * ============================================================================
 * Project: ArtemisML - C++ machine learning library
 * Main contributors: Han Lu & Yihan Wang
 * ============================================================================
 */
#include "art/data/dataset.h"

#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>

using art::data::DataFrame;
using art::data::Dataset;

namespace
{
    template <typename Function>
    std::string expect_error(Function&& function)
    {
        try
        {
            function();
        }
        catch (const std::runtime_error& error)
        {
            return error.what();
        }

        assert(false && "expected std::runtime_error");
        return {};
    }
}

int main()
{
    const DataFrame valid = {
        {"feature", "label"},
        {{"1.5", "0"}, {"2.5", "1"}}
    };

    const Dataset dataset = Dataset::from_dataframe(valid, "label");
    assert(dataset.sample_count() == 2);
    assert(dataset.feature_count() == 1);
    assert(dataset.features(0, 0) == 1.5);
    assert(dataset.labels(1) == 1.0);

    const std::string malformed_row = expect_error([] {
        Dataset::from_dataframe(
            DataFrame{{"feature", "label"}, {{"1.0"}}},
            "label"
        );
    });
    assert(malformed_row.find("row 0") != std::string::npos);
    assert(malformed_row.find("expected 2") != std::string::npos);

    const std::string duplicate_column = expect_error([] {
        Dataset::from_dataframe(
            DataFrame{{"feature", "feature", "label"}, {{"1.0", "2.0", "0"}}},
            "label"
        );
    });
    assert(duplicate_column.find("Duplicate DataFrame column name") != std::string::npos);

    const std::string no_samples = expect_error([] {
        Dataset::from_dataframe(DataFrame{{"feature", "label"}, {}}, "label");
    });
    assert(no_samples.find("at least one sample") != std::string::npos);

    const std::string no_features = expect_error([] {
        Dataset::from_dataframe(DataFrame{{"label"}, {{"1"}}}, "label");
    });
    assert(no_features.find("at least one feature") != std::string::npos);

    const std::string duplicate_feature = expect_error([&] {
        Dataset::from_dataframe(valid, "label", {"feature", "feature"});
    });
    assert(duplicate_feature.find("more than once") != std::string::npos);

    const std::string missing_label = expect_error([&] {
        Dataset::from_dataframe(valid, "missing");
    });
    assert(missing_label.find("Column not found") != std::string::npos);

    const std::string non_finite = expect_error([] {
        Dataset::from_dataframe(
            DataFrame{{"feature", "label"}, {{"nan", "0"}}},
            "label"
        );
    });
    assert(non_finite.find("Non-finite value") != std::string::npos);

    return 0;
}
