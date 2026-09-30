/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#include "art/data/dataframe.h"
#include "art/data/dtype.h"
#include "art/data/index.h"
#include "art/data/missing.h"
#include "art/data/series.h"

#include <cassert>
#include <string>
#include <vector>

int main()
{
    const art::data::Index index({"row-a", "row-b"});
    assert(index.size() == 2);
    assert(index.at(1) == "row-b");
    assert(index.position_of("row-a") == 0);
    assert(art::data::Index::default_index(2).at(1) == "1");

    const art::data::Index score_index({"row-a", "row-b", "row-c"});
    const art::data::Series series({"1", "2", "3"}, "score", score_index);
    assert(series.size() == 3);
    assert(series.at(1) == "2");
    assert(series.dtype().kind() == art::data::DTypeKind::integer);
    assert(series.to_numeric()[2] == 3.0);

    bool non_finite_rejected = false;
    try
    {
        art::data::Series(std::vector<std::string>{"inf"}, "score").to_numeric();
    }
    catch (const art::core::DataError&)
    {
        non_finite_rejected = true;
    }
    assert(non_finite_rejected);

    assert(art::data::is_missing("null"));
    assert(!art::data::is_missing(""));
    assert(art::data::is_missing("", art::data::MissingPolicy{true, true, true}));

    const art::data::DataFrame frame = {
        {"name", "score"},
        {{"a", "1.5"}, {"b", "2.5"}}
    };
    assert(frame.shape().rows == 2);
    assert(frame.shape().columns == 2);
    assert(frame.at(0, "name") == "a");
    assert(frame.column("score").to_numeric()[1] == 2.5);
    assert(frame.select_columns({"score"}).shape().columns == 1);
    assert(frame.select_rows({1}).at(0, 0) == "b");

    bool rejected = false;
    try
    {
        frame.at(5, 0);
    }
    catch (const art::core::IndexError&)
    {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
