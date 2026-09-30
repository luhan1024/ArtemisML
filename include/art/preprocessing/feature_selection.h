/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 * #####  ####      #    ###    # # #    #    ####   # # #  #
 * #   #  # #       #    #      #   #    #    # #    #   #  #
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
 */
#pragma once

#include "art/base/transformer.h"

#include <cstddef>
#include <vector>

namespace art::preprocessing
{
    class SelectColumns final : public base::Transformer
    {
    public:
        explicit SelectColumns(std::vector<std::size_t> columns);

        const std::vector<std::size_t>& columns() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        std::vector<std::size_t> columns_;
        std::size_t feature_count_ = 0;
    };
}
