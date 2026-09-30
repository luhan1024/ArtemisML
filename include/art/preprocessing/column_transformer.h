/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include "art/base/transformer.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace art::preprocessing
{
    class ColumnTransformer final : public base::Transformer
    {
    public:
        struct Specification
        {
            std::string name;
            std::vector<std::size_t> columns;
            std::shared_ptr<base::Transformer> transformer;
        };

        explicit ColumnTransformer(std::vector<Specification> specifications);

        const std::vector<Specification>& specifications() const noexcept;
        std::size_t input_feature_count() const noexcept;
        std::size_t output_feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        std::vector<Specification> specifications_;
        std::size_t input_feature_count_ = 0;
        std::size_t output_feature_count_ = 0;
    };
}
