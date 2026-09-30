/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #      #   #    #       #   #   #  #
 *   #   #  #   #    #    #####  #   #  #####  ####   #   #  #####
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor  : Yihan Wang
 *
 * ============================================================================
 */
#pragma once

#include "art/base/transformer.h"

#include <cstddef>

namespace art::preprocessing
{
    class PolynomialFeatures final : public base::Transformer
    {
    public:
        explicit PolynomialFeatures(
            std::size_t degree = 2,
            bool include_bias = true
        );

        std::size_t degree() const noexcept;
        bool include_bias() const noexcept;
        std::size_t feature_count() const noexcept;
        std::size_t output_feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        std::size_t degree_;
        bool include_bias_;
        std::size_t feature_count_ = 0;
        std::size_t output_feature_count_ = 0;
    };
}
