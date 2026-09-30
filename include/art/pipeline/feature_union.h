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
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace art::pipeline
{
    class FeatureUnion final : public base::Transformer
    {
    public:
        using Step = std::pair<std::string, std::shared_ptr<base::Transformer>>;

        // FeatureUnion owns the transformer objects through shared_ptr. Every
        // branch sees the same input and the branch matrices are concatenated
        // column-wise in declaration order. If a branch throws during fit,
        // this union remains unfitted; already-fitted branches are not rolled
        // back because the base Transformer protocol has no reset operation.
        explicit FeatureUnion(std::vector<Step> steps);

        std::size_t size() const noexcept;
        const std::vector<Step>& steps() const noexcept;
        const Step& step(std::size_t index) const;
        const Step& step(const std::string& name) const;

    protected:
        void do_fit(
            const base::FeatureInput& features,
            const base::TargetInput& targets
        ) override;

        base::DataInput do_transform(
            const base::DataInput& input
        ) const override;

    private:
        std::vector<Step> steps_;
    };
}
