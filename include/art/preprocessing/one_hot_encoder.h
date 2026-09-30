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
    enum class UnknownCategoryPolicy
    {
        Error,
        Ignore
    };

    class OneHotEncoder final : public base::Transformer
    {
    public:
        explicit OneHotEncoder(
            UnknownCategoryPolicy policy = UnknownCategoryPolicy::Error
        );

        UnknownCategoryPolicy unknown_policy() const noexcept;
        const std::vector<std::vector<double>>& categories() const;
        std::size_t feature_count() const noexcept;
        std::size_t output_feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        UnknownCategoryPolicy policy_;
        std::vector<std::vector<double>> categories_;
        std::size_t feature_count_ = 0;
        std::size_t output_feature_count_ = 0;
    };
}
