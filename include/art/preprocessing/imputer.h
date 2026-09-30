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

namespace art::preprocessing
{
    enum class ImputationStrategy
    {
        Mean,
        Constant
    };

    class Imputer final : public base::Transformer
    {
    public:
        explicit Imputer(
            ImputationStrategy strategy = ImputationStrategy::Mean,
            double fill_value = 0.0
        );

        ImputationStrategy strategy() const noexcept;
        const Eigen::VectorXd& statistics() const;
        std::size_t feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        ImputationStrategy strategy_;
        double fill_value_;
        Eigen::VectorXd statistics_;
        std::size_t feature_count_ = 0;
    };
}
