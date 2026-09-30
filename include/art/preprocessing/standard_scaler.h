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
    class StandardScaler final : public base::Transformer
    {
    public:
        explicit StandardScaler(bool with_mean = true, bool with_std = true);

        const Eigen::VectorXd& mean() const;
        const Eigen::VectorXd& scale() const;
        std::size_t feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        bool with_mean_;
        bool with_std_;
        Eigen::VectorXd mean_;
        Eigen::VectorXd scale_;
        std::size_t feature_count_ = 0;
    };
}
