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
    class MinMaxScaler final : public base::Transformer
    {
    public:
        MinMaxScaler(double lower = 0.0, double upper = 1.0);

        double lower() const noexcept;
        double upper() const noexcept;
        const Eigen::VectorXd& data_min() const;
        const Eigen::VectorXd& data_max() const;
        std::size_t feature_count() const noexcept;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::DataInput do_transform(const base::DataInput&) const override;

    private:
        double lower_;
        double upper_;
        Eigen::VectorXd data_min_;
        Eigen::VectorXd data_max_;
        Eigen::VectorXd scale_;
        std::size_t feature_count_ = 0;
    };
}
