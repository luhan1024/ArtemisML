#pragma once

#include "art/base/base.h"

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace art::pipeline
{
    class Pipeline final
    {
    public:
        using Step = std::pair<std::string, std::shared_ptr<base::Estimator>>;

        // Pipeline owns the step objects through shared_ptr. A repeated fit
        // retrains every step in order; the pipeline becomes fitted only after
        // the complete pass succeeds.
        explicit Pipeline(std::vector<Step> steps);

        void fit(
            const base::FeatureInput& features,
            const base::TargetInput& targets = base::TargetInput{}
        );

        base::DataInput transform(const base::DataInput& input) const;

        base::DataInput fit_transform(
            const base::DataInput& input,
            const base::TargetInput& targets = base::TargetInput{}
        );

        base::PredictionOutput predict(
            const base::FeatureInput& features
        ) const;

        double score(
            const base::FeatureInput& features,
            const base::TargetInput& targets
        ) const;

        bool is_fitted() const noexcept;
        std::size_t size() const noexcept;
        const std::vector<Step>& steps() const noexcept;
        const Step& step(std::size_t index) const;
        const Step& step(const std::string& name) const;
        const std::shared_ptr<base::Estimator>& final_estimator() const;

    private:
        std::vector<Step> steps_;
        bool fitted_ = false;
    };
}
