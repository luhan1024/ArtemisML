/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#pragma once

#include "art/base/classifier.h"

namespace art::linear_model
{
    class LogisticRegression final : public base::Classifier
    {
    public:
        LogisticRegression() = default;

    protected:
        void do_fit(const base::FeatureInput&, const base::TargetInput&) override;
        base::PredictionOutput do_predict(const base::FeatureInput&) const override;
        double do_score(const base::FeatureInput&, const base::TargetInput&) const override;
    };
}
