/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#include "art/linear_model/logistic_regression.h"

#include <stdexcept>

namespace art::linear_model
{
void LogisticRegression::do_fit(const base::FeatureInput&, const base::TargetInput&)
{
    throw std::logic_error("LogisticRegression is not implemented yet");
}

base::PredictionOutput LogisticRegression::do_predict(const base::FeatureInput&) const
{
    throw std::logic_error("LogisticRegression is not implemented yet");
}

double LogisticRegression::do_score(const base::FeatureInput&, const base::TargetInput&) const
{
    throw std::logic_error("LogisticRegression is not implemented yet");
}
}
