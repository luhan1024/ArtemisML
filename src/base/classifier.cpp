/*
 *  ###    ####   #####  #####  #   #  #####   ####  #   #  #    
 * #   #  #   #     #    #      ## ##    #    #   #  ## ##  #    
 * #####  ####      #    ###    # # #    #    ####   # # #  #    
 * #   #  # #       #    #      #   #    #    # #    #   #  #    
 * #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  ##### 
 */
#include "art/base/classifier.h"

namespace art::base
{
    double Classifier::score(
        const FeatureInput& features,
        const TargetInput& targets
    ) const
    {
        if (!is_fitted())
        {
            throw NotFittedError("score()");
        }
        return do_score(features, targets);
    }
}
