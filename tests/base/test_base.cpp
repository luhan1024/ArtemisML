#include "art/base/base.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>

namespace
{
    class AddOneTransformer final : public art::base::Transformer
    {
    protected:
        void do_fit(
            const art::base::FeatureInput&,
            const art::base::TargetInput&
        ) override
        {
            fitted_value_ = true;
        }

        art::base::DataInput do_transform(
            const art::base::DataInput& input
        ) const override
        {
            return input.array() + 1.0;
        }

    private:
        bool fitted_value_ = false;
    };

    class FailingTransformer final : public art::base::Transformer
    {
    public:
        void fail_next_fit() noexcept
        {
            fail_next_ = true;
        }

        std::size_t last_target_count() const noexcept
        {
            return last_target_count_;
        }

    protected:
        void do_fit(
            const art::base::FeatureInput&,
            const art::base::TargetInput& targets
        ) override
        {
            last_target_count_ = static_cast<std::size_t>(targets.size());
            if (fail_next_)
            {
                fail_next_ = false;
                throw std::runtime_error("synthetic transformer fit failure");
            }
        }

        art::base::DataInput do_transform(
            const art::base::DataInput& input
        ) const override
        {
            return input;
        }

    private:
        bool fail_next_ = false;
        std::size_t last_target_count_ = 0;
    };

    class MeanRegressor final : public art::base::Regressor
    {
    protected:
        void do_fit(
            const art::base::FeatureInput&,
            const art::base::TargetInput& targets
        ) override
        {
            mean_ = targets.mean();
        }

        art::base::PredictionOutput do_predict(
            const art::base::FeatureInput& features
        ) const override
        {
            return art::base::PredictionOutput::Constant(
                features.rows(), mean_
            );
        }

        double do_score(
            const art::base::FeatureInput& features,
            const art::base::TargetInput& targets
        ) const override
        {
            const auto predictions = do_predict(features);
            return (predictions - targets).squaredNorm();
        }

    private:
        double mean_ = 0.0;
    };

    class FailingRegressor final : public art::base::Regressor
    {
    public:
        void fail_next_fit() noexcept
        {
            fail_next_ = true;
        }

    protected:
        void do_fit(
            const art::base::FeatureInput&,
            const art::base::TargetInput& targets
        ) override
        {
            mean_ = targets.mean();
            if (fail_next_)
            {
                fail_next_ = false;
                throw std::runtime_error("synthetic regressor fit failure");
            }
        }

        art::base::PredictionOutput do_predict(
            const art::base::FeatureInput& features
        ) const override
        {
            return art::base::PredictionOutput::Constant(
                features.rows(), mean_
            );
        }

        double do_score(
            const art::base::FeatureInput& features,
            const art::base::TargetInput& targets
        ) const override
        {
            return (do_predict(features) - targets).squaredNorm();
        }

    private:
        bool fail_next_ = false;
        double mean_ = 0.0;
    };
}

int main()
{
    const art::base::FeatureInput features =
        art::base::FeatureInput::Zero(2, 1);
    const art::base::TargetInput targets =
        (art::base::TargetInput(2) << 1.0, 3.0).finished();

    AddOneTransformer transformer;
    assert(!transformer.is_fitted());
    bool transform_rejected = false;
    try
    {
        transformer.transform(features);
    }
    catch (const art::base::NotFittedError&)
    {
        transform_rejected = true;
    }
    assert(transform_rejected);

    const auto transformed = transformer.fit_transform(features, targets);
    assert(transformer.is_fitted());
    assert(std::abs(transformed(0, 0) - 1.0) < 1e-12);

    FailingTransformer failing_transformer;
    failing_transformer.fit(features);
    assert(failing_transformer.is_fitted());
    assert(failing_transformer.last_target_count() == 0);
    failing_transformer.fail_next_fit();
    bool transformer_fit_failed = false;
    try
    {
        failing_transformer.fit_transform(features, targets);
    }
    catch (const std::runtime_error&)
    {
        transformer_fit_failed = true;
    }
    assert(transformer_fit_failed);
    assert(!failing_transformer.is_fitted());
    bool failed_transform_rejected = false;
    try
    {
        failing_transformer.transform(features);
    }
    catch (const art::base::NotFittedError&)
    {
        failed_transform_rejected = true;
    }
    assert(failed_transform_rejected);
    failing_transformer.fit_transform(features, targets);
    assert(failing_transformer.last_target_count() ==
           static_cast<std::size_t>(targets.size()));

    MeanRegressor regressor;
    bool predict_rejected = false;
    try
    {
        regressor.predict(features);
    }
    catch (const art::base::NotFittedError&)
    {
        predict_rejected = true;
    }
    assert(predict_rejected);

    regressor.fit(features, targets);
    assert(regressor.is_fitted());
    assert(regressor.predict(features).size() == features.rows());
    assert(std::isfinite(regressor.score(features, targets)));

    FailingRegressor failing_regressor;
    failing_regressor.fit(features, targets);
    assert(failing_regressor.is_fitted());
    failing_regressor.fail_next_fit();
    bool regressor_fit_failed = false;
    try
    {
        failing_regressor.fit(features, targets);
    }
    catch (const std::runtime_error&)
    {
        regressor_fit_failed = true;
    }
    assert(regressor_fit_failed);
    assert(!failing_regressor.is_fitted());
    bool failed_predict_rejected = false;
    bool failed_score_rejected = false;
    try
    {
        failing_regressor.predict(features);
    }
    catch (const art::base::NotFittedError&)
    {
        failed_predict_rejected = true;
    }
    try
    {
        failing_regressor.score(features, targets);
    }
    catch (const art::base::NotFittedError&)
    {
        failed_score_rejected = true;
    }
    assert(failed_predict_rejected);
    assert(failed_score_rejected);

    std::cout << "Base protocol test passed.\n";
    return 0;
}
