/*
 * +----------------------------------------------------------------------------+
 * |                                                                            |
 * | .###.  ####.  #####  #####  #...#  #####  .####  #...#  #....              |
 * | #...#  #...#  ..#..  #....  ##.##  ..#..  #....  ##.##  #....              |
 * | #####  ####.  ..#..  ####.  #.#.#  ..#..  .###.  #.#.#  #....              |
 * | #...#  #.#..  ..#..  #....  #...#  ..#..  ....#  #...#  #....              |
 * | #...#  #..##  ..#..  #####  #...#  #####  ####.  #...#  #####              |
 * |                                                                            |
 * | Han Lu & Yihan Wang                                                        |
 * +----------------------------------------------------------------------------+
 */
#include "art/pipeline/feature_union.h"
#include "art/pipeline/pipeline.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace
{
    class AddTransformer final : public art::base::Transformer
    {
    public:
        explicit AddTransformer(double value) : value_(value) {}

    protected:
        void do_fit(const art::base::FeatureInput&, const art::base::TargetInput&) override
        {
            ++fit_count_;
        }

        art::base::DataInput do_transform(const art::base::DataInput& input) const override
        {
            ++transform_count_;
            return input.array() + value_;
        }

    public:
        int fit_count() const { return fit_count_; }
        int transform_count() const { return transform_count_; }

    private:
        double value_;
        int fit_count_ = 0;
        mutable int transform_count_ = 0;
    };

    class SumRegressor final : public art::base::Regressor
    {
    protected:
        void do_fit(const art::base::FeatureInput& features, const art::base::TargetInput&) override
        {
            fitted_columns_ = features.cols();
        }

        art::base::PredictionOutput do_predict(const art::base::FeatureInput& features) const override
        {
            return features.rowwise().sum();
        }

        double do_score(const art::base::FeatureInput& features, const art::base::TargetInput& targets) const override
        {
            return (do_predict(features) - targets).squaredNorm();
        }

    public:
        Eigen::Index fitted_columns() const { return fitted_columns_; }

    private:
        Eigen::Index fitted_columns_ = 0;
    };

    class FailingTransformer final : public art::base::Transformer
    {
    public:
        explicit FailingTransformer(bool fail) : fail_(fail) {}

        void set_fail(bool fail) { fail_ = fail; }

    protected:
        void do_fit(const art::base::FeatureInput&, const art::base::TargetInput&) override
        {
            if (fail_)
            {
                throw std::runtime_error("intentional transformer fit failure");
            }
        }

        art::base::DataInput do_transform(const art::base::DataInput& input) const override
        {
            return input;
        }

    private:
        bool fail_;
    };

    class FailingRegressor final : public art::base::Regressor
    {
    protected:
        void do_fit(const art::base::FeatureInput&, const art::base::TargetInput&) override
        {
            throw std::runtime_error("intentional estimator fit failure");
        }

        art::base::PredictionOutput do_predict(const art::base::FeatureInput& features) const override
        {
            return art::base::PredictionOutput::Zero(features.rows());
        }

        double do_score(const art::base::FeatureInput&, const art::base::TargetInput&) const override
        {
            return 0.0;
        }
    };
}

int main()
{
    const art::base::FeatureInput input =
        (art::base::FeatureInput(2, 1) << 1.0, 2.0).finished();
    const art::base::TargetInput targets =
        (art::base::TargetInput(2) << 4.0, 5.0).finished();
    auto first = std::make_shared<AddTransformer>(1.0);
    auto second = std::make_shared<AddTransformer>(2.0);
    auto model = std::make_shared<SumRegressor>();

    art::pipeline::Pipeline pipeline({
        {"first", first}, {"second", second}, {"model", model}
    });
    bool rejected = false;
    try { pipeline.predict(input); } catch (const art::base::NotFittedError&) { rejected = true; }
    assert(rejected);
    pipeline.fit(input, targets);
    assert(pipeline.is_fitted());
    assert(first->fit_count() == 1 && second->fit_count() == 1);
    assert((pipeline.predict(input).array() ==
            (Eigen::ArrayXd(2) << 4.0, 5.0).finished().array()).all());
    assert(std::abs(pipeline.score(input, targets)) < 1e-12);
    pipeline.fit(input, targets);
    assert(first->fit_count() == 2 && second->fit_count() == 2);

    auto failing_prefix = std::make_shared<FailingTransformer>(false);
    auto failing_model = std::make_shared<FailingRegressor>();
    art::pipeline::Pipeline failed_pipeline({
        {"prefix", failing_prefix}, {"model", failing_model}
    });
    bool final_fit_failed = false;
    try { failed_pipeline.fit(input, targets); }
    catch (const std::runtime_error&) { final_fit_failed = true; }
    assert(final_fit_failed && !failed_pipeline.is_fitted());

    failing_prefix->set_fail(true);
    bool repeated_fit_failed = false;
    try { failed_pipeline.fit(input, targets); }
    catch (const std::runtime_error&) { repeated_fit_failed = true; }
    assert(repeated_fit_failed && !failed_pipeline.is_fitted());

    auto fit_transform_first = std::make_shared<AddTransformer>(1.0);
    auto fit_transform_last = std::make_shared<AddTransformer>(2.0);
    art::pipeline::Pipeline transform_pipeline({
        {"first", fit_transform_first}, {"last", fit_transform_last}
    });
    const auto fit_transformed = transform_pipeline.fit_transform(input);
    assert(fit_transformed(0, 0) == 4.0);
    assert(fit_transform_first->fit_count() == 1);
    assert(fit_transform_first->transform_count() == 1);
    assert(fit_transform_last->fit_count() == 1);
    assert(fit_transform_last->transform_count() == 1);

    bool empty_pipeline_rejected = false;
    try { art::pipeline::Pipeline empty_pipeline({}); }
    catch (const std::invalid_argument&) { empty_pipeline_rejected = true; }
    assert(empty_pipeline_rejected);

    bool invalid_order_rejected = false;
    try
    {
        art::pipeline::Pipeline invalid_pipeline({
            {"model", model}, {"transformer", first}
        });
    }
    catch (const std::invalid_argument&) { invalid_order_rejected = true; }
    assert(invalid_order_rejected);

    auto left = std::make_shared<AddTransformer>(1.0);
    auto right = std::make_shared<AddTransformer>(10.0);
    art::pipeline::FeatureUnion union_transformer({{"left", left}, {"right", right}});
    union_transformer.fit(input);
    const auto union_output = union_transformer.transform(input);
    assert(union_output.rows() == 2 && union_output.cols() == 2);
    assert(union_output(0, 0) == 2.0 && union_output(0, 1) == 11.0);

    auto union_ok = std::make_shared<AddTransformer>(1.0);
    auto union_failing = std::make_shared<FailingTransformer>(true);
    art::pipeline::FeatureUnion failed_union({
        {"ok", union_ok}, {"failing", union_failing}
    });
    bool union_fit_failed = false;
    try { failed_union.fit(input); }
    catch (const std::runtime_error&) { union_fit_failed = true; }
    assert(union_fit_failed && !failed_union.is_fitted());
    assert(union_ok->is_fitted());

    const art::base::FeatureInput empty_input(0, 1);
    const auto empty_union_output = union_transformer.transform(empty_input);
    assert(empty_union_output.rows() == 0 && empty_union_output.cols() == 2);

    std::cout << "Pipeline tests passed.\n";
    return 0;
}
