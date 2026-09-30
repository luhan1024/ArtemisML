/*
 * ============================================================================
 *
 *    ###   ####   #####  #####  #   #  #####   ####   #   #  #
 *   #   #  #   #    #    #      ## ##    #    #      ## ##  #
 *   #####  ####     #    ####   # # #    #     ###   # # #  #
 *   #   #  #  #     #    #      #   #    #       #   #  #  #
 *   #   #  #   #    #    #####  #   #  #####  ####   #   #  #####
 *
 *                         ARTEMISML
 *
 * Author      : Han Lu
 * Contributor  : Yihan Wang
 *
 * ============================================================================
 */
#include "art/io/model.h"

#include <fstream>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace art
{
namespace
{
    constexpr const char* signature = "ARTEMISML_MODEL";
    constexpr std::size_t format_version = 1;
    constexpr std::size_t maximum_dimension = 1000000;

    template <typename Value>
    void write_field(std::ostream& output, const char* key, const Value& value)
    {
        output << key << ' ' << value << '\n';
    }

    void expect_key(std::istream& input, const char* expected)
    {
        std::string key;
        if (!(input >> key) || key != expected)
        {
            throw std::runtime_error(
                std::string("Invalid ARTEMISML model field; expected ") + expected
            );
        }
    }

    template <typename Value>
    Value read_field(std::istream& input, const char* key)
    {
        expect_key(input, key);
        Value value{};
        if (!(input >> value))
        {
            throw std::runtime_error(
                std::string("Invalid ARTEMISML model value for ") + key
            );
        }
        return value;
    }

    void write_model(
        const linear_model::LogisticRegression& model,
        const std::string& path
    )
    {
        if (!model.is_fitted())
        {
            throw std::logic_error("Cannot save an unfitted model");
        }

        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open model file for writing: " + path);
        }
        output << signature << '\n';
        write_field(output, "format_version", format_version);
        output << "type_name " << std::quoted(std::string(model.type_name())) << '\n';
        write_field(output, "fitted", 1);
        write_field(output, "feature_count", model.coefficients().rows() - 1);
        write_field(output, "class_count", model.class_count());
        write_field(output, "binary", model.class_count() == 2 ? 1 : 0);
        write_field(output, "decision_threshold", model.decision_threshold());
        write_field(output, "max_iterations", model.options().max_iterations);
        write_field(output, "tolerance", model.options().tolerance);
        write_field(output, "l2_penalty", model.options().l2_penalty);
        write_field(output, "label_count", model.class_labels().size());
        for (const std::string& label : model.class_labels())
        {
            output << "label " << std::quoted(label) << '\n';
        }
        const Eigen::MatrixXd& coefficients = model.coefficients();
        write_field(output, "coefficient_rows", coefficients.rows());
        write_field(output, "coefficient_cols", coefficients.cols());
        output << "coefficients\n" << std::setprecision(17);
        for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < coefficients.cols(); ++column)
            {
                output << coefficients(row, column) << ' ';
            }
            output << '\n';
        }
        if (!output)
        {
            throw std::runtime_error("Failed while writing model file: " + path);
        }
    }

    void write_model(
        const linear_model::LinearRegression& model,
        const std::string& path
    )
    {
        if (!model.is_fitted())
        {
            throw std::logic_error("Cannot save an unfitted model");
        }
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output.is_open())
        {
            throw std::runtime_error("Failed to open model file for writing: " + path);
        }
        output << signature << '\n';
        write_field(output, "format_version", format_version);
        output << "type_name " << std::quoted(std::string(model.type_name())) << '\n';
        write_field(output, "fitted", 1);
        write_field(output, "feature_count", model.feature_count());
        write_field(output, "max_iterations", model.max_iterations());
        write_field(output, "tolerance", model.tolerance());
        const Eigen::VectorXd& parameters = model.parameters();
        write_field(output, "parameter_count", parameters.size());
        output << "parameters\n" << std::setprecision(17);
        for (Eigen::Index index = 0; index < parameters.size(); ++index)
        {
            output << parameters(index) << ' ';
        }
        output << '\n';
        if (!output)
        {
            throw std::runtime_error("Failed while writing model file: " + path);
        }
    }

    void write_model(const linear_model::Ridge& model, const std::string& path)
    {
        if (!model.is_fitted()) throw std::logic_error("Cannot save an unfitted model");
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output.is_open()) throw std::runtime_error("Failed to open model file for writing: " + path);
        output << signature << '\n';
        write_field(output, "format_version", format_version);
        output << "type_name " << std::quoted(std::string(model.type_name())) << '\n';
        write_field(output, "fitted", 1);
        write_field(output, "feature_count", model.feature_count());
        write_field(output, "alpha", model.alpha());
        write_field(output, "iterations", model.iterations());
        write_field(output, "converged", model.converged() ? 1 : 0);
        const Eigen::VectorXd& parameters = model.parameters();
        write_field(output, "parameter_count", parameters.size());
        output << "parameters\n" << std::setprecision(17);
        for (Eigen::Index index = 0; index < parameters.size(); ++index)
            output << parameters(index) << ' ';
        output << '\n';
    }

    void write_model(
        const linear_model::RidgeClassifier& model,
        const std::string& path
    )
    {
        if (!model.is_fitted()) throw std::logic_error("Cannot save an unfitted model");
        std::ofstream output(path, std::ios::out | std::ios::trunc);
        if (!output.is_open()) throw std::runtime_error("Failed to open model file for writing: " + path);
        output << signature << '\n';
        write_field(output, "format_version", format_version);
        output << "type_name " << std::quoted(std::string(model.type_name())) << '\n';
        write_field(output, "fitted", 1);
        write_field(output, "feature_count", model.feature_count());
        write_field(output, "class_count", model.class_count());
        write_field(output, "binary", model.class_count() == 2 ? 1 : 0);
        write_field(output, "alpha", model.alpha());
        write_field(output, "label_count", model.class_labels().size());
        for (const std::string& label : model.class_labels())
            output << "label " << std::quoted(label) << '\n';
        const Eigen::MatrixXd& coefficients = model.coefficients();
        write_field(output, "coefficient_rows", coefficients.rows());
        write_field(output, "coefficient_cols", coefficients.cols());
        output << "coefficients\n" << std::setprecision(17);
        for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < coefficients.cols(); ++column)
                output << coefficients(row, column) << ' ';
            output << '\n';
        }
    }
}

Model::Model(Storage model)
    : model_(std::move(model))
{
}

void Model::restore_state(
    const Eigen::MatrixXd& coefficients,
    std::size_t feature_count,
    std::size_t class_count,
    double decision_threshold,
    const linear_model::LogisticRegressionOptions& options,
    const std::vector<std::string>& class_labels
)
{
    std::visit([&](auto& model) {
        if (!model) throw std::logic_error("Model handle is empty");
        using ModelPointer = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<
            ModelPointer, std::shared_ptr<linear_model::LogisticRegression>>)
        {
            model->restore_state(
                coefficients, feature_count, class_count, decision_threshold,
                options, class_labels
            );
        }
        else
        {
            throw std::logic_error("Model type does not support classification state");
        }
    }, model_);
}

void Model::restore_linear_state(
    const Eigen::VectorXd& parameters,
    std::size_t feature_count,
    std::size_t max_iterations,
    double tolerance
)
{
    auto model = std::get_if<
        std::shared_ptr<linear_model::LinearRegression>
    >(&model_);
    if (model == nullptr || !*model)
    {
        throw std::logic_error("Model type does not support regression state");
    }
    (*model)->restore_state(
        parameters, feature_count, max_iterations, tolerance
    );
}

void Model::restore_ridge_state(
    const Eigen::VectorXd& parameters,
    std::size_t feature_count,
    double alpha,
    std::size_t iterations,
    bool converged
)
{
    auto model = std::get_if<std::shared_ptr<linear_model::Ridge>>(&model_);
    if (model == nullptr || !*model)
        throw std::logic_error("Model type does not support Ridge state");
    (*model)->restore_state(parameters, feature_count, alpha, iterations, converged);
}

void Model::restore_ridge_classifier_state(
    const Eigen::MatrixXd& coefficients,
    std::size_t feature_count,
    std::size_t class_count,
    double alpha,
    const std::vector<std::string>& labels
)
{
    auto model = std::get_if<
        std::shared_ptr<linear_model::RidgeClassifier>
    >(&model_);
    if (model == nullptr || !*model)
        throw std::logic_error("Model type does not support RidgeClassifier state");
    (*model)->restore_state(
        coefficients, feature_count, class_count, alpha, labels
    );
}

std::string Model::type_name() const
{
    return std::visit([](const auto& model) -> std::string {
        if (!model) throw std::logic_error("Model handle is empty");
        return model->type_name();
    }, model_);
}

bool Model::is_fitted() const noexcept
{
    return std::visit([](const auto& model) {
        return model != nullptr && model->is_fitted();
    }, model_);
}

Eigen::VectorXd Model::predict(const Eigen::MatrixXd& features) const
{
    return std::visit([&](const auto& model) {
        if (!model) throw std::logic_error("Model handle is empty");
        return model->predict(features);
    }, model_);
}

Eigen::MatrixXd Model::predict_proba(const Eigen::MatrixXd& features) const
{
    if (const auto* model = std::get_if<
        std::shared_ptr<linear_model::LogisticRegression>
    >(&model_); model != nullptr && *model)
    {
        return (*model)->predict_proba(features);
    }
    if (const auto* model = std::get_if<
        std::shared_ptr<linear_model::RidgeClassifier>
    >(&model_); model != nullptr && *model)
    {
        return (*model)->predict_proba(features);
    }
    else
    {
        throw std::logic_error("predict_proba is unavailable for this model type");
    }
}

double Model::evaluate(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& targets
) const
{
    return std::visit([&](const auto& model) {
        if (!model) throw std::logic_error("Model handle is empty");
        return model->score(features, targets);
    }, model_);
}

linear_model::RegressionMetrics Model::regression_metrics(
    const Eigen::MatrixXd& features,
    const Eigen::VectorXd& targets
) const
{
    return std::visit([&](const auto& model) -> linear_model::RegressionMetrics {
        if (!model) throw std::logic_error("Model handle is empty");
        using ModelPointer = std::decay_t<decltype(model)>;
        if constexpr (std::is_same_v<ModelPointer,
            std::shared_ptr<linear_model::LinearRegression>>)
        {
            return model->evaluate(features, targets);
        }
        else if constexpr (std::is_same_v<ModelPointer,
            std::shared_ptr<linear_model::Ridge>>)
        {
            return {
                model->mean_squared_error(features, targets),
                model->mean_absolute_error(features, targets),
                model->r2_score(features, targets)
            };
        }
        else
        {
            throw std::logic_error(
                "regression_metrics is unavailable for this model type"
            );
        }
    }, model_);
}

void Model::save(const std::filesystem::path& path) const
{
    art::save(*this, path.string());
}

void save(
    const linear_model::LogisticRegression& model,
    const std::string& path
)
{
    write_model(model, path);
}

void save(
    const linear_model::LinearRegression& model,
    const std::string& path
)
{
    write_model(model, path);
}

void save(const linear_model::Ridge& model, const std::string& path)
{
    write_model(model, path);
}

void save(const linear_model::RidgeClassifier& model, const std::string& path)
{
    write_model(model, path);
}

void save(const Model& model, const std::string& path)
{
    std::visit([&](const auto& value) {
        if (!value) throw std::logic_error("Model handle is empty");
        write_model(*value, path);
    }, model.model_);
}

Model load(const std::string& path)
{
    std::ifstream input(path);
    if (!input.is_open())
    {
        throw std::runtime_error("Failed to open model file for reading: " + path);
    }

    std::string actual_signature;
    std::getline(input, actual_signature);
    if (actual_signature != signature)
    {
        throw std::runtime_error("Invalid ARTEMISML model signature");
    }
    const std::size_t version = read_field<std::size_t>(input, "format_version");
    if (version != format_version)
    {
        throw std::runtime_error("Unsupported ARTEMISML model format version");
    }
    expect_key(input, "type_name");
    std::string type_name;
    if (!(input >> std::quoted(type_name)))
    {
        throw std::runtime_error("Unsupported ARTEMISML model type");
    }
    if (read_field<int>(input, "fitted") != 1)
    {
        throw std::runtime_error("Serialized ARTEMISML model is not fitted");
    }

    if (type_name == "art::linear_model::LinearRegression")
    {
        const std::size_t feature_count =
            read_field<std::size_t>(input, "feature_count");
        const std::size_t max_iterations =
            read_field<std::size_t>(input, "max_iterations");
        const double tolerance = read_field<double>(input, "tolerance");
        const std::size_t parameter_count =
            read_field<std::size_t>(input, "parameter_count");
        if (feature_count == 0 || feature_count > maximum_dimension ||
            max_iterations == 0 || !std::isfinite(tolerance) || tolerance < 0.0 ||
            parameter_count != feature_count + 1 ||
            parameter_count > maximum_dimension)
        {
            throw std::runtime_error("Invalid LinearRegression model metadata");
        }
        expect_key(input, "parameters");
        Eigen::VectorXd parameters(
            static_cast<Eigen::Index>(parameter_count)
        );
        for (Eigen::Index index = 0; index < parameters.size(); ++index)
        {
            if (!(input >> parameters(index)) || !std::isfinite(parameters(index)))
            {
                throw std::runtime_error("Invalid LinearRegression parameter value");
            }
        }
        auto model = std::make_shared<linear_model::LinearRegression>();
        Model result(std::move(model));
        result.restore_linear_state(
            parameters, feature_count, max_iterations, tolerance
        );
        return result;
    }

    if (type_name == "art::linear_model::Ridge")
    {
        const std::size_t feature_count = read_field<std::size_t>(input, "feature_count");
        const double alpha = read_field<double>(input, "alpha");
        const std::size_t iterations = read_field<std::size_t>(input, "iterations");
        const int converged = read_field<int>(input, "converged");
        const std::size_t parameter_count = read_field<std::size_t>(input, "parameter_count");
        if (feature_count == 0 || parameter_count != feature_count + 1 ||
            parameter_count > maximum_dimension || !std::isfinite(alpha) || alpha < 0.0 ||
            iterations == 0 || (converged != 0 && converged != 1))
            throw std::runtime_error("Invalid Ridge model metadata");
        expect_key(input, "parameters");
        Eigen::VectorXd parameters(static_cast<Eigen::Index>(parameter_count));
        for (Eigen::Index index = 0; index < parameters.size(); ++index)
        {
            if (!(input >> parameters(index)) || !std::isfinite(parameters(index)))
                throw std::runtime_error("Invalid Ridge parameter value");
        }
        auto model = std::make_shared<linear_model::Ridge>();
        Model result(std::move(model));
        result.restore_ridge_state(
            parameters, feature_count, alpha, iterations, converged != 0
        );
        return result;
    }

    if (type_name == "art::linear_model::RidgeClassifier")
    {
        const std::size_t feature_count = read_field<std::size_t>(input, "feature_count");
        const std::size_t class_count = read_field<std::size_t>(input, "class_count");
        const int binary = read_field<int>(input, "binary");
        const double alpha = read_field<double>(input, "alpha");
        const std::size_t label_count = read_field<std::size_t>(input, "label_count");
        if (feature_count == 0 || class_count < 2 ||
            feature_count > maximum_dimension || class_count > maximum_dimension ||
            binary != (class_count == 2 ? 1 : 0) || label_count != class_count ||
            !std::isfinite(alpha) || alpha < 0.0)
            throw std::runtime_error("Invalid RidgeClassifier model metadata");
        std::vector<std::string> labels;
        labels.reserve(label_count);
        for (std::size_t index = 0; index < label_count; ++index)
        {
            expect_key(input, "label");
            std::string label;
            if (!(input >> std::quoted(label)))
                throw std::runtime_error("Invalid RidgeClassifier model label");
            labels.push_back(std::move(label));
        }
        const std::size_t rows = read_field<std::size_t>(input, "coefficient_rows");
        const std::size_t columns = read_field<std::size_t>(input, "coefficient_cols");
        if (rows != feature_count + 1 ||
            columns != (class_count == 2 ? 1 : class_count) ||
            rows > maximum_dimension || columns > maximum_dimension)
            throw std::runtime_error("Invalid RidgeClassifier coefficient dimensions");
        expect_key(input, "coefficients");
        Eigen::MatrixXd coefficients(
            static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)
        );
        for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
        {
            for (Eigen::Index column = 0; column < coefficients.cols(); ++column)
            {
                if (!(input >> coefficients(row, column)) ||
                    !std::isfinite(coefficients(row, column)))
                    throw std::runtime_error("Invalid RidgeClassifier coefficient value");
            }
        }
        auto model = std::make_shared<linear_model::RidgeClassifier>();
        Model result(std::move(model));
        result.restore_ridge_classifier_state(
            coefficients, feature_count, class_count, alpha, labels
        );
        return result;
    }

    if (type_name != "art::linear_model::LogisticRegression")
    {
        throw std::runtime_error("Unsupported ARTEMISML model type");
    }
    const std::size_t feature_count = read_field<std::size_t>(input, "feature_count");
    const std::size_t class_count = read_field<std::size_t>(input, "class_count");
    const int binary = read_field<int>(input, "binary");
    const double decision_threshold = read_field<double>(input, "decision_threshold");
    linear_model::LogisticRegressionOptions options;
    options.max_iterations = read_field<std::size_t>(input, "max_iterations");
    options.tolerance = read_field<double>(input, "tolerance");
    options.l2_penalty = read_field<double>(input, "l2_penalty");
    const std::size_t label_count = read_field<std::size_t>(input, "label_count");
    if (feature_count == 0 || class_count < 2 ||
        feature_count > maximum_dimension || class_count > maximum_dimension ||
        binary != (class_count == 2 ? 1 : 0) || label_count != class_count)
    {
        throw std::runtime_error("Invalid ARTEMISML model metadata");
    }
    std::vector<std::string> labels;
    labels.reserve(label_count);
    for (std::size_t index = 0; index < label_count; ++index)
    {
        expect_key(input, "label");
        std::string label;
        if (!(input >> std::quoted(label)))
        {
            throw std::runtime_error("Invalid ARTEMISML model label");
        }
        labels.push_back(std::move(label));
    }
    const std::size_t rows = read_field<std::size_t>(input, "coefficient_rows");
    const std::size_t columns = read_field<std::size_t>(input, "coefficient_cols");
    if (rows != feature_count + 1 ||
        columns != (class_count == 2 ? 1 : class_count) ||
        rows > maximum_dimension || columns > maximum_dimension)
    {
        throw std::runtime_error("Invalid ARTEMISML coefficient dimensions");
    }
    expect_key(input, "coefficients");
    Eigen::MatrixXd coefficients(
        static_cast<Eigen::Index>(rows), static_cast<Eigen::Index>(columns)
    );
    for (Eigen::Index row = 0; row < coefficients.rows(); ++row)
    {
        for (Eigen::Index column = 0; column < coefficients.cols(); ++column)
        {
            if (!(input >> coefficients(row, column)) ||
                !std::isfinite(coefficients(row, column)))
            {
                throw std::runtime_error("Invalid ARTEMISML coefficient value");
            }
        }
    }

    auto model = std::make_shared<linear_model::LogisticRegression>();
    Model result(std::move(model));
    result.restore_state(
        coefficients, feature_count, class_count, decision_threshold,
        options, labels
    );
    return result;
}
}
