/*
 *    ###   ####   #####  #####  #   #   ###   ####   #   #  #
 *   #   #  #   #    #    #     ## ##  #   #  #      ## ##  #
 *   #####  ####     #    ###   # # #  #####  ###    # # #  #
 *   #   #  #  #     #    #     #   #  #   #  #      #   #  #
 *   #   #  #   #  #####  #####  #   #  #   #  ####   #   #  #####
 */
#pragma once

#include <Eigen/Dense>

#include <cmath>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace art::core::autodiff
{
namespace detail
{
    struct Node
    {
        double value = 0.0;
        std::shared_ptr<Node> left;
        std::shared_ptr<Node> right;
        double left_partial = 0.0;
        double right_partial = 0.0;
    };
}

class Variable
{
public:
    Variable() = default;

    explicit Variable(double value)
        : node_(std::make_shared<detail::Node>())
    {
        node_->value = value;
    }

    double value() const
    {
        ensureValid();
        return node_->value;
    }

    bool valid() const noexcept
    {
        return static_cast<bool>(node_);
    }

private:
    explicit Variable(std::shared_ptr<detail::Node> node)
        : node_(std::move(node))
    {
    }

    void ensureValid() const
    {
        if (!node_)
        {
            throw std::logic_error("art::core::autodiff: invalid variable");
        }
    }

    std::shared_ptr<detail::Node> node_;

    friend class GradientTape;
    friend Variable makeBinary(
        const Variable&,
        const Variable&,
        double,
        double,
        double
    );
    friend Variable operator+(const Variable&, const Variable&);
    friend Variable operator-(const Variable&, const Variable&);
    friend Variable operator*(const Variable&, const Variable&);
    friend Variable operator/(const Variable&, const Variable&);
    friend Variable exp(const Variable&);
    friend Variable log(const Variable&);
};

inline Variable makeBinary(
    const Variable& left,
    const Variable& right,
    double value,
    double left_partial,
    double right_partial
)
{
    if (!left.valid() || !right.valid())
    {
        throw std::invalid_argument("art::core::autodiff: binary operation needs valid variables");
    }

    auto node = std::make_shared<detail::Node>();
    node->value = value;
    node->left = left.node_;
    node->right = right.node_;
    node->left_partial = left_partial;
    node->right_partial = right_partial;
    return Variable(std::move(node));
}

inline Variable operator+(const Variable& left, const Variable& right)
{
    return makeBinary(left, right, left.value() + right.value(), 1.0, 1.0);
}

inline Variable operator-(const Variable& left, const Variable& right)
{
    return makeBinary(left, right, left.value() - right.value(), 1.0, -1.0);
}

inline Variable operator*(const Variable& left, const Variable& right)
{
    return makeBinary(
        left,
        right,
        left.value() * right.value(),
        right.value(),
        left.value()
    );
}

inline Variable operator/(const Variable& left, const Variable& right)
{
    if (right.value() == 0.0)
    {
        throw std::domain_error("art::core::autodiff: division by zero");
    }

    const double denominator = right.value() * right.value();
    return makeBinary(
        left,
        right,
        left.value() / right.value(),
        1.0 / right.value(),
        -left.value() / denominator
    );
}

inline Variable exp(const Variable& input)
{
    if (!input.valid())
    {
        throw std::invalid_argument("art::core::autodiff: exp needs a valid variable");
    }

    const double value = std::exp(input.value());
    return makeBinary(input, Variable(0.0), value, value, 0.0);
}

inline Variable log(const Variable& input)
{
    if (!input.valid())
    {
        throw std::invalid_argument("art::core::autodiff: log needs a valid variable");
    }
    if (input.value() <= 0.0)
    {
        throw std::domain_error("art::core::autodiff: log domain must be positive");
    }

    return makeBinary(input, Variable(0.0), std::log(input.value()), 1.0 / input.value(), 0.0);
}

using Gradient = Eigen::VectorXd;

class GradientTape
{
public:
    Variable variable(double value) const
    {
        return Variable(value);
    }

    Gradient gradient(
        const Variable& output,
        const std::vector<Variable>& inputs
    ) const
    {
        if (!output.valid())
        {
            throw std::invalid_argument("art::core::autodiff: output must be valid");
        }

        std::vector<std::shared_ptr<detail::Node>> topology;
        std::unordered_set<detail::Node*> visited;
        collect(output.node_, visited, topology);

        std::unordered_map<detail::Node*, double> adjoints;
        adjoints[output.node_.get()] = 1.0;
        for (auto iterator = topology.rbegin(); iterator != topology.rend(); ++iterator)
        {
            const auto& node = *iterator;
            const double adjoint = adjoints[node.get()];
            if (node->left)
            {
                adjoints[node->left.get()] += adjoint * node->left_partial;
            }
            if (node->right)
            {
                adjoints[node->right.get()] += adjoint * node->right_partial;
            }
        }

        Gradient result(static_cast<Eigen::Index>(inputs.size()));
        for (Eigen::Index index = 0; index < result.size(); ++index)
        {
            if (!inputs[static_cast<std::size_t>(index)].valid())
            {
                throw std::invalid_argument("art::core::autodiff: input must be valid");
            }

            const auto node = inputs[static_cast<std::size_t>(index)].node_.get();
            const auto found = adjoints.find(node);
            result(index) = found == adjoints.end() ? 0.0 : found->second;
        }
        return result;
    }

private:
    static void collect(
        const std::shared_ptr<detail::Node>& node,
        std::unordered_set<detail::Node*>& visited,
        std::vector<std::shared_ptr<detail::Node>>& topology
    )
    {
        if (!node || !visited.insert(node.get()).second)
        {
            return;
        }
        collect(node->left, visited, topology);
        collect(node->right, visited, topology);
        topology.push_back(node);
    }
};
}
