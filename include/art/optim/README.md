# art::optim

## 职责

该层提供与具体模型解耦的优化协议：

- `OptimizationProblem)
- `GradientDescent)
- `AdaGrad)
- `Adam)
- `NewtonOptimizer)

一阶梯度下降：

[
\\theta_{t+1}=\\theta_t-\\eta g_t
]

AdaGrad 使用累计平方梯度：

[
G_t=G_{t-1}+g_t\\odot g_t,qquad
\\theta_{t+1}=\\theta_t-\\frac{\\eta}{\\sqrt{G_t}+\\epsilon}\\odot g_t
]

Adam 使用一阶矩和二阶矩的偏置修正。Newton 法使用：

[
\\theta_{t+1}=\\theta_t-H_t^{-1}g_t
]

优化器应统一处理维度、非有限值、停止条件和失败状态。

