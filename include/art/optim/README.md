# art::optim

## 当前公共 API

该层提供与具体模型解耦的 `OptimizationProblem`、`OptimizerOptions`、`OptimizerState`、`OptimizationStepResult`、`OptimizationResult`、`GradientDescent` 和 `NewtonOptimizer`。优化器只执行一次 step；模型负责训练循环、停止判断和结果汇总。

梯度下降使用 `theta_next = theta - eta * gradient`。默认构造的 `GradientDescent` 从 `OptimizerOptions::learning_rate` 读取学习率；`GradientDescent(learning_rate)` 使用构造时保存的学习率。学习率必须为有限正数。

Newton 使用完整方向 `theta_next = theta - H^{-1} gradient`，不使用 `learning_rate` 对步长阻尼；问题必须提供合法且维度匹配的 Hessian。

两种优化器均检查非空且有限的参数、目标值、梯度和候选参数，并检查梯度/Hessian 维度。AdaGrad、Adam 和 line-search 当前没有实际公共头文件或实现，属于计划/待实现功能。

## 新增优化器规范

优化器只负责一次参数更新，不负责 epoch/while 循环、数据读取、模型状态或持久化。
新增优化器应继承 `Optimizer`，实现单步 `step(problem, parameters, state)`，并明确所需
的 gradient/Hessian/额外状态、维度要求、更新公式、失败条件和参数所有权。

学习率属于具体优化器实例，例如 `GradientDescent(0.001)`；不得在模型和
`OptimizerOptions` 中重复定义同一个有效学习率。模型负责调用 step、记录历史和判断
停止条件。新增优化器必须在 `tests/optim` 验证一步更新、边界参数、数值错误和状态行为。

