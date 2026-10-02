# tests/optim

该目录验证当前已实现的 `OptimizationProblem`、`GradientDescent`、`NewtonOptimizer`、`OptimizerOptions`、`OptimizerState` 和单步结果协议。

测试应覆盖梯度下降和 Newton 更新、显式/旧式学习率语义、状态步数、停止参数、空参数、维度不匹配和非有限值。AdaGrad、Adam 和 line-search 尚未实现，不建立伪测试目标。

对应设计说明：`include/art/optim/README.md`。
