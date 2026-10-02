# src/optim

该目录实现 `OptimizationProblem`、`GradientDescent` 和 `NewtonOptimizer`。实现提供单步计算与数值/维度检查，训练循环、初始参数和收敛判断由调用方模型负责。

当前不存在 AdaGrad、Adam 或自动 line-search 的实现；相关扩展只能标记为计划，不能由 README 暗示为已完成。

本轮只更新文档，未执行 WSL 配置、编译、CTest 或运行命令。
