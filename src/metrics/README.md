# src/metrics

该目录目前没有 metrics 生产实现。`mean_squared_error`、`mean_absolute_error`、`r2_score` 和 `accuracy_score` 的独立接口仍处于计划阶段。

未来实现必须与具体模型解耦，明确预测输出、标签类型、样本数、空输入、非有限值和指标方向，并与 `art::linear_model` 当前的聚合评价结构区分开。完成公共头文件和测试后，才可加入构建目标。

本轮只更新文档，未执行 WSL 配置、编译、CTest 或运行命令。
