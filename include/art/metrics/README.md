# art::metrics

## 当前状态

该目录预留与模型实现无关的评价函数层，计划提供 `mean_squared_error`（lower-is-better）、`mean_absolute_error`（lower-is-better）、`r2_score`（通常 higher-is-better）和 `accuracy_score`（higher-is-better）。

截至当前代码检查，`include/art/metrics` 没有公共头文件，`src/metrics` 没有实现，`tests/metrics` 只有说明文件。因此这些函数、输入维度检查、空输入策略、非有限值策略以及回归/分类标签类型均属于计划/待实现规范，不是当前可调用 API。

当前线性模型使用 `art::linear_model::RegressionMetrics` 和 `ClassificationMetrics`，它们不是本层独立函数的替代实现。

## 新增评估函数规范

新增 metric 应提供纯函数式接口，不保存模型状态，不修改输入，不依赖具体模型。接口
必须说明 higher-is-better 或 lower-is-better、输入形状、标签编码、空输入、维度错误
和非有限值策略。新增函数应在 `tests/metrics` 覆盖已知数值、边界值和错误输入，并由
模型的 `evaluate` 通过明确适配层调用，避免复制同一公式。

