# tests/metrics

`art::metrics` 当前没有生产 API，因此本目录暂不建立可执行测试目标。

待实现接口后，测试应覆盖 MSE、MAE、R2 和 accuracy 的正常值、空输入、样本数/维度错误、非有限值、回归标签与分类标签语义，并验证 lower-is-better 或 higher-is-better 方向。

当前线性模型的 `RegressionMetrics`/`ClassificationMetrics` 测试属于 `tests/linear_model`，不应被误记为独立 `art::metrics` 已实现。
