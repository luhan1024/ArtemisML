# tests/model_selection

该目录验证 model_selection 层公共接口的行为、边界条件和错误语义。

当前测试覆盖：

- `train_test_split` 的比例、空输入、样本数不一致、固定种子和训练/测试不重叠；
- `cross_validate` 的折数检查、固定种子、每个 fold 创建独立估计器和评分结果；
- `GridSearch` 的参数笛卡尔积、最佳参数选择、lower-is-better 选择、失败组合记录和状态隔离。

测试使用本地最小 `base::Predictor`，不依赖 metrics、具体线性模型或 CSV 文件；因此可以单独验证 model_selection 的生命周期与泄漏防护。

对应设计说明：`include/art/model_selection/README.md`。
