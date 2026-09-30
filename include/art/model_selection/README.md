# art::model_selection

## 职责

该层负责模型评估和参数选择：

- `train_test_split)
- `cross_validate)
- `GridSearch)

数据划分必须可复现，可通过随机种子控制。交叉验证必须在每个 fold 中独立训练模型，避免训练集统计量泄漏到验证集。参数搜索应记录参数组合、评分和最佳模型，而不是只返回一个未解释的结果。

