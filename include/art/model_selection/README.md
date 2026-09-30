# art::model_selection

## 职责

该层负责与具体模型、损失函数和数据格式无关的数据划分、交叉验证和参数搜索：

- `train_test_split`：按行产生互不重叠的训练集和测试集。
- `cross_validate`：按 fold 重新创建估计器、只在当前训练折上 fit，再对验证折评分。
- `GridSearch`：枚举参数笛卡尔积，记录每组参数的评分或失败原因，并选择最优结果。

接口只依赖 `art::base` 的矩阵、向量和估计器协议，不依赖 `metrics` 或具体模型。评分器是回调，因此 higher-is-better/lower-is-better 由调用方或 `GridSearchOptions::greater_is_better` 明确指定。

## 数据泄漏与生命周期

`cross_validate` 和 `GridSearch` 都要求估计器工厂。每个 fold、每个参数组合都会调用工厂获得新的对象；因此已有 fitted 状态和预处理统计量不会跨 fold 或参数组合复用。fit/predict 回调接收当前训练特征、训练标签和验证特征，Pipeline 等非 `base::Predictor` 对象可通过适配器接入，而不需要 model_selection 反向依赖 pipeline。

随机种子通过 `std::optional<std::uint64_t>` 传入。提供种子时，行置换和 fold 划分可复现；不提供时使用随机设备。`train_test_split` 要求 `test_size` 位于 `(0, 1)`，并保证训练集和测试集均非空；交叉验证要求 `2 <= folds <= sample_count`。

空输入、样本数不一致、无效比例/折数、空回调、空参数名或空参数值列表均抛出 `std::invalid_argument`。评分器返回非有限值也会被拒绝。GridSearch 的失败组合保留在 `GridSearchTrial::error` 中并继续搜索；若所有组合失败，`best_index` 为空。

## 完成度

已实现：`train_test_split`、回调式 `cross_validate`、Predictor 便捷版 `cross_validate` 和 `GridSearch`。

未实现：具体 metrics、损失函数、模型 clone 协议和自动拟合最佳模型。上述功能属于其他层或尚未由公共协议提供。

