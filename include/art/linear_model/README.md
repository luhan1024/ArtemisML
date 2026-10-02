# art::linear_model

## 当前职责与统一 fit 协议

该层实现线性回归、正则化回归和线性分类模型。所有模型遵循 `art::base` 状态协议：`fit(X, y)` 是统一数值训练入口，未 fit 时不得预测或评分，fit 失败后保持未训练状态，重复 fit 重新建立状态。`X` 为 `Eigen::MatrixXd`，标签为 `Eigen::VectorXd`，样本数和特征数必须匹配。

显式优化器入口包括 `LinearRegression::fit(X, y, Optimizer)`、`Ridge::fit(X, y, Optimizer)`、
`RidgeClassifier::fit(X, y, Optimizer)`、`RidgeClassifier::fit(X, one_hot, Optimizer)`、
`Lasso::fit(X, y, Optimizer)`、`ElasticNet::fit(X, y, Optimizer)` 和
`LogisticRegression::fit(X, y, Optimizer)`；原有 Dataset + Optimizer + Options 入口保持兼容。
默认 `fit(X, y)` 分别使用 LinearRegression 的 GradientDescent、Ridge/RidgeClassifier
的 NewtonOptimizer、Lasso/ElasticNet/LogisticRegression 的 GradientDescent。模型不解析
CSV 或文本标签。

## 模型与评价

- `LinearRegression` 使用带截距列的最小二乘目标，支持 GradientDescent 和 NewtonOptimizer。
- `Ridge` 使用不惩罚截距的 L2 正则化；`Lasso` 和 `ElasticNet` 分别使用 L1 与 L1/L2 组合目标。
- `RidgeClassifier` 使用平方损失和 L2 正则化，接受 0/1、-1/1 数值标签或 one-hot 矩阵。
- `LogisticRegression` 支持二分类和多分类 logits、概率预测、L2 正则化、决策阈值和训练历史；
  二分类接受 0/1 与 -1/1，Newton 当前仅支持二分类。

回归模型提供 MSE、MAE、R2 和 `RegressionMetrics`；分类模型提供 `ClassificationMetrics`；`art::Model::evaluate` 返回二者的 variant。独立 `art::metrics` 层目前没有公共头文件或实现，其函数只能记为计划/待实现，不能与模型专用评价接口混同。

## 标签编码

模型接收数值标签，不直接训练字符串标签。字符串标签由 `preprocessing::LabelEncoder` 按训练数据首次出现顺序建立词典：`Binary01` 为 `0/1`，`BinarySigned` 为 `-1/1`，`SignedOrdinal` 为 `-1/0/1`，`Ordinal` 为 `0,1,...`，`OneHot` 按词典顺序输出列。未知标签默认报错，也可配置为忽略。

分类模型训练后默认保存 `"0"`、`"1"` 等字符串化类别索引，可通过 `set_class_labels` 替换外部标签映射。标签编码不从测试数据学习新类别。

## 保存与加载

当前支持 `LogisticRegression`、`LinearRegression`、`Ridge`、`RidgeClassifier`、`Lasso` 和 `ElasticNet` 通过 `art::save(model, path)`、成员式 `model.save(path)` 和 `art::load(path)` 完成 round-trip。持久化包含模型类型、参数、特征数、必要配置、类别信息和决策阈值，不包含训练数据、临时梯度或优化器指针。加载结果为 `art::Model`，可继续预测、评价和保存。

## 完成度边界

上述模型、评价聚合结构、统一显式优化器入口和持久化入口均已有代码。独立 metrics 函数、
自动从字符串标签训练、自动选择优化器以及未出现在公共头文件中的模型仍为计划/待实现功能。

## 新增线性模型模板

新增线性模型应按以下顺序实现：

1. 在本层头文件声明模型类，继承 `base::Regressor` 或 `base::Classifier`。
2. 使用 `using base::<type>::fit` 保留 `fit(X, y)`，并增加显式 `fit(X, y, const optim::Optimizer&)`。
3. 将目标函数和梯度/Hessian 放入 `OptimizationProblem`；训练循环和停止条件留在模型。
4. 用 `predict`、`evaluate`、`score` 暴露用户结果，评价结构必须与任务类型一致。
5. 保存参数、超参数、特征数、类别信息和预测所需配置，不保存训练数据或优化器指针。
6. 在 `tests/linear_model` 覆盖训练、重复训练、未训练错误、维度错误、标签语义和 round-trip。

```cpp
class NewLinearModel final : public art::base::Regressor {
public:
    using art::base::Regressor::fit;
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
             const art::optim::Optimizer& optimizer);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    RegressionMetrics evaluate(const Eigen::MatrixXd& X,
                               const Eigen::VectorXd& y) const;
    void save(const std::filesystem::path& path) const;
};
```

