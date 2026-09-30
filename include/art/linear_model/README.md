# art::linear_model

## 职责

该层实现线性模型：

- `LinearRegression`
- `Ridge`
- `RidgeClassifier`
- `LogisticRegression`

`LogisticRegression` 当前支持：

- 二分类 logits 与 Binary Cross Entropy；
- 多分类 logits 与 Softmax Cross Entropy；
- 梯度下降训练、停止条件和截距项；
- `predict_proba`、`predict`、`score`；
- `loss_history` 用于查看每一步梯度下降的目标函数值；
- `training_history` 记录每一步的 objective、gradient norm 和 step norm；
- 可通过 `fit(X, y, optimizer)` 显式选择 `GradientDescent` 或 `NewtonOptimizer`；
- Newton 当前仅支持二分类，多分类选择 Newton 会明确拒绝；
- 支持不惩罚截距项的 L2 正则化；
- 二分类系数矩阵为 `(feature_count + 1) × 1`，多分类系数矩阵为
  `(feature_count + 1) × class_count`。

`GradientDescent(learning_rate)` 将学习率保存在优化器对象中；
`NewtonOptimizer` 不使用学习率。旧的 `fit(X, y)` 调用默认使用梯度下降。
`LogisticRegressionOptions` 只保存最大迭代次数、停止容差和 L2 正则化系数，
不重复保存优化器学习率。

`RidgeClassifier` 复用平方损失和 L2 正则化，支持二分类 0/1、-1/1 目标，以及
整数类别和 one-hot 多分类目标；其分类输出取线性响应的最大类别。

模型可以通过 `art::save` 保存、通过 `art::load` 恢复。LinearRegression、Ridge、
RidgeClassifier 和 LogisticRegression 均支持 round-trip。持久化包含系数、截距列约定、
类别数量或回归特征数、类别标签映射、决策阈值和必要配置，不包含训练数据、临时梯度
或优化器指针。LinearRegression 的 `evaluate` 提供 MSE、MAE 和 R2。

也可以直接调用 `model.save(std::filesystem::path{"model.artemisml"})`；加载后的
`art::Model` 同样支持 `save(path)`，并保留 `predict`、`predict_proba` 和 `evaluate`。

线性回归最小二乘目标为：

[
J(\\beta)=\\frac{1}{2n}\\|X\\beta-y\\|_2^2
]

其梯度和 Hessian 为：

[
\\nabla J=\\frac{1}{n}X^T(X\\beta-y),\\qquad
H=\\frac{1}{n}X^TX
]

Ridge 在目标中增加 (\\lambda\\|\\beta\\|_2^2/2)。模型可以使用手写导数，也可以接入 `core::autodiff`，两者必须通过一致性测试。

