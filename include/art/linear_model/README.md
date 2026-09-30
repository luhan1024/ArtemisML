# art::linear_model

## 职责

该层实现线性模型：

- `LinearRegression`
- `Ridge`
- `LogisticRegression`

`LogisticRegression` 当前支持：

- 二分类 logits 与 Binary Cross Entropy；
- 多分类 logits 与 Softmax Cross Entropy；
- 梯度下降训练、停止条件和截距项；
- `predict_proba`、`predict`、`score`；
- `loss_history` 用于查看每一步梯度下降的目标函数值；
- 二分类系数矩阵为 `(feature_count + 1) × 1`，多分类系数矩阵为
  `(feature_count + 1) × class_count`。

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

