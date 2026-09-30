# art::loss

## 公共 API

`art::loss` 提供以 Eigen 矩阵为输入的损失函数协议：

- `LossFunction`：统一的 `value(prediction, target)` 和 `gradient(prediction, target)` 接口。
- `Reduction::Mean` 与 `Reduction::Sum`：控制 value 和 gradient 的共同缩放。
- `MeanSquaredError`、`MeanAbsoluteError`。
- `BinaryCrossEntropy`：输入 logits，目标值在 `[0, 1]`。
- `SoftmaxCrossEntropy`：输入 `N x C` logits，目标是 `N x C` one-hot 矩阵。
- `SparseCrossEntropy`：输入 `N x C` logits，目标是 `N` 个整数类别索引，首选 `Eigen::VectorXi`，也支持 `N x 1` 整数矩阵。

`gradient()` 始终返回对 prediction/logits 的导数；target 被视为常量，不提供目标梯度。

## 公式与缩放

回归损失和 BCE 的 Mean 按 prediction 中的元素数取平均。Softmax Cross Entropy 和 Sparse Cross Entropy 的 Mean 按样本行数取平均。Sum 不做缩放，且 value 与 gradient 使用完全相同的缩放因子。

对元素数为 `K` 的输入，MSE 为 `sum((prediction - target)^2) / K`，MAE 为
`sum(abs(prediction - target)) / K`；Sum 版本去掉分母。MAE 在 prediction 等于
target 的位置返回 0 作为次梯度。

Softmax Cross Entropy 使用稳定的 log-sum-exp，未缩放梯度为 `softmax(logits) - one_hot_target`。BCE 使用稳定 logits 公式 `max(z, 0) - z*y + log(1 + exp(-abs(z)))`，未缩放梯度为 `sigmoid(z) - y`。

所有实现都会检查空输入、形状、非有限值，以及 one-hot 或类别索引的合法性。
