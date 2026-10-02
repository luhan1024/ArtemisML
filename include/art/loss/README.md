# art::loss

## 公共 API

`art::loss` 提供以 Eigen 矩阵为输入的损失函数协议：

- `LossFunction`：统一的 `value(prediction, target)` 和 `gradient(prediction, target)` 接口。
- `Reduction::Mean` 与 `Reduction::Sum`：控制 value 和 gradient 的共同缩放。
- `MeanSquaredError`、`MeanAbsoluteError`。
- `BinaryCrossEntropy`：输入 logits，目标值在 `[0, 1]`。
- `SoftmaxCrossEntropy`：输入 `N x C` logits，目标是 `N x C` one-hot 矩阵。
- `SparseCrossEntropy`：输入 `N x C` logits，目标是 `N` 个整数类别索引，首选 `Eigen::VectorXi`，也支持 `N x 1` 整数矩阵。

所有 `value()` 返回标量，`gradient()` 返回与 prediction/logits 同形状的矩阵。

Ridge 与 RidgeClassifier 使用现有 `MeanSquaredError` 的平方损失语义，并在模型层
增加不惩罚截距项的 L2 正则化；不新增重复的分类平方损失。

Lasso 和 ElasticNet 同样复用该平方损失，并在模型层增加 L1/L2 组合正则项。

`gradient()` 始终返回对 prediction/logits 的导数；target 被视为常量，不提供目标梯度。

## 公式与缩放

回归损失和 BCE 的 Mean 按 prediction 中的元素数取平均。Softmax Cross Entropy 和 Sparse Cross Entropy 的 Mean 按样本行数取平均。Sum 不做缩放，且 value 与 gradient 使用完全相同的缩放因子。

对元素数为 `K` 的输入，MSE 为 `sum((prediction - target)^2) / K`，MAE 为
`sum(abs(prediction - target)) / K`；Sum 版本去掉分母。MAE 在 prediction 等于
target 的位置返回 0 作为次梯度。

Softmax Cross Entropy 使用稳定的 log-sum-exp，未缩放梯度为 `softmax(logits) - one_hot_target`。Sparse Cross Entropy 对样本 `i` 使用 `logsumexp(z_i) - z_{i,y_i}`，其未缩放梯度同样为 softmax 概率减去对应的 one-hot 标签。BCE 使用稳定 logits 公式 `max(z, 0) - z*y + log(1 + exp(-abs(z)))`，未缩放梯度为 `sigmoid(z) - y`。

所有实现都会检查空输入、形状、非有限值，以及 one-hot 或类别索引的合法性。

## 新增损失函数规范

新增损失必须放在 `art::loss`，优先继承 `LossFunction`，不得把损失公式直接嵌入具体模型。

```cpp
class NewLoss final : public art::loss::LossFunction {
public:
    double value(const Eigen::MatrixXd& prediction,
                 const Eigen::MatrixXd& target) const override;
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& prediction,
                             const Eigen::MatrixXd& target) const override;
};
```

交付清单：说明 prediction 是值、概率还是 logits，target 是连续值、类别索引还是 one-hot；
明确 Mean/Sum 缩放并保持 value 与 gradient 一致；检查空输入、形状、有限值和超参数；为
不可导点定义次梯度；在 `tests/loss` 用有限差分验证解析梯度；同步更新 include/src/tests
三处 README；由模型层显式组合该 loss，不得反向依赖具体模型。
