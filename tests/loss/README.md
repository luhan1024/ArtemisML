# tests/loss

损失层测试覆盖：

- 五种损失的已知数值、梯度形状和 Mean/Sum 缩放。
- MSE 的有限差分解析梯度检查。
- BCE logits 稳定性和 sigmoid-logit 梯度。
- Softmax 的稳定 log-sum-exp、one-hot 合法性和梯度公式。
- Sparse Cross Entropy 的整数标签、类别范围和 one-hot 等价数值。
- 形状错误、空输入、非有限输入、非法 one-hot 与非法类别索引。

当前 `art::core::autodiff::GradientTape` 只接受标量 `Variable` 图并返回标量输出对输入的梯度，不能直接记录 Eigen 矩阵 loss。测试因此使用有限差分核对矩阵解析梯度，并将该标量自动求导边界记录在文档中；待 autodiff 支持矩阵/Jacobian 后再补充直接 GradientTape 一致性测试。
