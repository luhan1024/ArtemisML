# src/loss

该目录实现 `include/art/loss/loss.h` 声明的损失函数，不依赖具体模型、优化器或数据格式。实现边界包括：

- 统一 reduction 语义和输入检查。
- MSE、MAE、logits 形式 BCE。
- 直接对 logits 计算的稳定 Softmax Cross Entropy。
- one-hot 与整数类别索引两种多分类目标表示。

Softmax 与 BCE 的数值稳定处理在本层完成；本目录不实现模型参数梯度、不扩展 autodiff 核心，也不实现 AdaGrad、Adam、FocalLoss 或其他未列入第一阶段的损失。
