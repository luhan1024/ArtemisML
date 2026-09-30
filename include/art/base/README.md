# art::base

## 职责

`base` 定义所有估计器的公共协议：

- `Estimator)：可训练对象。
- `Transformer)：训练并转换数据。
- `Predictor)：训练并预测。
- `Regressor)：回归模型。
- `Classifier)：分类模型。

核心状态约束：

- 未 fit 的对象不得 predict、transform 或 score。
- fit 失败后对象必须保持未训练状态。
- 重复 fit 必须覆盖旧模型状态，而不是叠加旧参数。
- 输入样本数、特征数和标签形状必须明确检查。

该层只定义协议，不实现具体算法。

