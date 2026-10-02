# art::base

## 职责

`base` 定义所有估计器的公共协议：

- `Estimator)：可训练对象。
- `Transformer)：训练并转换数据。
- `Predictor)：训练并预测。
- `Regressor)：回归模型。
- `Classifier)：分类模型。

## 训练接口

`Estimator::fit` 使用统一的 `FeatureInput`/`TargetInput` 类型。`TargetInput`
默认为空向量，用于保留无监督 Transformer 的 `fit(X)` 兼容调用；监督模型
使用 `fit(X, y)`。

`Transformer` 和 `Predictor` 对 `fit` 做最终覆写，并将具体算法转发到受保护的
`do_fit`。因此具体模型通过 `using base::Regressor::fit` 或
`using base::Classifier::fit` 可以保留统一的 `fit(X, y)` 主接口，同时继续
提供自己的低层或优化器重载。

优化器重载不属于 `Estimator` 的强制协议：并非所有估计器（尤其是
Transformer）都由通用优化器训练。需要显式控制优化器的具体模型应提供
`fit(X, y, const optim::Optimizer&)`，而不改变 base 层的基本继承关系。

## 状态约束

- 未 fit 的对象不得 predict、transform 或 score。
- fit 失败后对象必须保持未训练状态。
- 重复 fit 必须覆盖旧模型状态，而不是叠加旧参数。
- 输入样本数、特征数和标签形状必须明确检查。

该层只定义协议，不实现具体算法。

## 新增模型必须遵守的协议

新增监督模型至少应暴露 `fit(X, y)`，并通过 `using base::Regressor::fit` 或
`using base::Classifier::fit` 保留基类入口。需要显式优化器时增加
`fit(X, y, const optim::Optimizer&)`，但不得改变 base 层基本协议。

模型状态必须满足：构造后未训练；成功 fit 后可 predict/evaluate/score；fit 失败后保持
原有有效状态或明确回到未训练状态；重复 fit 不叠加旧参数。公共头文件必须说明参数、
输入输出形状、未训练行为和异常类型。

