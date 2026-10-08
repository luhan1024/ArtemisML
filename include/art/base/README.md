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
- 新增模型的 `fit` 应采用失败原子性：训练失败后模型处于未训练状态，不暴露半成品参数。
- 重复 fit 必须覆盖旧模型状态，而不是叠加旧参数。
- 输入样本数、特征数和标签形状必须明确检查。

当前历史实现可能在训练开始时清除已有状态；不得假设失败的重复 fit 会保留旧模型。若要
变更为“失败后保留上一次成功模型”，须作为单独的公共行为变更提出，并为所有估计器统一
设计强异常保证及测试，不得由单个模型自行改变。

新增数值监督模型默认按样本行、特征列解释 `X`；回归/单标签目标使用长度为样本数的向量，
矩阵目标必须在接口中明确其类别轴和编码。空输入、维度不匹配和非有限值行为应在公共头文件
或本层 README 中注明，具体错误类型遵守公共 API 总览的错误契约。

该层只定义协议，不实现具体算法。

## 新增模型必须遵守的协议

新增监督模型至少应暴露 `fit(X, y)`，并通过 `using base::Regressor::fit` 或
`using base::Classifier::fit` 保留基类入口。需要显式优化器时增加
`fit(X, y, const optim::Optimizer&)`，但不得改变 base 层基本协议。

模型状态必须满足：构造后未训练；成功 fit 后可 predict/evaluate/score；fit 失败后处于
未训练状态且不得暴露半成品参数；重复 fit 不叠加旧参数。公共头文件必须说明参数、
输入输出形状、未训练行为和异常类型。

