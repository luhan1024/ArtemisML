# art::pipeline

## 职责

`pipeline` 将多个 Transformer 和最终 Estimator 组成可复用的数据流。

顺序 Pipeline 的基本流程是：

[
X \\xrightarrow{T_1} X_1
\\xrightarrow{T_2} X_2
\\xrightarrow{Estimator} \\hat{y}
]

Pipeline 必须保证：

- fit 时按顺序训练并传递数据。
- predict/score 时只调用 transform，不重复训练。
- 失败时清理 fitted 状态。
- 步骤名称唯一、类型合法、样本数一致。

`FeatureUnion` 用于并行变换后按列拼接特征。

