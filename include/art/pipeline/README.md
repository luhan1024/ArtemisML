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

## 新增 pipeline 组件规范

新增步骤必须实现已有 Transformer/Estimator 协议，不得在 Pipeline 内部复制模型训练逻辑。
组件必须定义 fit 时的数据流、predict 时的只读流、失败后的状态回滚、输入输出列数和
步骤命名规则；在 `tests/pipeline` 覆盖单步、组合、重复 fit、失败回滚和数据泄漏边界。

