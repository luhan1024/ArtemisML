# art::metrics

## 职责

该层提供与模型实现无关的评估函数：

- `mean_squared_error)
- `mean_absolute_error)
- `r2_score)
- `accuracy_score)

例如均方误差：

[
MSE=\\frac{1}{n}\\sum_{i=1}^{n}(y_i-\\hat y_i)^2
]

指标函数只计算结果，不修改模型状态；输入长度、标签类型和空样本行为必须明确规定。

