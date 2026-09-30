# art::preprocessing

## 职责

该层提供类似 sklearn 的可训练转换器：

- `StandardScaler)
- `MinMaxScaler)
- `Imputer)
- `OneHotEncoder)
- `ColumnTransformer)
- 特征选择和特征生成器

转换器必须区分 `fit)、`transform` 和 `fit_transform)。训练阶段保存统计量，例如标准化：

[
z_{ij}=\\frac{x_{ij}-\\mu_j}{\\sigma_j}
]

预测阶段只能使用训练阶段保存的参数，不能重新计算数据集统计量。

