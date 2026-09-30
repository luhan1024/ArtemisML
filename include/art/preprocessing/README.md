# art::preprocessing

## 职责与输入边界

该层提供基于 `Eigen::MatrixXd` 的、类似 scikit-learn 的可训练转换器。
当前 `base::Transformer` 协议只接受数值矩阵，因此分类特征在进入
`OneHotEncoder` 前必须已经表示为有限数值；字符串 `DataFrame` 转换仍属于
`data`/`io` 层，不在本层读取 CSV 或解析字符串。

所有转换器都遵循：`fit` 只从训练矩阵学习状态，`transform` 只使用已保存状态，
`fit_transform` 等价于一次 `fit` 后对同一输入执行一次 `transform`。未 fit、空输入、
维度不匹配和不支持的非有限值会抛出明确异常；行数保持不变，列数由转换器决定。

## 已实现 API

- `StandardScaler`：按列保存均值和总体标准差，使用
  `z_{ij}=(x_{ij}-\mu_j)/s_j`；常数列的尺度取 `1`。
- `MinMaxScaler`：按列保存训练集最小值和最大值，映射到配置的 `[lower, upper]`；
  常数列映射为 `lower`。
- `Imputer`：`Mean` 忽略 `NaN` 计算列均值，`Constant` 使用固定填充值；只有
  `NaN` 被视为缺失，正负无穷会被拒绝。均值策略的全缺失列拒绝 fit。
- `OneHotEncoder`：按输入列独立学习升序类别，输出列顺序为输入列顺序再接该列的
  类别升序；未知类别可选择 `Error` 或 `Ignore`（后者输出全零块）。当前类别为有限
  数值，类别查找使用精确 `double` 相等，不支持字符串或自动排序混合类型。
- `ColumnTransformer`：按不重叠的列索引选择输入，按 specification 声明顺序 fit、
  transform 并横向拼接；未声明列丢弃，子转换器必须保持样本数。
- `SelectColumns`：按固定列索引选择并重排特征。
- `PolynomialFeatures`：生成包含偏置项（可选）及一阶到指定阶数的、有序组合重复单项式。
- `LabelEncoder`：独立于数值特征编码器处理 `std::vector<std::string>` 标签。支持
  `Binary01`、`BinarySigned`、`SignedOrdinal`、`Ordinal`、`OneHot` 和 `Auto`；
  `Auto` 对二分类选择 `Binary01`，对三分类及以上选择 `OneHot`，单类别退化为
  `Ordinal`。

## 文本标签编码语义

`LabelEncoder` 在 `fit` 阶段按训练数据首次出现顺序建立稳定类别词典，后续
`transform` 不会从测试标签添加类别。未知标签默认抛出异常，也可以配置为 `Ignore`：
向量输出写入 `NaN`，OneHot 输出对应行全零。

- `Binary01`：第一个训练类别为 `0`，第二个为 `1`。
- `BinarySigned`：第一个训练类别为 `-1`，第二个为 `1`。
- `SignedOrdinal`：三个训练类别依次为 `-1/0/1`。
- `Ordinal`：类别依次为 `0,1,...`。
- `OneHot`：输出列顺序为类别词典顺序，列名为 `<prefix>=<category>`。

`transform` 返回 `EncodedTarget`（`std::variant<Eigen::VectorXd, Eigen::MatrixXd>`），
也可通过 `transform_vector` 或 `transform_matrix` 使用与策略匹配的具体输出类型。

## 防止数据泄漏

统计量、类别集合、输入宽度和输出宽度均只在 `fit` 阶段建立。`transform` 只校验输入
宽度并读取这些已保存状态，不会从测试输入重新计算均值、范围或类别。

## 完成度

上述数值矩阵和文本标签 API、生命周期和边界检查已实现；DataFrame 列名选择、
字符串特征值、缺失值 dtype 体系和 remainder/passthrough 语义仍待稳定的 `data` API
后再扩展。

