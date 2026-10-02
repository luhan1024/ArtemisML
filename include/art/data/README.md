# art::data

## 职责

`data` 提供类似 pandas 的表格数据抽象：

    - `Series`：带名称和索引的一维数据。
    - `DataFrame`：带行列标签的二维表。
- `Index)：行列标签。
- `dtype)：数据类型描述。
- 缺失值、选择、切片、连接和表格运算。

## 语义要求

数据对象必须明确区分空值、缺失值、非法值和非有限数值。选择操作应保持索引和列名语义；数值算法需要显式要求可转换的数值列，不能静默把字符串当作数值。

该层不依赖 IO、预处理器或模型。

## 当前最小 API 与语义

- `Index` 使用字符串标签；`default_index(n)` 生成 `"0"` 到 `"n-1"`。标签必须唯一，
  位置越界和标签不存在统一抛出 `IndexError`。
- `Series` 当前以 `std::vector<std::string>` 保存原始字段，附带名称和 `Index`。这是对
  现有 CSV/DataFrame 兼容的明确边界；`dtype()` 推断 integer、floating、boolean、string
  或 unknown，`to_numeric()` 只做显式转换，不静默填补缺失或非法值。
- `DataFrame` 保留公开 `column_names` 与 `rows`，保证旧 CSV 聚合初始化兼容。`shape()`
  返回 `(rows.size(), column_names.size())`；`validate()` 检查重复列名和每行宽度。列、行、
  多列和多行选择都返回副本，不会修改原对象。未保存自定义行索引，`row_index()` 返回默认
  整数索引。
- 缺失值由 data 层按 `MissingPolicy` 分类：默认空字符串不是缺失，`null/NULL` 和
  `nan/NaN/NAN` 是缺失；CSV 读取只提供原始字符串，不在 IO 层固化策略。

Dataset 仍是 data 层中从 DataFrame 到 Eigen 特征矩阵/标签向量的桥接对象，保留
`from_csv` 兼容入口；它不承担 fit、transform 或模型训练。数据层当前不实现完整 pandas
动态类型、视图、广播、连接、分组和聚合。

对于 Iris 等文本标签数据，`TextLabelDataset::from_dataframe` 保留特征字段和标签原文；
`LabelVocabulary` 只按首次出现顺序建立去重词典和查找，不执行 one-hot、二分类或其他
编码。数值 `Dataset::from_dataframe/from_csv` 的 double 标签行为保持不变，因此文本标签
调用方应使用 IO 的 `CsvReader::read_text_label_dataset`；`numeric_features()` 可将明确的
数值特征转换为当前 Eigen 矩阵协议，再由 preprocessing 的 `LabelEncoder` 决定标签编码策略。

## 新增 data API 规范

新增 Series、DataFrame 或 Dataset 能力必须先定义 dtype、缺失值、索引、复制/视图和
错误语义；不得在 data 层读取 CSV、训练模型或执行标签编码。新增表格操作应说明输入
列名/位置、输出形状、是否修改原对象，并在 `tests/data` 覆盖空表、重复列、行宽错误、
缺失值和类型转换边界。

