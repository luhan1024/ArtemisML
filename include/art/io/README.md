# art::io

## 职责

`io` 负责外部数据格式与 `art::data` 对象之间的转换。

当前重点是 CSV，后续扩展 JSON 和其他格式。读取流程应包含：

1. 解析字节和行尾。
2. 处理引号、转义和多行字段。
3. 检查列数一致性。
4. 构造 DataFrame 或 Dataset。

IO 层报告格式错误和路径错误，不负责训练模型、填补缺失值或推断算法参数。

`CsvReader::read_text_label_dataset` 是面向文本标签数据的兼容适配器：它将 CSV 原始
字符串交给 data 层的 `TextLabelDataset`，保留标签文本并可生成首次出现顺序的
`LabelVocabulary`。IO 不执行 one-hot、二分类或其他标签编码。

