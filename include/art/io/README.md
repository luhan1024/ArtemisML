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

模型持久化提供统一入口 `art::save(model, path)` 和 `art::load(path)`。
第一版支持 `linear_model::LogisticRegression`、`LinearRegression`、`Ridge` 和
`RidgeClassifier`、`Lasso` 和 `ElasticNet`，使用带 `ARTEMISML_MODEL` 签名和
格式版本的单文件文本格式。加载时验证签名、版本、类型、矩阵维度、类别数量和
有限值，不执行文件中的代码。

具体模型和运行时包装均提供成员式便捷调用：

```cpp
model.save(std::filesystem::path{"model.artemisml"});
auto loaded = art::load("model.artemisml");
loaded.save(std::filesystem::path{"model-copy.artemisml"});
```

## 新增 IO 或持久化格式规范

新增数据格式必须提供 reader/writer 的明确入口，负责格式解析和 `data` 对象转换，
不得在 IO 层调用具体模型。新增模型持久化必须使用稳定类型名、格式签名和版本号，
保存所有预测所需状态，并在加载时验证维度、有限值、类别数量和版本。

持久化扩展必须增加合法文件 round-trip、损坏签名、未知版本、维度错误和不完整字段
测试。加载结果必须能继续 `predict/evaluate/save`，不能只验证文件可读。

