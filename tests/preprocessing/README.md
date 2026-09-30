# tests/preprocessing

`test_preprocessing.cpp` 从用户可见 API 验证：

- 未 fit 调用 `transform`、重复 fit 和输入维度错误；
- StandardScaler 的均值/尺度变换与 MinMaxScaler 的目标范围；
- Imputer 的均值、常数策略和 `NaN` 缺失语义；
- OneHotEncoder 的类别顺序、未知类别 `Ignore`/`Error` 行为和列数变化；
- ColumnTransformer 的按列组合、行数保持和输出列数；
- SelectColumns 与 PolynomialFeatures 的基础特征选择/生成。
- LabelEncoder 从 `iris.csv` 读取三类文本标签，验证 Auto OneHot、SignedOrdinal、
  类别顺序、输出列名和未知标签策略；另验证二分类 `Binary01` 与 `BinarySigned`。

测试只覆盖数值矩阵边界，不伪造字符串 DataFrame API。建议由程序员0统一串行执行唯一
验证命令：

```bash
cmake -S . -B build-wsl && cmake --build build-wsl && ctest --test-dir build-wsl --output-on-failure
```
