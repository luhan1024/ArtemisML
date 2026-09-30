# tests/data

该目录验证 data 层公共接口的行为、边界条件和错误语义。

测试应优先从用户可见 API 出发，覆盖正常路径、空输入、维度错误、状态错误和数值稳定性。若该层仍处于预留阶段，应记录当前未实现状态，不能用空测试掩盖缺口。

对应设计说明：`include/art/data/README.md`。

`test_dataset.cpp` 覆盖 Dataset 的合法转换、行宽错误、重复列/特征、零样本、零特征、
缺失标签和非有限数值。`test_data_api.cpp` 覆盖 Index 标签与默认索引、Series 访问/dtype/
数值转换、缺失值策略、DataFrame 形状、按列/行选择和越界错误。测试不依赖文件系统或
Windows 行为；WSL 构建测试由程序员0串行调度。
