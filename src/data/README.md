# src/data

该目录实现 `include/art/data` 中声明的 ArtemisML data 层公共接口。

## 实现边界

实现必须遵循对应公共 README 中的职责、公式和状态约束，不得在源码层新增未讨论的公共 API。算法、错误处理和边界条件应通过 `tests/data` 验证。

源码修改完成后，必须由程序员0安排唯一的 WSL 构建验证；验证通过后自动提交并推送。

Series、Index、dtype、缺失值和 DataFrame 的最小实现目前采用 header-only 形式，以避免
引入不必要的 ABI 和构建单元。`dataset.cpp` 保持 Dataset/CSV 桥接实现，并在访问行字段
前完成 DataFrame 结构校验。实现依赖方向为 `data -> core`；不得包含文件打开、CSV 解析、
预处理或模型训练逻辑。
