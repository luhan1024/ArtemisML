# src/core

该目录实现 `include/art/core` 中声明的 ArtemisML core 层公共接口。

## 实现边界

实现必须遵循对应公共 README 中的职责、公式和状态约束，不得在源码层新增未讨论的公共 API。算法、错误处理和边界条件应通过 `tests/core` 验证。

源码修改完成后，必须由程序员0安排唯一的 WSL 构建验证；验证通过后自动提交并推送。

当前 core 新增接口为 header-only，因而不需要 `src/core` 的编译单元。该目录保留用于
后续需要稳定 ABI 或非内联实现时的 `.cpp` 文件。core 只依赖 Eigen 和标准库，不依赖
`data`、IO 或任何模型层。
