# src/base

该目录实现 `include/art/base` 中声明的 ArtemisML base 层公共接口。

## 实现边界

实现必须遵循对应公共 README 中的职责、公式和状态约束，不得在源码层新增未讨论的公共 API。算法、错误处理和边界条件应通过 `tests/base` 验证。

源码修改完成后，必须由程序员0安排唯一的 WSL 构建验证；验证通过后自动提交并推送。
