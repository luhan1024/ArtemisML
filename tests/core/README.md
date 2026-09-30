# tests/core

该目录验证 core 层公共接口的行为、边界条件和错误语义。

测试应优先从用户可见 API 出发，覆盖正常路径、空输入、维度错误、状态错误和数值稳定性。若该层仍处于预留阶段，应记录当前未实现状态，不能用空测试掩盖缺口。

对应设计说明：`include/art/core/README.md`。

`test_core_types.cpp` 覆盖 Shape 的默认/非空行为、NumericConfig 的默认策略以及错误
继承关系。`test_autodiff.cpp` 保留并覆盖现有标量反向模式接口。当前未测试尚未实现的
Jacobian、Hessian 和 Hessian-vector product。
