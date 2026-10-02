# art::core

## 职责

`core` 提供所有上层模块共享的基础设施：

- 基础数值类型和矩阵/向量别名。
- 统一错误类型。
- 参数与配置协议。
- Eigen 等数值后端适配。
- 自动求导的公共抽象。

该层不依赖具体模型、数据格式或 Pipeline。

## 当前最小 API

- `types.h`：`Size`、`RowCount`、`ColumnCount`、`Position` 和 `Shape`。
- `errors.h`：`ParameterError`、`DataError`、`DimensionError`、`TypeError`、
  `MissingValueError`、`IndexError`、`UnsupportedOperationError`。这些类型均可按
  `std::runtime_error` 捕获，细分类别用于调用方精确处理。
- `config.h`：仅提供通用 `NumericConfig::require_finite` 策略；模型超参数不属于 core。
- `backend.h`：`Eigen::MatrixXd` 和 `Eigen::VectorXd` 的稳定别名，不替换 Eigen。

`Shape{r,c}` 的元素容量语义为 `r*c`；只要行数或列数为零，`empty()` 为真。当前配置
没有非法范围，`validate()` 保留为后端统一策略的扩展点。

完成度：基础类型、错误分类、配置占位和 Eigen 适配已实现；统一错误体系尚未迁移所有
历史模块，自动求导仍由现有 `autodiff.h` 独立维护。

## 自动求导

当前已实现 `Variable + GradientTape` 协议和标量反向模式。标量目标函数为 (f(\\theta)) 时，反向模式计算：

[
g_i = \\frac{\\partial f}{\\partial \\theta_i}
]

二阶算法可提供 Hessian 或 Hessian-vector product：

[
H v = \\nabla_{\\theta}(\\nabla_{\\theta} f \\cdot v)
]

当前支持基本加减乘除、`exp` 和 `log`；后续扩展 Jacobian、Hessian-vector product 和高阶导数。

## 新增 core API 规范

core 新接口必须与具体数据格式、模型和优化器解耦。新增类型先定义生命周期、所有权、
维度和错误类型，再决定是否 header-only；不得在 core 中引入 CSV、模型参数或具体 loss。
涉及求导时必须说明输入变量、输出变量、梯度/Jacobian 形状和不可导点策略，并在
`tests/core` 用已知公式或有限差分验证。

