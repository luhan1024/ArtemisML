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

## 错误类型选择规范

新增 API 应按 `include/art/README.md` 的错误契约抛出最具体的项目异常：参数配置错误用
`ParameterError`；数据错误用 `DataError` 及其维度、类型、缺失值和索引子类；不支持的
操作用 `UnsupportedOperationError`。未训练模型状态属于 `base::NotFittedError`，不应为此
让 core 依赖 base。

这些项目异常当前统一继承 `core::Error`（`std::runtime_error`）。历史代码中的
`std::invalid_argument`、`std::out_of_range` 和 `std::logic_error` 尚未全部迁移；新增实现
应使用项目异常，历史接口迁移时须同步调整异常测试并记录兼容影响。错误消息供诊断使用，
程序逻辑应优先依据异常类型处理，不应解析消息文本。

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

