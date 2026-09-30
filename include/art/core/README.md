# art::core

## 职责

`core` 提供所有上层模块共享的基础设施：

- 基础数值类型和矩阵/向量别名。
- 统一错误类型。
- 参数与配置协议。
- Eigen 等数值后端适配。
- 自动求导的公共抽象。

该层不依赖具体模型、数据格式或 Pipeline。

## 自动求导规划

自动求导建议采用 `Variable + GradientTape` 协议。标量目标函数为 (f(\\theta)) 时，反向模式计算：

[
g_i = \\frac{\\partial f}{\\partial \\theta_i}
]

二阶算法可提供 Hessian 或 Hessian-vector product：

[
H v = \\nabla_{\\theta}(\\nabla_{\\theta} f \\cdot v)
]

第一阶段先稳定标量反向模式，再扩展 Jacobian、Hessian-vector product 和高阶导数。

