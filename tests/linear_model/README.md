# tests/linear_model

该目录验证线性模型公共接口、统一 fit 状态、显式优化器选择、评价结果、标签映射和模型持久化。

测试应覆盖 `fit(X, y)` 与显式优化器入口、LinearRegression/Ridge/Lasso/ElasticNet 回归指标、LogisticRegression 二分类/多分类与决策阈值、RidgeClassifier 数值标签和 one-hot 输入，以及 `art::save`/`art::load` round-trip。

LabelEncoder 的 Binary01、BinarySigned、SignedOrdinal、Ordinal 和 OneHot 语义也应被验证。独立 `art::metrics` 尚未实现，不应由本目录伪造覆盖。

对应设计说明：`include/art/linear_model/README.md`。

## 统一 API 覆盖矩阵

`test_unified_model_api.cpp` 定义统一线性模型 API 的验收样例。该文件只负责测试
公共行为，不改变生产接口；若 CMake 尚未注册该测试，应由程序员0在统一构建调度中
接入对应测试目标。

| 模型 | `fit(X,y)` | 显式 Optimizer | 二分类 0/1 | 二分类 -1/1 | 多分类 one-hot | 文本标签边界 | evaluate/predict | save/load |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LinearRegression | ✓ | Dataset + GD/Newton | N/A | N/A | N/A | N/A | ✓ | 现有持久化测试 |
| Ridge | ✓ | Dataset + GD/Newton | N/A | N/A | N/A | N/A | ✓ | 现有持久化测试 |
| RidgeClassifier | ✓ | X/y 或 one-hot + Optimizer | ✓ | ✓ | ✓ | 需预编码 | ✓ | 现有持久化测试 |
| Lasso | ✓ | X/y + Optimizer | N/A | N/A | N/A | N/A | ✓ | 现有持久化测试 |
| ElasticNet | ✓ | X/y + Optimizer | N/A | N/A | N/A | N/A | ✓ | 现有持久化测试 |
| LogisticRegression | ✓ | X/y + Optimizer | ✓ | ✓ | 当前由向量类别编码覆盖 | LabelEncoder 边界 | ✓ | ✓ |

### 验收条件

- 统一 `fit(X,y)` 路径将模型置为 fitted，`predict`、`evaluate` 返回有限且维度正确的结果。
- 显式 Optimizer 路径必须真正使用传入的 GradientDescent 或 NewtonOptimizer；Newton 只用于当前支持的二分类范围。
- 分类输入的 0/1 与 -1/1 语义必须明确，one-hot 目标必须保持类别数和输出列数一致。
- 文本标签不能隐式进入数值模型；必须经 LabelEncoder 编码，并通过 `class_labels` 保留映射。
- 保存后加载的模型必须保持 fitted 状态、预测值、概率输出和类别映射。
- `std::filesystem::path` 保存与 `art::load(path.string())` 加载必须明确，不得通过字符串参数重载产生路径/标签歧义。

### 当前缺口

- RidgeClassifier 的显式 Optimizer 入口已补齐；WSL 验证需覆盖向量和 one-hot 两种目标入口。
- LogisticRegression 的多分类输入当前头文件只暴露 `VectorXd`，没有直接 `MatrixXd one-hot` 重载；测试不伪造不存在的接口，而以类别向量和文本标签编码路径覆盖。
- 测试文件尚未修改 CMake 注册；由程序员0在唯一串行 WSL 验证中决定注册目标和执行顺序。
