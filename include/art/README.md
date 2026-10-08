# art 公共 API

`include/art` 是 ArtemisML 的公共 API 根目录。所有公开接口均使用 `art` 命名空间，并按功能层组织。

## 层级

- `core`：基础类型、错误、配置、数值后端和自动求导协议。
- `data`：Series、DataFrame、Index、dtype 和缺失值语义。
- `io`：CSV、JSON 等外部格式转换。
- `preprocessing`：可训练的特征变换器。
- `pipeline`：转换器和估计器的组合。
- `base`：Estimator、Transformer、Predictor、Regressor、Classifier 协议。
- `linear_model`：线性回归、Ridge、LogisticRegression。
- `metrics`：评估函数。
- `model_selection`：切分、交叉验证和参数搜索。
- `optim`：目标函数、梯度、一阶和二阶优化器。
- `loss`：回归、二分类和多分类损失函数及其对 prediction/logits 的梯度。

`linear_model::LogisticRegression` 支持通过显式优化器对象选择梯度下降或二分类
Newton 求解；训练记录包含目标函数值、梯度范数和步长范数。

`art::save`/`art::load` 提供第一版模型持久化入口，当前支持
`linear_model` 六个线性模型的状态恢复；
回归模型加载后可获得 MSE、MAE 和 R2 指标。

依赖应从基础层流向组合层；数据和 IO 不得依赖具体模型。

## 公共 API 兼容与演进

公共头文件中的类型、函数、成员函数、枚举值和可观察行为都视为公共 API。新增接口应
优先采用兼容式扩展，并遵守以下规则：

- 不因内部重构随意删除、改名或改变既有接口的含义。
- 新增重载或配置项时，检查现有调用是否出现重载歧义；默认参数不得悄悄改变旧调用结果。
- 必须进行不兼容变更时，先说明原因、影响范围、替代接口和迁移示例，并同步更新测试与 README。
- 若旧接口仍需支持，保留兼容包装并委托到新实现；弃用标记和移除时点应与正式版本策略一致。
- 持久化文件格式是独立于 C++ API 的公共契约；格式变更必须更新版本和兼容测试。
- 在项目明确版本策略前，不承诺 ABI 稳定；每次公共 API 变更仍须在变更报告中说明源码兼容和潜在二进制兼容影响。

## 输入、输出与错误契约

数值机器学习 API 默认采用 `X.rows() == 样本数`、`X.cols() == 特征数` 的布局；回归目标和
单标签分类目标通常是长度为样本数的向量，多分类 one-hot 目标是 `样本数 × 类别数` 矩阵。
支持哪些形式必须由具体接口明示，不得静默猜测或改变编码。

每个数值 API 都必须定义空输入、维度不匹配、NaN/Inf、类别范围、未知类别及输出形状行为。
缺失值只按所属层的明确策略处理：例如预处理器可接受并填补 NaN，而要求有限数值的模型
应拒绝 NaN/Inf；不得把所有层的缺失值规则混成一个全局假设。

新 API 的错误类型按语义选择：非法参数使用 `core::ParameterError`；数据内容错误使用
`core::DataError` 或其子类；形状错误使用 `core::DimensionError`；类型、缺失值、索引和
不支持操作分别使用 `core::TypeError`、`core::MissingValueError`、`core::IndexError` 和
`core::UnsupportedOperationError`；未训练对象调用模型操作使用 `base::NotFittedError`。
调用方应能捕获对应基类。历史 API 尚未全部迁移，迁移期间 README 和变更报告必须注明
实际抛出的异常；新增接口不得继续扩大标准异常与项目异常混用的范围。

## 标准化 API 扩展流程

新增公共组件必须先确定所属层级、输入输出类型、状态生命周期和错误语义，再编写
头文件、实现、测试和本层 README。不得在具体模型中私自定义只供单一模型使用的公共协议。

每个新增组件至少完成以下闭环：

1. 在 `include/art/<layer>/` 声明最小公共 API，说明命名空间、输入输出形状、数值缩放和异常条件。
2. 在 `src/<layer>/` 实现声明的行为；header-only 组件必须在 README 中说明。
3. 在 `tests/<layer>/` 覆盖正常路径、边界条件、错误状态和数值验证。
4. 同步更新 include、src、tests 三处 README，准确区分已实现、兼容接口和计划接口。
5. 更新 CMake 测试目标，由项目维护者在 WSL `zsh` 中串行编译、测试和运行。
6. 验证通过后再提交；公共 API 变更必须记录兼容策略和依赖影响。

### 新增 loss 的标准入口

新损失应优先实现 `art::loss::LossFunction`，而不是在具体模型中新增散落的损失函数：

```cpp
class HuberLoss final : public art::loss::LossFunction {
public:
    explicit HuberLoss(double delta = 1.0,
                       art::loss::Reduction reduction = art::loss::Reduction::Mean);
    double value(const Eigen::MatrixXd& prediction,
                 const Eigen::MatrixXd& target) const override;
    Eigen::MatrixXd gradient(const Eigen::MatrixXd& prediction,
                             const Eigen::MatrixXd& target) const override;
};
```

实现顺序：定义公式和 reduction；检查形状、空输入、有限值和超参数；实现 `value()`；
实现对 prediction 的 `gradient()`；用有限差分核对梯度；最后才接入模型。target 默认是
常量，不提供隐含的 target 梯度。必须明确输入是值、概率、logits、稀疏类别索引还是 one-hot。

### 新增 model 的标准入口

新模型必须实现 `art::base` 协议，并提供统一生命周期：

```cpp
class HuberRegressor final : public art::base::Regressor {
public:
    using art::base::Regressor::fit;
    void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y,
             const art::optim::Optimizer& optimizer);
    Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;
    void save(const std::filesystem::path& path) const;
};
```

标准顺序：定义参数；实现 `fit(X,y)` 默认优化器入口；实现显式 optimizer 入口；让
optimizer 每次只更新一步而由模型负责循环和停止；实现 predict/evaluate/score；定义
未训练、空输入、维度错误和非有限值行为；保存预测所需完整状态；补充 round-trip 测试。
模型不得直接读取 CSV 或训练字符串标签，数据读取由 io 完成，标签编码由 preprocessing 完成。

