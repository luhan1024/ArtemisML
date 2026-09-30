# ArtemisML

```text
  ###    ####   #####  #####  #   #  #####   ####  #   #  #
 #   #  #   #     #    #      ## ##    #    #   #  ## ##  #
 #####  ####      #    ###    # # #    #    ####   # # #  #
 #   #  # #       #    #      #   #    #    # #    #   #  #
 #   #  #  ##   #####  #####  #   #  #####  #  ##  #   #  #####
```

ArtemisML 是一个基于 C++17 的机器学习库，目标是将类似 pandas 的表格数据处理能力与类似 scikit-learn 的机器学习对象协议结合起来，形成清晰、可组合、可扩展的本地机器学习基础设施。

主要贡献人：**Han Lu & Yihan Wang**

## 项目定位

ArtemisML 面向以下工作流：

```text
外部数据
   ↓
DataFrame / Dataset
   ↓
数据预处理与特征变换
   ↓
Pipeline
   ↓
机器学习模型
   ↓
指标评估与模型选择
```

项目采用分层设计：数据对象负责数据语义，IO 负责外部格式转换，预处理器负责可训练的数据变换，Estimator/Transformer/Predictor 定义机器学习协议，模型和优化器负责具体算法实现。

## 当前目录结构

```text
ArtemisML/
├── include/art/
│   ├── core/
│   ├── data/
│   ├── io/
│   ├── preprocessing/
│   ├── pipeline/
│   ├── base/
│   ├── linear_model/
│   ├── metrics/
│   ├── model_selection/
│   ├── optim/
│   └── loss/
├── src/
│   ├── core/
│   ├── data/
│   ├── io/
│   ├── preprocessing/
│   ├── pipeline/
│   ├── base/
│   ├── linear_model/
│   ├── metrics/
│   ├── model_selection/
│   ├── optim/
│   └── loss/
├── tests/
├── third_party/eigen/
├── CMakeLists.txt
└── AGENTS.md
```

各层职责和依赖方向由 [AGENTS.md](AGENTS.md) 统一规定。

## 分层文档

每个项目自有目录均维护对应的 README，文档与代码层级保持一致：

- [公共头文件总览](include/README.md)
- [art 公共 API](include/art/README.md)
- [源码实现总览](src/README.md)
- [测试总览](tests/README.md)
- 各功能层的 API、公式、算法和完成度说明位于对应目录的 README。

自动求导是当前接口标准的核心基础设施。当前已实现
`art::core::autodiff::Variable`、`GradientTape`、标量反向模式梯度以及
`exp/log` 运算；Jacobian、Hessian 和 Hessian-vector product 将在兼容现有协议的基础上继续扩展。

损失函数位于 `art::loss`，提供 MSE、MAE、logits 形式 BCE、Softmax Cross Entropy
和 Sparse Cross Entropy。损失梯度是对 prediction/logits 的解析梯度，Mean/Sum
缩放语义与目标编码要求见 [loss 公共 API](include/art/loss/README.md)。

## 当前已实现功能

### CSV 与数据集

- CSV 文件读取和写出。
- LF、CRLF 行尾处理。
- 引号字段、逗号字段和多行字段处理。
- CSV 格式错误、列数不一致和未闭合引号检查。
- `art::data::DataFrame` 基础表格容器。
- `art::data::Dataset` 从 CSV 或 DataFrame 提取数值特征与标签。
- 特征列、标签列、重复列、空样本和非有限数值检查。

### 基础机器学习协议

当前提供：

- `art::base::Estimator`
- `art::base::Transformer`
- `art::base::Predictor`
- `art::base::Regressor`
- `art::base::Classifier`

支持：

- `fit`
- `transform`
- `fit_transform`
- `predict`
- `score`
- fitted 状态检查。
- 未训练调用错误。
- 失败训练后的状态清理。

当前数值协议基于 Eigen 的矩阵和向量类型。

### Pipeline

当前提供：

- `art::pipeline::Pipeline`
- `art::pipeline::FeatureUnion`

支持：

- 顺序连接 Transformer 和最终 Estimator。
- 前置变换器的 `fit → transform` 数据流。
- 预测和评分阶段不重新训练预处理器。
- 多分支特征横向拼接。
- 空步骤、重复名称、非法步骤类型和样本数不一致检查。

### 线性模型与优化器

当前提供：

- `art::linear_model::LinearRegression`
- `art::linear_model::Ridge`
- `art::linear_model::LogisticRegression` 接口预留。
- `art::optim::GradientDescent`
- `art::optim::NewtonOptimizer`
- `art::optim::OptimizationProblem`
- `art::core::autodiff::GradientTape` 标量反向模式梯度。

线性回归和 Ridge 支持目标函数、梯度、Hessian、训练、预测和回归评分。LogisticRegression 当前明确标记为尚未实现，不作为可用分类模型发布。

### 损失函数

当前提供：

- `art::loss::LossFunction`
- `art::loss::Reduction::Mean` 与 `art::loss::Reduction::Sum`
- `art::loss::MeanSquaredError`
- `art::loss::MeanAbsoluteError`
- `art::loss::BinaryCrossEntropy`
- `art::loss::SoftmaxCrossEntropy`
- `art::loss::SparseCrossEntropy`

BCE 和 Softmax Cross Entropy 直接接收 logits，并使用稳定的数值公式。第一阶段暂不
实现 AdaGrad、Adam、FocalLoss 或其他扩展损失。

## 当前测试

当前已注册并通过的 WSL CTest 包括：

- `csv_reader`
- `optimizer`
- `linear_regression`
- `base`
- `dataset`
- `pipeline`
- `autodiff`
- `preprocessing`
- `loss`
- `core_types`
- `data_api`
- `model_selection`

最近一次完整验证结果：

```text
100% tests passed out of 12
```

## 构建与测试

项目规定在 WSL 中完成配置、编译、运行和测试，在本机 Windows 工作区完成 Git 提交、发布和 GitHub 同步。

```bash
cmake -S . -B build-wsl
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
```

多任务不得同时写入或使用 `build-wsl`。构建验证由程序员0统一串行调度。

## 当前未完成部分

以下模块已经预留目录，但尚未形成完整实现：

- `core` 基础类型和统一错误体系。
- pandas-like `Series`、`Index`、dtype 和完整缺失值语义。
- `preprocessing`。
- `metrics`。
- `model_selection`。
- 完整的 `LogisticRegression`。
- 更完整的 DataFrame 选择、连接、分组和聚合操作。
- 更丰富的损失函数和矩阵级自动求导支持。

这些模块应在现有层级基础上逐步实现，不应通过空实现或未经讨论的 API 重命名掩盖未完成状态。

## 贡献与开发约定

- 先阅读 [AGENTS.md](AGENTS.md) 再修改代码。
- 保留已有用户修改，不擅自回退或删除文件。
- 编译和测试统一在 WSL 完成。
- 发布和 GitHub 同步基于本机 Git 工作区完成。
- 每个项目自有 `.h`、`.cpp` 和测试源码包含纯符号 ASCII 签名区。
- 主要贡献人：**Han Lu & Yihan Wang**。
