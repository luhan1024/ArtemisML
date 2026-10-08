# ArtemisML

```text
============================================================================

   ###   ####   #####  #####  #   #  #####   ####   #   #  #
  #   #  #   #    #    #      ## ##    #    #      ## ##  #
  #####  ####     #    ####   # # #    #     ###   # # #  #
  #   #  #  #     #    #      #   #    #       #   #   #  #
  #   #  #   #    #    #####  #   #  #####  ####   #   #  #####

                         ARTEMISML

 Author      : Han Lu
 Contributor : Yihan Wang

============================================================================
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

新增 loss、model、optimizer、preprocessing、IO、pipeline、metrics 和 model_selection 的
标准 API 构建方法见 [art 公共 API 扩展规范](include/art/README.md)。该规范要求同步更新
公共头文件、实现、测试和三层 README，并明确兼容策略、输入输出形状、错误语义及验证命令。
公共 API 的兼容演进、数值输入布局和错误类型选择也以该规范为准；新增接口遵循目标契约，
历史接口的迁移状态以对应层级 README 为准。

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
- `art::data::TextLabelDataset` 保留文本特征与文本标签。
- `art::data::LabelVocabulary` 按首次出现顺序管理标签类别。
- `art::io::CsvReader::read_text_label_dataset` 用于 Iris 等文本标签 CSV。
- 特征列、标签列、重复列、空样本和非有限数值检查。

文本标签编码由 `art::preprocessing::LabelEncoder` 负责，与 CSV 读取解耦：
`Auto` 对二分类默认输出 `0/1`，对三分类及以上默认输出 one-hot；同时支持
`BinarySigned` 的 `-1/1`、三分类 `SignedOrdinal` 的 `-1/0/1`、普通序数编码和显式 one-hot。

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
- `art::linear_model::LogisticRegression`
- `art::optim::GradientDescent`
- `art::optim::NewtonOptimizer`
- `art::optim::OptimizationProblem`
- `art::core::autodiff::GradientTape` 标量反向模式梯度。
- `art::preprocessing::LabelEncoder` 文本标签自动编码。

线性回归和 Ridge 支持目标函数、梯度、Hessian、训练、预测和回归评分。
LogisticRegression 支持二分类 BCE 和多分类 Softmax Cross Entropy，训练使用梯度下降，
并提供概率预测、类别预测和准确率评分。

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

## 构建环境与 CMake 入门

ArtemisML 使用 CMake 管理构建。`CMakeLists.txt` 要求 CMake 3.20 或更新版本，项目使用
C++17；Eigen 已随仓库放在 `third_party/eigen`，不需要另外下载。构建、运行和测试统一在
WSL 的 `zsh` 中完成；Git 提交和 GitHub 同步在 Windows 工作区完成。

### 1. 准备环境并进入项目

请准备 WSL、`zsh`、CMake 3.20+ 和支持 C++17 的 C++ 编译器（例如 GCC）。在 Windows
PowerShell 中启动 WSL 的 zsh：

```powershell
wsl.exe -- zsh
```

进入仓库在 WSL 中对应的目录。Windows 的 `F:\Project\ArtemisML` 通常映射为：

```zsh
cd /mnt/f/Project/ArtemisML
```

如果仓库放在其他 Windows 盘符，将盘符映射为 `/mnt/<小写盘符>/...`；也可以在 WSL 中
先运行 `pwd` 确认当前位置，再切换到实际仓库目录。构建前确认当前目录包含
`CMakeLists.txt`：

```zsh
pwd
ls CMakeLists.txt
cmake --version
zsh --version
```

### 2. 首次配置

在仓库根目录运行：

```zsh
cmake -S . -B build-wsl
```

`-S .` 指定当前仓库为源码目录，`-B build-wsl` 指定独立的构建目录。CMake 会在该目录
生成构建系统和缓存；源码文件不会放进构建目录。首次配置后，同一工作区通常不需要每次
重新配置，修改 `CMakeLists.txt` 或构建选项时再运行配置命令即可。

如需显式选择构建类型，可在首次配置时指定：

```zsh
cmake -S . -B build-wsl -DCMAKE_BUILD_TYPE=Release
```

调试项目时可将 `Release` 换成 `Debug`。同一个构建目录应保持同一构建类型；切换类型时
重新配置，或使用另一个构建目录，例如 `build-wsl-debug`。

### 3. 编译库和测试程序

```zsh
cmake --build build-wsl
```

这会构建 `artemisml` 库以及 CMake 中注册的测试可执行程序。只编译单个目标时，可以指定
目标名称，例如：

```zsh
cmake --build build-wsl --target test_unified_model_api
```

查看已注册的目标可使用：

```zsh
cmake --build build-wsl --target help
```

### 4. 运行测试

先列出 CTest 测试：

```zsh
ctest --test-dir build-wsl -N
```

运行全部测试并显示失败信息：

```zsh
ctest --test-dir build-wsl --output-on-failure
```

只运行一个测试时使用 `-R` 匹配测试名：

```zsh
ctest --test-dir build-wsl -R unified_model_api --output-on-failure
```

常用测试名包括 `csv_reader`、`optimizer`、`linear_regression`、`logistic_binary`、
`model_persistence`、`unified_model_api`、`regularized_regression`、`base`、`dataset`、
`pipeline`、`autodiff`、`preprocessing`、`label_encoder`、`loss`、`core_types`、
`data_api` 和 `model_selection`。测试数量会随 CMake 注册目标的变更而变化；以
`ctest -N` 的当前输出为准。

### 5. 修改后重新构建

修改 C++ 源码或头文件后，重新编译并运行相关测试；修改构建配置后先重新配置：

```zsh
cmake -S . -B build-wsl
cmake --build build-wsl
ctest --test-dir build-wsl --output-on-failure
```

只想重新编译时可以跳过配置；只验证某一功能时，可以先构建对应测试目标，再用 CTest
的 `-R` 运行该测试。合并或发布前应运行完整测试集。

### 6. 清理构建结果

优先使用 CMake 的 clean 目标清理编译产物：

```zsh
cmake --build build-wsl --target clean
```

如果构建缓存损坏或需要完全重新配置，可在确认目标仅为生成的 `build-wsl` 后删除该构建
目录，再从首次配置步骤开始。不要把源码、数据集或其他工作目录当作构建产物删除。

### 常见问题

- **`The source directory does not appear to contain CMakeLists.txt`**：当前目录不在仓库根目录；先 `cd` 到包含 `CMakeLists.txt` 的目录。
- **CMake 版本过旧**：项目最低要求为 3.20；确认 WSL 中调用的 `cmake --version`，不要用 Windows 的 CMake 版本代替 WSL 环境。
- **找不到 C++ 编译器或生成失败**：确认 WSL 已安装并可调用 GCC/G++，然后重新运行 CMake 配置。
- **测试找不到 CSV 文件**：通过 CTest 运行已注册测试；CMake 为项目测试设置了仓库根目录作为工作目录。手动运行测试程序时也应从仓库根目录启动。
- **修改 CMake 后仍使用旧目标**：重新运行 `cmake -S . -B build-wsl`，再编译并检查 `ctest -N`。
- **多个任务同时编译出现文件冲突**：同一时间只运行一个 WSL 配置、编译、测试或项目程序流程；不要让多个任务并发写入或使用 `build-wsl`。

每次构建记录应注明开始/结束、执行命令、结果和失败原因。多任务的 WSL 构建验证由项目维护者统一串行安排。

## 当前实现范围

项目当前已包含 CSV 读写和数据集桥接、文本标签编码、数值预处理、Pipeline/FeatureUnion、
线性回归与分类模型、模型保存加载、损失函数、标量反向自动求导、优化器和模型选择等
实现。pandas 风格的完整动态 dtype、DataFrame 连接/分组/聚合、矩阵级自动求导以及更多
优化器和算法仍在扩展中；每项能力的准确实现状态以对应层级 README 和公共头文件为准。

## 贡献与开发约定

- 先阅读 [AGENTS.md](AGENTS.md) 再修改代码。
- 如借助 AI 辅助开发，请遵循 [AGENTS.md](AGENTS.md) 及各层级 README.md 中的具体规则。
- 保留已有用户修改，不擅自回退或删除文件。
- 编译和测试统一在 WSL 完成。
- 发布和 GitHub 同步基于本机 Git 工作区完成。
- 每个项目自有 `.h`、`.cpp` 和测试源码包含纯符号 ASCII 签名区。
- 主要贡献人：**Han Lu & Yihan Wang**。
