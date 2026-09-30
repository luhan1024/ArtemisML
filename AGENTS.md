# ArtemisML 项目规则

## 基础架构层级

后续开发以以下 `art` 层级作为基础结构，允许在各层内部增删查改和扩展，但不得在未说明依赖影响的情况下跨层混合职责：

```text
art
├── core
│   ├── 基础类型
│   ├── 错误类型
│   ├── 参数与配置
│   └── 数值后端适配
│
├── data
│   ├── Series
│   ├── DataFrame
│   ├── Index
│   ├── dtype
│   ├── 缺失值
│   └── 表格选择与运算
│
├── io
│   ├── CSV
│   ├── JSON
│   └── 其他数据格式
│
├── preprocessing
│   ├── StandardScaler
│   ├── MinMaxScaler
│   ├── Imputer
│   ├── OneHotEncoder
│   ├── ColumnTransformer
│   └── 特征选择与特征生成
│
├── pipeline
│   ├── Pipeline
│   └── FeatureUnion
│
├── base
│   ├── Estimator
│   ├── Transformer
│   ├── Predictor
│   ├── Regressor
│   └── Classifier
│
├── linear_model
│   ├── LinearRegression
│   ├── Ridge
│   └── LogisticRegression
│
├── metrics
│   ├── mean_squared_error
│   ├── mean_absolute_error
│   ├── r2_score
│   └── accuracy_score
│
├── model_selection
│   ├── train_test_split
│   ├── cross_validate
│   └── GridSearch
│
└── optim
    ├── GradientDescent
    ├── NewtonOptimizer
    └── OptimizationProblem
```

层级职责约束：

- `core` 提供基础类型、错误、配置和数值后端适配，不依赖具体模型。
- `data` 负责表格数据结构及其选择、运算和缺失值语义。
- `io` 负责外部数据格式与 `data` 对象之间的读写转换。
- `preprocessing` 负责可训练或可组合的数据预处理与特征变换。
- `pipeline` 负责串联转换器和估计器。
- `base` 定义估计器、转换器、预测器、回归器和分类器的公共协议。
- `linear_model` 放置线性模型实现。
- `metrics` 放置模型评估函数。
- `model_selection` 放置数据集切分、交叉验证和参数搜索。
- `optim` 放置通用优化问题、梯度下降和牛顿法等数学求解组件。
- 依赖方向应保持从基础设施和数据层流向算法组合层；`data` 和 `io` 不得反向依赖具体模型，`optim` 不得依赖具体数据格式。

## 构建与运行

- 项目的编译、测试和可执行程序运行统一在 WSL 中进行。
- WSL 内统一使用 `zsh` 执行命令；不得使用 `bash` 作为项目构建、测试或运行 shell。
- 优先使用 WSL 下的 CMake、CTest 和项目构建目录；不得把本机 Windows 构建结果作为 Linux/WSL 构建验证的替代品。
- 每次修改涉及源码、头文件或构建配置时，应在 WSL 中重新编译并运行相关测试。
- 多任务不得同时进行 WSL 配置、编译、测试或可执行程序运行。
- 同一时刻只能由一个线程占用构建验证流程；由程序员0统一安排任务顺序并在前一项完成后启动下一项。
- 未获得程序员0调度许可的线程只能进行静态分析、代码准备和报告整理，不得启动构建或测试命令。
- 每项构建任务必须报告开始、结束、命令、结果和失败原因，避免多个线程同时写入 `build-wsl` 造成冲突。

## 执行环境

- 不得使用子进程执行项目工作。
- 不得使用沙箱执行项目工作。
- 项目操作必须直接基于已授权的终端环境完成；不得通过额外的隔离执行层、代理执行层或后台子任务绕过该要求。

## 发布与 GitHub 同步

- 发布、提交、分支管理、推送以及与 GitHub 的同步统一基于本机 Windows 工作区执行。
- GitHub 远程仓库为 `https://github.com/luhan1024/ArtemisML.git`，默认远程名为 `origin`。
- WSL 仅负责编译、测试和运行，不作为发布或 GitHub 同步的工作区。
- 发布前应先在 WSL 中完成构建与测试，再回到本机检查 Git 差异、提交并按用户要求推送。

## 变更纪律

- 保留用户已有修改，不擅自覆盖或回退未请求的工作。
- 删除、覆盖重要文件或清理构建产物前，必须确认目标范围，并优先采用可恢复方式。
- 报告结果时区分 WSL 编译/运行验证与本机 GitHub 发布/同步状态。
