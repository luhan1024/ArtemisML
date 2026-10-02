# src/linear_model

该目录实现 `include/art/linear_model` 中声明的模型接口。模型通过 `art::base` 接收数值矩阵和数值标签；显式优化器通过 `art::optim::Optimizer::step` 注入，模型不持久化优化器指针。

当前实现包括 `LinearRegression`、`Ridge`、`RidgeClassifier`、`Lasso`、`ElasticNet` 和 `LogisticRegression`。回归模型提供 `RegressionMetrics`，分类模型提供 `ClassificationMetrics`、概率/决策输出和类别标签映射。模型保存委托给 `art::io` 的统一 `art::save`/`art::load` 格式，不在本目录解析持久化文件。

字符串标签编码由 `preprocessing::LabelEncoder` 负责；本目录只消费编码后的数值标签或 one-hot 矩阵。新增模型、独立 metrics 函数和额外优化器必须先补充公共头文件、测试和完成度说明。

统一 `fit(X, y)` 使用各模型默认优化器；新增的 `fit(X, y, optimizer)` 仅在训练调用期间
注入传入优化器，异常或训练完成后清空指针，不进入模型持久化状态。本轮未执行 WSL 配置、
编译、CTest 或运行命令。
