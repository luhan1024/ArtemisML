# src/preprocessing

该目录以 `preprocessing.cpp` 实现公共头文件中的数值矩阵转换器，依赖方向为
`preprocessing -> base`，数值运算由 Eigen 提供；不依赖 `io`、`pipeline`、模型、指标、
模型选择或优化器。

## 实现约定

- 公共生命周期由 `art::base::Transformer` 提供，具体类只实现 `do_fit` 和
  `do_transform`。
- `fit` 失败后由 base 协议保持未训练状态；转换器自身在 fit 阶段保存输入宽度和
  统计量，transform 阶段拒绝宽度变化。
- 所有本层转换器保持样本行数；`OneHotEncoder`、`ColumnTransformer` 和
  `PolynomialFeatures` 可以改变特征列数。
- `LabelEncoder` 是独立的文本标签组件，不继承数值矩阵 `base::Transformer`；它在
  `std::vector<std::string>` 上执行 fit/transform，输出 `Eigen::VectorXd` 或
  `Eigen::MatrixXd`，避免把文本标签职责混入数值特征转换器。
- 输入为空、列索引越界、重复选择、非有限数值、未知类别和全缺失均按公共 README
  约定抛出 `std::invalid_argument`；未 fit 由 `base::NotFittedError` 处理。

本轮只完成静态源码与测试准备。WSL 配置、编译、CTest 和提交推送由项目维护者统一安排，
各并行任务不得自行启动构建流程。
