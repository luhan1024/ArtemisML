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
`linear_model` 四个线性模型的状态恢复；
回归模型加载后可获得 MSE、MAE 和 R2 指标。

依赖应从基础层流向组合层；数据和 IO 不得依赖具体模型。

