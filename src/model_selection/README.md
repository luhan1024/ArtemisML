# src/model_selection

该目录实现 `include/art/model_selection` 中声明的 ArtemisML model_selection 层公共接口。

## 实现边界

实现只使用 `art::base::Estimator`/`Predictor` 和 Eigen 数值类型。每个 fold 或参数组合都通过工厂创建独立对象；`FitPredictFunction` 将 fit/predict 过程作为回调注入，以避免依赖 Pipeline、具体模型或 metrics。源码不保存跨调用的 fitted 模型状态。

参数网格使用 `std::map<std::string, std::vector<double>>`，按键序和输入值序确定性枚举笛卡尔积。单组失败不终止其他组合，错误文本保存在 trial 中；只有成功 trial 能成为最佳结果。

该层源码和测试完成后，必须由程序员0安排唯一的 WSL 构建验证；本线程不启动构建、测试或运行命令。
