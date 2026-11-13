# 材料试验数据版本库

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

trialvault/contracts 定义数据字段和质量约束；artifacts 保存内容寻址文件；catalog 保存数据版本及血缘；releases 管理发布计划、批准、撤回及导出；studies 在获批发布固定的数据版本上执行响应面研究。

数据版本、发布批准和实际导出各有身份与摘要。更正创建新版本。被撤回的数据不能继续作为有效发布依据。运行请求通过 action.json 的 action 字段分发。

## 响应面研究

研究只读取获批发布中已固定数据版本，不修改、补丁或删除来源版本。登记时给出连续因子（两到三个）及其登记实验范围、响应字段、独立试验批次字段和中心点重复标识；因子按 `center` 与半幅编码，模型保留主效应、两两交互和平方项。登记用 `study_register`，重算用 `study_evaluate`，读取用 `study`，交付状态用 `study_status`，导出用 `study_export`/`study_export_verify`。

研究结果包含：

- 设计矩阵秩与条件数（按单位化列计算），线性与完整二次模型分别报告；秩亏时明确拒绝拟合，不使用伪逆。
- 系数、标准误、检验统计量、残差、杠杆值和标准化残差。
- 纯误差与失拟检验：只有同一条件下属于不同独立批次的重复才计入纯误差；没有独立重复时明确说明无法作失拟判断。
- 线性模型与完整二次模型按试验批次隔离的交叉验证比较：整个批次只进入训练或验证之一，不在最终验证集上反复选项。
- 在登记因子盒和线性工艺约束内最大化预测均值，枚举内部驻点与全部边界面，报告候选位置、活跃约束、KKT 乘子和切空间二阶条件；报告声明是否具备全局证明，启发式搜索会明确标注。
- 推荐点不超出登记实验范围，并显示是否落在实际设计点的凸包内；范围内但凸包外的推荐带外推提示。
- 按批次重采样的最优值与推荐位置稳定性摘要，以及最优点跳到不同边界/约束集和拟合失败的出现情况。
- 来源发布撤回或不再合格后，研究保留可追溯记录，但不能作为有效交付，导出被拒绝。

`study_register` 请求示例：

```json
{
  "action": "study_register",
  "study_id": "window-study",
  "release_id": "release",
  "dataset": "trials",
  "version": 1,
  "factors": [
    {"field": "temperature", "low": "100", "high": "200", "center": "150"},
    {"field": "pressure", "low": "1", "high": "5", "center": "3"}
  ],
  "response": "strength",
  "batch_field": "batch",
  "center_repeat_field": "center_repeat",
  "factor_box": [
    {"field": "temperature", "low": "140", "high": "180"},
    {"field": "pressure", "low": "2", "high": "4"}
  ],
  "process_constraints": [
    {"id": "thermal-pressure", "coefficients": {"temperature": 1, "pressure": 20}, "bound": "210"}
  ],
  "folds": 3,
  "bootstrap": 200,
  "seed": 7
}
```

`factor_box` 只可收窄登记实验范围，不能超出；`process_constraints` 为线性工艺约束（按原始因子单位给出）。同一 `study_id` 的登记请求必须一致；`study_evaluate` 会用冻结版本和同一随机种子确定性地重算。

在项目根目录运行完整测试：

```sh
python3 -m unittest discover -s tests -v
```

编译检查：

```sh
python3 -m compileall -q .
```

实际演示：

```sh
python3 -m trialvault demo
```

演示在临时目录内建立本地状态，结束后自动清理。测试、编译与演示在依赖已准备的环境内均不下载数据或软件。输入错误以非零退出码及错误对象报告。更多子命令可通过对应入口的 --help 查看。

容器环境可执行 `docker build -t local-analysis .`，再运行 `docker run --rm --network none local-analysis`。镜像固定基础内容摘要，构建后运行只访问本地文件。完整测试与演示也可直接使用已有 Python 环境。
