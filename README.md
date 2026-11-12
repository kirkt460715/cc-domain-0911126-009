# 材料试验数据版本库

本项目使用 Python 3.11 及以上版本，仅依赖标准库，无需安装第三方包。数据文件均为本地合成记录，不含真实个人或机构资料。

trialvault/contracts 定义数据字段和质量约束；artifacts 保存内容寻址文件；catalog 保存数据版本及血缘；releases 管理发布计划、批准、撤回及导出。

数据版本、发布批准和实际导出各有身份与摘要。更正创建新版本。被撤回的数据不能继续作为有效发布依据。运行请求通过 action.json 的 action 字段分发。

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
