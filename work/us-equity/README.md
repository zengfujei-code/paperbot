# Paperbot 美股模拟模块

从仓库首页的离线测试开始。美股核心使用 Python 3.10+ 标准库。行情扫描与研究按美东交易日历处理已收盘的 15 分钟 K 线。

| 入口 | 职责 |
| --- | --- |
| `python -m paperbot_us.runner` | 读取 IEX 行情，归档扫描结果 |
| `python -m paperbot_us.shadow` | 根据扫描结果生成影子观察记录 |
| `python -m paperbot_us.model_runner` | 生成独立的 DeepSeek 分析记录 |
| `python -m paperbot_us.pilot_runner` | 执行受策略、账户状态与风控约束的模拟周期 |

在 `work/us-equity` 目录运行 `python -B -m paperbot_us.runner --help` 可查看扫描器参数。扫描与观察所需的模拟账户凭据通过进程环境注入。执行配置、账本和状态文件各自有独立路径。Linux 服务和定时器示例位于 `deploy/`，使用前须核对账户、数据源、路径和文件权限。

- [模拟券商接口约定](docs/BROKER_CONTRACT.md)
- [订单与账户对账约定](docs/RECONCILIATION_CONTRACT.md)
- [研究路线](RESEARCH_ROADMAP.md)
