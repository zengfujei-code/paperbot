# BTC/USDT 模拟实验

Paperbot 的早期实验使用 Gate 的 BTC/USDT 现货行情，初始虚拟资金为 100 USDT。Freqtrade 以 `dry_run` 运行。实验脚本建立 30 天观察期，状态页展示模拟资产、持仓、成交和最近分析。

| 程序 | 职责 |
| --- | --- |
| `advisor.py` | 读取 15 分钟 K 线，计算指标，校验 DeepSeek 的结构化响应 |
| `DeepSeekPaperStrategy.py` | 将有效信号和量化阈值应用到 Freqtrade dry-run |
| `optimizer.py` | 根据已完成的模拟交易小幅调整有界阈值 |
| `dashboard.py` | 提供只读状态页及美股状态汇总 |

## 在独立的 Linux 模拟环境中运行

需要 Docker Compose 和 `compose.yaml` 中固定摘要的本地 Freqtrade 镜像。复制 `.env.deepseek.example` 为 `.env.deepseek`，在本机填写专用模型密钥并将文件权限设置为 600。交易所凭据留空。

- `bash doctor.sh` 校验本地镜像、Compose 配置与离线测试。
- `bash start.sh` 初始化运行目录并启动模拟容器。脚本需具备创建运行目录和设置所有者的权限。
- `bash status.sh` 查看实验、容器和最近日志。
- `bash stop.sh` 停止容器并保留记录。

Compose 将状态页绑定到本机 `127.0.0.1:8080`。SQLite 账本、决策记录和日志均位于 Git 忽略的 `runtime/`。
