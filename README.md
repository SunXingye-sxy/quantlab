# quantlab · A 股单因子回测框架

一个从零搭建的量化研究工具箱：数据管道 → 因子计算 → 分层回测 → 绩效评估。
面向 A 股日频数据，强调「可复现」与「无未来函数」。

## 进度

- [x] 数据管道（akshare 下载、前复权、本地 parquet 缓存、停牌剔除）
- [ ] 因子计算（动量 / 反转，MAD 去极值、z-score 标准化）
- [ ] 分层回测（五分组、月度调仓、交易成本）
- [ ] 绩效指标（年化收益、Sharpe、最大回撤、换手率、IC）

## 快速开始

```bash
# 1. 安装依赖（建议先建虚拟环境）
python -m venv .venv
# Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 下载数据（首次约 5-15 分钟，取决于网络）
python scripts/download_data.py --limit 20   # 先小样本验证链路
python scripts/download_data.py              # 全量沪深300，2018-2024

# 3. 数据体检
python scripts/check_data.py
```

数据缓存在 `data/`（已被 `.gitignore` 排除），换电脑后需重新执行下载脚本。

## 目录结构

```
quantlab/
├── quantlab/
│   ├── data.py      # 数据下载、缓存、宽表转换、停牌剔除
│   └── __init__.py
├── scripts/
│   ├── download_data.py   # 一键下载
│   └── check_data.py      # 数据体检
├── results/         # 图表与结果表（提交进仓库，README 引用）
├── data/            # 原始数据（不提交）
└── requirements.txt
```

## 数据治理（重要）

1. **前复权**：用 `adjust="qfq"`，否则除权日会出现假暴跌。
2. **停牌剔除**：成交量为 0 的记录不参与计算。
3. **交易日历**：用真实开市日，不把节假日当缺失。
4. **已知局限**：当前使用最新成分股名单，**存在幸存者偏差**（未使用 point-in-time 成分股）；后续版本将改为按历史成分股构造样本。

## License

MIT
