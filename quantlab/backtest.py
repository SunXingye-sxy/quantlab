"""分层回测：按因子值分成 N 组，日频调仓，扣除交易成本。

三个最容易做错的地方，这里都显式处理
------------------------------------
1. 信号错位（未来函数）
   T 日收盘才拿得到 T 日因子，不可能当天收盘成交。所以信号一律 shift(1)：
   T 日的因子 → T+1 日成交 → 吃 T+1 的收益。少这一行，年化收益能凭空多出十几个点。

2. 分组要按当日截面百分位重新排
   不能用「全样本排名」或固定持仓，否则组合会漂移成「一直拿着原来那只票」，
   测的就不是因子，而是个股。

3. 交易成本按权重变化计费
   不是按调仓次数拍一个常数，而是 sum(|w_t - w_{t-1}|) * 单边费率。
   换手大的因子会被真实扣分，这才公平。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_wide


def group_weights(signal: pd.DataFrame, n_groups: int = 5) -> dict[str, pd.DataFrame]:
    """按日截面把股票分成 n_groups 组，组内等权。

    signal 必须是「已错位到可交易时点」的因子矩阵：第 t 行的值表示
    t 日开盘前就应建好的仓位。G1 是因子值最小的一组，G{n} 是最大的一组。

    注意：若某日有效股票数少于组数，靠后的组会是空仓（权重全 0），
    样本极小时属正常现象，换成真实截面（几百只）就不会出现。
    """
    valid = signal.notna()
    n_valid = valid.sum(axis=1).replace(0, np.nan)

    # rank 从 1 开始，(rank-1)/n 得到 [0,1) 的百分位，再切成 n_groups 段
    rank = signal.rank(axis=1)
    pct = rank.sub(1).div(n_valid, axis=0)
    bucket = np.floor(pct * n_groups).clip(upper=n_groups - 1)

    weights: dict[str, pd.DataFrame] = {}
    for g in range(n_groups):
        mask = (bucket == g) & valid
        count = mask.sum(axis=1).replace(0, np.nan)
        w = mask.astype(float).div(count, axis=0).fillna(0.0)
        weights[f"G{g + 1}"] = w
    return weights


def _apply_rebalance(w: pd.DataFrame, rebalance: int) -> pd.DataFrame:
    """把每日权重降频为「每 rebalance 日调一次，中间持仓不变」。

    日频调仓的换手极高，弱因子会被成本吃光。降频是最直接的补救：
    持仓 5 天不动，换手理论上降到 1/5，成本也降到 1/5。
    """
    if rebalance <= 1:
        return w
    sampled = w.iloc[::rebalance]
    return sampled.reindex(w.index).ffill().fillna(0.0)


def backtest_factor(
    panel: pd.DataFrame,
    factor: pd.DataFrame,
    n_groups: int = 5,
    cost_rate: float = 0.001,
    long_short: bool = True,
    rebalance: int = 1,
) -> dict:
    """对单个因子做分层回测。

    Parameters
    ----------
    factor : 因子宽表（未错位；函数内部会 shift(1) 到可交易时点）
    cost_rate : 单边费率（佣金 + 印花税 + 滑点合计）。A 股实操约 0.001–0.002，
                保守取 0.001（万十），做成本敏感性时再上调。
    rebalance : 调仓周期（交易日）。1 = 每日调仓，5 = 每周，20 = 每月。
    """
    close = to_wide(panel, "close")
    ret = close.pct_change(fill_method=None)  # t 日收益 = close_t / close_{t-1} - 1

    signal_exec = factor.shift(1)  # 杜绝未来函数
    weights = group_weights(signal_exec, n_groups=n_groups)

    result = {
        "weights": {}, "returns": {}, "turnover": {}, "cost": {}, "navs": {},
    }

    for name, w in weights.items():
        w = _apply_rebalance(w, rebalance)
        gross = (w * ret).sum(axis=1)                              # 毛收益
        traded = (w - w.shift(1)).abs().sum(axis=1).fillna(0.0)    # 双边换手额
        net = gross - traded * cost_rate                           # 扣费后
        result["weights"][name] = w
        result["returns"][name] = net
        result["turnover"][name] = traded
        result["cost"][name] = traded * cost_rate
        result["navs"][name] = (1 + net).cumprod()

    if long_short and n_groups >= 2:
        ls_name = "多空(高-低)"
        ls_ret = result["returns"][f"G{n_groups}"] - result["returns"]["G1"]
        result["returns"][ls_name] = ls_ret
        result["navs"][ls_name] = (1 + ls_ret).cumprod()

    return result
