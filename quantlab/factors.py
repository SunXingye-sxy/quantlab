"""因子构造：动量、反转，以及截面处理（MAD 去极值、z-score 标准化）。

设计原则
--------
1. 全部向量化：用宽表（index=date, columns=code）做矩阵运算，不逐股票循环。
2. 无未来函数：因子在 T 日只用 T 日及之前的收盘价，回测时必须把信号 shift(1)
   到 T+1 才成交，这一步在 backtest 层做，本模块只保证「T 日因子不含 T 日以后的信息」。
3. 截面处理按日进行：同一交易日内跨股票比较，不跨日，否则会把不同市场环境混在一起。

为什么动量要跳过最近 1 日
------------------------
A 股短期（1–5 日）存在明显的反转效应：昨天涨停的股票今天往往回落。
若动量因子直接用「昨天收盘 / 60 天前收盘」，短期反转会污染真实的趋势信号。
标准做法是 skip=1，用 T-1 到 T-60 的区间收益，避开最近一天的噪声。

为什么用 MAD 而不是直接删极值
----------------------------
按 3σ 或 5σ 截断依赖均值与标准差，而这两者本身就被极值污染（一个涨停就能把标准差
拉大）。MAD（中位数绝对偏差）用中位数做锚，对极值不敏感，去极值后再算均值方差才有意义。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import to_wide

# 正态分布下 MAD 与标准差的换算系数：sigma ≈ 1.4826 * MAD
MAD_SCALE = 1.4826


def mad_winsorize(wide: pd.DataFrame, n: float = 5.0, scale: float = MAD_SCALE) -> pd.DataFrame:
    """按日截面 MAD 去极值（缩尾）。

    Parameters
    ----------
    wide : DataFrame
        index=date, columns=code 的因子矩阵。
    n : float
        倍数阈值，默认 5 倍 MAD，对应正态下约 ±5σ。因子通常取 3–5。

    Returns
    -------
    DataFrame
        与输入同形状，超出上下界的值被压到边界（不是删除）。
    """
    median = wide.median(axis=1)
    mad = wide.sub(median, axis=0).abs().median(axis=1)
    upper = median.add(n * scale * mad, axis=0)
    lower = median.sub(n * scale * mad, axis=0)
    return wide.clip(lower=lower, upper=upper, axis=0)


def zscore(wide: pd.DataFrame) -> pd.DataFrame:
    """按日截面标准化，均值为 0、标准差为 1。

    标准化是为了让不同因子可以相加/比较（多因子合成时必需），
    也让「因子值」变成「相对全市场的排名位置」，便于分组。
    """
    mean = wide.mean(axis=1)
    std = wide.std(axis=1).replace(0, np.nan)  # 全同值时避免除 0
    return wide.sub(mean, axis=0).div(std, axis=0)


def momentum(close_wide: pd.DataFrame, window: int = 60, skip: int = 1) -> pd.DataFrame:
    """动量因子：过去 window 个交易日的累计收益，跳过最近 skip 日。

    close_wide 需为前复权收盘价宽表。skip=1 是为了避开短期反转效应。
    """
    return close_wide.shift(skip) / close_wide.shift(skip + window) - 1.0


def reversal(close_wide: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """反转因子：最近 window 日收益取负（跌得多的后续可能反弹）。"""
    return -(close_wide / close_wide.shift(window) - 1.0)


# 默认因子集合：两个不同周期的动量 + 一个短期反转
DEFAULT_SPECS: dict[str, tuple] = {
    "mom_20": ("momentum", 20, 1),
    "mom_60": ("momentum", 60, 1),
    "rev_5": ("reversal", 5),
}


def build_factor(
    spec: tuple,
    close_wide: pd.DataFrame,
    winsorize_n: float = 5.0,
    normalize: bool = True,
) -> pd.DataFrame:
    """按一条 spec 构造单个因子，并完成去极值与标准化。"""
    kind = spec[0]
    args = spec[1:]
    if kind == "momentum":
        raw = momentum(close_wide, *args)
    elif kind == "reversal":
        raw = reversal(close_wide, *args)
    else:
        raise ValueError(f"未知因子类型：{kind}")

    raw = raw.replace([np.inf, -np.inf], np.nan)
    if winsorize_n:
        raw = mad_winsorize(raw, winsorize_n)
    if normalize:
        raw = zscore(raw)
    return raw


def build_factors(
    panel: pd.DataFrame,
    specs: dict[str, tuple] | None = None,
    winsorize_n: float = 5.0,
    normalize: bool = True,
) -> dict[str, pd.DataFrame]:
    """批量构造因子，返回 {因子名: 宽表}。

    Parameters
    ----------
    panel : DataFrame
        长表，需包含 date / code / close，且已剔除停牌（close 缺失）。

    注意：本函数只保证 T 日因子不含未来信息，不做「T 日因子 -> T+1 成交」的错位，
    错位由 backtest 层用 signal.shift(1) 完成。这是分工，不是遗漏。
    """
    specs = specs or DEFAULT_SPECS
    close_wide = to_wide(panel, "close")
    return {
        name: build_factor(spec, close_wide, winsorize_n, normalize)
        for name, spec in specs.items()
    }
