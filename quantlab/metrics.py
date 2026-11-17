"""绩效指标：年化收益、波动、夏普、最大回撤。

所有函数都只吃「日频收益序列」或「净值序列」，不依赖回测细节，
这样无论是单组、多空组合还是基准，都能用同一套口径评价。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annual_return(nav: pd.Series, periods: int = TRADING_DAYS) -> float:
    """几何年化收益：把总收益按交易日数折算成年化。"""
    nav = nav.dropna()
    if len(nav) < 2 or nav.iloc[0] <= 0:
        return np.nan
    years = len(nav) / periods
    return float(nav.iloc[-1] ** (1.0 / years) - 1.0)


def annual_vol(returns: pd.Series, periods: int = TRADING_DAYS) -> float:
    """年化波动率：日收益标准差 × sqrt(252)。"""
    r = returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std() * np.sqrt(periods))


def sharpe(returns: pd.Series, rf: float = 0.0, periods: int = TRADING_DAYS) -> float:
    """夏普比率。rf 为年化无风险利率。"""
    r = returns.dropna() - rf / periods
    if len(r) < 2 or r.std() == 0:
        return np.nan
    return float(r.mean() / r.std() * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> float:
    """最大回撤（正数表示回撤幅度）。"""
    nav = nav.dropna()
    if len(nav) < 2:
        return np.nan
    peak = nav.cummax()
    return float((1 - nav / peak).max())


def summary_table(navs: dict[str, pd.Series], rf: float = 0.0) -> pd.DataFrame:
    """把多组净值汇总成一张绩效表（每行一个组合）。"""
    rows = []
    for name, nav in navs.items():
        rets = nav.pct_change(fill_method=None)
        rows.append({
            "组合": name,
            "年化收益": annual_return(nav),
            "年化波动": annual_vol(rets),
            "夏普": sharpe(rets, rf=rf),
            "最大回撤": max_drawdown(nav),
            "累计收益": float(nav.iloc[-1] - 1) if len(nav) else np.nan,
        })
    table = pd.DataFrame(rows).set_index("组合")
    return table.round(4)
