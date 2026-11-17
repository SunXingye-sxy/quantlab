"""一键构造因子、落盘，并做一次轻量的有效性体检。

用法：
    python scripts/build_factors.py

输出：
    1. 每个因子的有效日数、覆盖度、缺失率
    2. 因子与「次日收益」的截面 IC 均值、ICIR、t 值（快速判断因子有没有用）
    3. 因子矩阵落盘到 data/factors.pkl，供回测脚本读取
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from quantlab.data import DATA_DIR, filter_untradable, load_panel, to_wide
from quantlab.factors import DEFAULT_SPECS, build_factors


def main() -> None:
    panel = load_panel()
    print(f"原始样本  : {len(panel):,} 条 / {panel['code'].nunique()} 只")
    tradable = filter_untradable(panel)
    print(f"剔除停牌后: {len(tradable):,} 条（保留 {len(tradable)/len(panel):.1%}）")

    factors = build_factors(tradable, DEFAULT_SPECS)
    close = to_wide(tradable, "close")
    # T 日收盘买入 -> T+1 收盘卖出，这是 IC 检验用的「次日收益」，
    # 与回测里 signal.shift(1) 的逻辑一致：T 日的因子只能吃到 T+1 的收益。
    fwd_ret = close.shift(-1) / close - 1.0

    print("\n因子概况（IC 为每日截面 Spearman 相关，因子值 vs 次日收益）")
    header = f"{'因子':<8}{'有效日数':>10}{'日均覆盖':>10}{'缺失率':>9}{'IC均值':>10}{'ICIR':>9}{'t值':>8}"
    print(header)
    print("-" * len(header.encode("gbk", errors="ignore").decode("gbk")))
    for name, f in factors.items():
        valid_rows = int(f.notna().any(axis=1).sum())
        daily_cover = f.notna().sum(axis=1).replace(0, np.nan).median()
        miss = 1 - f.notna().sum().sum() / f.size
        # 截面 Spearman 相关 = 先转排名再算 Pearson。
        # 不直接用 corrwith(method="spearman")，因为 pandas 3.x 对该参数的行为有调整，
        # 先 rank 再相关是等价写法，跨版本更稳。
        ic = f.rank(axis=1).corrwith(fwd_ret.rank(axis=1), axis=1).dropna()
        ic_mean = ic.mean()
        ic_std = ic.std()
        icir = ic_mean / ic_std if ic_std else np.nan
        t_val = icir * np.sqrt(len(ic)) if ic_std else np.nan
        print(
            f"{name:<8}{valid_rows:>10}{daily_cover:>10.0f}{miss:>9.1%}"
            f"{ic_mean:>10.3f}{icir:>9.2f}{t_val:>8.2f}"
        )

    out_path = DATA_DIR / "factors.pkl"
    pd.to_pickle(factors, out_path)
    print(f"\n因子矩阵已保存: {out_path}")

    sample = next(iter(factors.values()))
    print("\n样例（最后 3 个交易日 × 前 5 只股票）")
    print(sample.iloc[-3:, :5].round(3).to_string())

    print("\n体检结束。IC 均值的绝对值 < 0.02 通常说明因子偏弱；")
    print("|t| > 2 才算统计上显著。样本只有少数股票时 IC 波动会很大，属正常。")


if __name__ == "__main__":
    main()
