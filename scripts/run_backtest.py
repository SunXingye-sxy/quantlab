"""一键分层回测：五分组 + 调仓周期/成本敏感性 + 出图。

用法：
    python scripts/run_backtest.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from quantlab.backtest import backtest_factor
from quantlab.data import PROJECT_ROOT, filter_untradable, load_panel
from quantlab.factors import build_factors
from quantlab.metrics import summary_table
from quantlab.plots import plot_group_bar, plot_navs

REBALANCE_LIST = (1, 5, 20)
COST_RATE = 0.001


def main() -> None:
    panel = filter_untradable(load_panel())
    factors = build_factors(panel)
    print(f"样本: {panel['code'].nunique()} 只 / {len(panel):,} 条 / "
          f"{panel['date'].nunique()} 个交易日")

    table_dir = PROJECT_ROOT / "results" / "tables"
    fig_dir = PROJECT_ROOT / "results" / "figures"

    rows = []
    for name, factor in factors.items():
        for rb in REBALANCE_LIST:
            res = backtest_factor(
                panel, factor, n_groups=5, cost_rate=COST_RATE, rebalance=rb
            )
            summ = summary_table(res["navs"])
            ls = res["navs"]["多空(高-低)"]
            rows.append({
                "因子": name,
                "调仓周期": f"{rb}日",
                "多空年化": summ.loc["多空(高-低)", "年化收益"],
                "多空夏普": summ.loc["多空(高-低)", "夏普"],
                "多空回撤": summ.loc["多空(高-低)", "最大回撤"],
                "日均换手": res["turnover"]["G5"].mean(),
                "G1年化": summ.loc["G1", "年化收益"],
                "G5年化": summ.loc["G5", "年化收益"],
            })

            if rb == 1:
                summ.to_csv(table_dir / f"backtest_{name}.csv", encoding="utf-8-sig")
                plot_group_bar(summ, fig_dir / f"group_return_{name}.png",
                               title=f"{name} group annual return")
                plot_navs(res["navs"], fig_dir / f"nav_{name}.png",
                          title=f"{name} five groups (rebalance 1d)",
                          emphasize="多空(高-低)")

    sens = pd.DataFrame(rows)
    print("\n========== 调仓周期敏感性（单边成本 0.1%）==========")
    print(sens.to_string(index=False))
    sens.to_csv(table_dir / "rebalance_sensitivity.csv", encoding="utf-8-sig")

    # 多空曲线对比：同一因子在不同调仓周期下
    for name, factor in factors.items():
        ls_navs = {}
        for rb in (1, 20):
            res = backtest_factor(
                panel, factor, n_groups=5, cost_rate=COST_RATE, rebalance=rb
            )
            ls_navs[f"{name} {rb}d"] = res["navs"]["多空(高-低)"]
        plot_navs(ls_navs, fig_dir / f"longshort_{name}.png",
                  title=f"{name} long-short: 1d vs 20d rebalance")

    print(f"\n结果表: {table_dir}\n图表: {fig_dir}")
    print("判断标准：多空夏普 > 0.5 且最大回撤 < 30%，才算值得继续研究的因子。")


if __name__ == "__main__":
    main()
