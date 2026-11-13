"""数据体检：在写任何因子之前，先确认数据是干净的。

真实研究里，80% 的错误来自数据而不是策略：
未来函数、复权错、停牌未剔、重复行、异常收益……

用法
----
    python scripts/check_data.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from quantlab.data import filter_untradable, load_panel, to_wide  # noqa: E402

# 不同板块的涨跌停幅度不同，用统一阈值会把正常波动误判成异常
BOARD_LIMIT = {
    "沪市主板": 10.5,
    "深市主板": 10.5,
    "创业板": 20.5,
    "科创板": 20.5,
    "其他": 21.0,
}


def classify_board(code: str) -> str:
    """按股票代码前缀识别所属板块。"""
    p = code[:3]
    if p in ("600", "601", "603", "605"):
        return "沪市主板"
    if p in ("688", "689"):
        return "科创板"
    if p in ("300", "301"):
        return "创业板"
    if p in ("000", "001", "002", "003"):
        return "深市主板"
    return "其他"


def main() -> None:
    panel = load_panel()
    print("=" * 46)
    print("数据体检报告")
    print("=" * 46)

    # 1. 规模
    n_rows, n_stocks = len(panel), panel["code"].nunique()
    n_dates = panel["date"].nunique()
    print(f"记录数      : {n_rows:,}")
    print(f"股票数      : {n_stocks}")
    print(f"交易日数    : {n_dates}")
    print(f"日期范围    : {panel['date'].min().date()} ~ {panel['date'].max().date()}")

    # 2. 重复行（主键冲突会让回测结果虚高）
    dup = panel.duplicated(["date", "code"]).sum()
    print(f"重复记录    : {dup}")

    # 3. 缺失率（停牌、未上市、接口缺数据都会体现为空值）
    miss = panel[["close", "volume"]].isna().mean()
    print(f"收盘价缺失率: {miss['close']:.2%}")
    print(f"成交量缺失率: {miss['volume']:.2%}")

    # 4. 样本覆盖率：实际记录数 / 理论满格（交易日 × 股票数）
    #    停牌日往往直接缺行而不是记 0 成交量，所以覆盖率比"零成交量"更能说明问题
    coverage = n_rows / (n_dates * n_stocks)
    print(f"样本覆盖率  : {coverage:.1%}  （缺失部分=停牌/退市/上市晚，正常）")

    zero_vol = (panel["volume"] == 0).sum()
    print(f"零成交量记录: {zero_vol:,}  （若为 0 说明数据源直接省略停牌行）")

    # 5. 板块分布 + 分板块极端收益检查
    panel["board"] = panel["code"].map(classify_board)
    print("\n板块分布:")
    for board, cnt in panel["board"].value_counts().items():
        print(f"  {board:<6}: {cnt:,} 条 / {panel.loc[panel['board'] == board, 'code'].nunique()} 只")

    print("\n分板块涨跌幅异常检查（各板块按自身涨跌停幅度判断）:")
    abnormal_total = 0
    for board, limit in BOARD_LIMIT.items():
        sub = panel[panel["board"] == board]["pct_chg"].dropna()
        if sub.empty:
            continue
        n_bad = (sub.abs() > limit).sum()
        abnormal_total += n_bad
        print(f"  {board:<6}: |涨跌幅|>{limit}% 共 {n_bad:,} 条（{n_bad / len(sub):.3%}）")

    if abnormal_total == 0:
        print("  -> 无越界样本，复权处理正确。")
    else:
        print(f"  -> 合计 {abnormal_total:,} 条越界。若占比很低（<0.5%）多为新股上市首日，正常；")
        print("     若某板块占比明显偏高，检查该板块是否误用未复权数据。")

    # 6. 每个交易日实际可交易股票数（看样本是否随时间剧烈变化）
    daily = panel.groupby("date")["code"].nunique()
    print(f"\n每日股票数  : 中位数 {int(daily.median())}，最少 {int(daily.min())}（{daily.idxmin().date()}）")

    # 7. 剔除停牌后的有效样本
    clean = filter_untradable(panel.drop(columns=["board"]))
    print(f"剔除停牌后  : {len(clean):,} 条（保留 {len(clean) / max(len(panel), 1):.1%}）")

    # 8. 抽样预览
    print("\n样例（最近 5 个交易日，前 3 只股票）:")
    recent = panel.sort_values("date").tail(5 * 3)
    cols = ["date", "code", "close", "volume", "pct_chg"]
    print(recent[cols].head(10).to_string(index=False))

    # 9. 宽表维度自检（后面所有因子计算都基于宽表）
    close_wide = to_wide(clean, "close")
    print(f"\n收盘价宽表  : {close_wide.shape[0]} 个交易日 × {close_wide.shape[1]} 只股票")

    print("=" * 46)
    print("体检结束。重点关注：覆盖率、分板块越界占比、每日股票数是否平稳。")


if __name__ == "__main__":
    main()
