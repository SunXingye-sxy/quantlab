"""一键下载数据：指数成分股 + 成分股日线行情，落地到 data/。

用法
----
    python scripts/download_data.py                 # 默认沪深300，2018-2024
    python scripts/download_data.py --index 000905  # 换中证500
    python scripts/download_data.py --limit 20      # 只下前20只，先试通链路
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 让脚本可以直接运行（把项目根目录加入 sys.path）
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from quantlab.data import (  # noqa: E402
    CACHE_SUFFIX,
    DEFAULT_END,
    DEFAULT_START,
    download_panel,
    get_index_constituents,
    get_trade_calendar,
    load_panel,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下载 A 股行情数据")
    parser.add_argument("--index", default="000300", help="指数代码，默认 000300（沪深300）")
    parser.add_argument("--start", default=DEFAULT_START, help="开始日期 YYYYMMDD")
    parser.add_argument("--end", default=DEFAULT_END, help="结束日期 YYYYMMDD")
    parser.add_argument("--limit", type=int, default=0, help="只下载前 N 只（调试用，0=全部）")
    parser.add_argument("--overwrite", action="store_true", help="忽略本地缓存，强制重新下载")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[1/4] 获取指数 {args.index} 成分股 ...")
    cons = get_index_constituents(args.index)
    print(f"      成分股数量：{len(cons)}")

    codes = cons["code"].tolist()
    if args.limit > 0:
        codes = codes[: args.limit]
        print(f"      调试模式：只下载前 {args.limit} 只")

    print(f"[2/4] 获取交易日历 {args.start} ~ {args.end} ...")
    calendar = get_trade_calendar(args.start, args.end)
    print(f"      交易日数量：{len(calendar)}")

    print(f"[3/4] 下载日线行情（前复权）...")
    panel, failed = download_panel(
        codes, start=args.start, end=args.end, overwrite=args.overwrite
    )

    print(f"[4/4] 汇总 ...")
    if failed:
        print(f"      警告：{len(failed)} 只下载失败：{failed[:10]}{' ...' if len(failed) > 10 else ''}")

    if panel.empty:
        print("      未获得任何数据，请检查网络或 akshare 版本。")
        return

    panel = load_panel() if (args.limit == 0 and not failed) else panel

    n_stocks = panel["code"].nunique()
    d0, d1 = panel["date"].min().date(), panel["date"].max().date()
    print("\n========== 下载完成 ==========")
    print(f"股票数    : {n_stocks}")
    print(f"日期范围  : {d0} ~ {d1}")
    print(f"交易日数  : {panel['date'].nunique()}")
    print(f"总记录数  : {len(panel):,}")
    print(f"数据落盘  : data/panel{CACHE_SUFFIX}")
    print("==============================")
    print("\n下一步：python scripts/check_data.py   （做数据体检）")


if __name__ == "__main__":
    main()
