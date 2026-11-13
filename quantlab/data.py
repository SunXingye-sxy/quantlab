"""数据管道：A 股日线行情与指数成分股的下载、缓存、加载。

设计原则
--------
1. 一次下载，落盘缓存（parquet），后续分析全部从本地读取，避免重复请求接口。
2. 列名统一成英文小写，日期统一为 datetime64，方便后续对齐与合并。
3. 单只股票下载失败不影响整体，失败清单最后统一打印，可重跑补齐。
4. 数据文件写在 data/ 目录下，且已被 .gitignore 排除，绝不进版本库。

数据源
------
akshare（免费、无需 token）。接口文档：https://akshare.akfamily.xyz/

注意：akshare 的接口字段偶有变动。本文件对关键字段做了兼容处理，
若某天报列名错误，先看 akshare 版本（pip show akshare），再对照官方文档。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

# 项目根目录（本文件位于 <root>/quantlab/data.py）
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_START = "20180101"
DEFAULT_END = "20241231"

# akshare 返回的中文列名 -> 统一英文列名
COLUMN_MAP = {
    "日期": "date",
    "股票代码": "code",
    "开盘": "open",
    "收盘": "close",
    "最高": "high",
    "最低": "low",
    "成交量": "volume",
    "成交额": "amount",
    "换手率": "turnover",
    "涨跌幅": "pct_chg",
}

PANEL_COLUMNS = [
    "date", "code", "open", "high", "low", "close",
    "volume", "amount", "turnover", "pct_chg",
]


def _parquet_available() -> bool:
    """检测能否读写 parquet（依赖 pyarrow / fastparquet）。

    部分环境下 pyarrow 与 numpy 版本冲突，装不上。此时自动降级为 pickle，
    保证「数据能落地、能读回」这一核心功能不受影响。
    """
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


USE_PARQUET = _parquet_available()
CACHE_SUFFIX = ".parquet" if USE_PARQUET else ".pkl"


def _cache_path(code: str) -> Path:
    return RAW_DIR / f"{code}{CACHE_SUFFIX}"


def _save_frame(df: pd.DataFrame, path: Path) -> None:
    if path.suffix == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_pickle(path)


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_pickle(path)


def _import_akshare():
    """延迟导入 akshare，避免仅 import quantlab 时就加载重依赖。"""
    try:
        import akshare as ak
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "缺少 akshare，请先执行：pip install -r requirements.txt"
        ) from exc
    return ak


def get_index_constituents(index_code: str = "000300") -> pd.DataFrame:
    """获取指数成分股列表。

    Parameters
    ----------
    index_code : str
        指数代码，000300 = 沪深300，000905 = 中证500，000852 = 中证1000。

    Returns
    -------
    DataFrame
        列：code（6 位股票代码）、name（股票名称）。
    """
    ak = _import_akshare()

    df = None
    # 首选：中证指数官网接口（字段稳定）
    try:
        df = ak.index_stock_cons_csindex(symbol=index_code)
    except Exception:
        df = None

    if df is not None and not df.empty:
        code_col = "成分券代码" if "成分券代码" in df.columns else None
        name_col = "成分券名称" if "成分券名称" in df.columns else None
        if code_col:
            out = pd.DataFrame({
                "code": df[code_col].astype(str).str.zfill(6),
                "name": df[name_col].astype(str) if name_col else "",
            })
            return out.drop_duplicates("code").reset_index(drop=True)

    # 备用：新浪接口
    df = ak.index_stock_cons(symbol=index_code)
    code_col = "品种代码" if "品种代码" in df.columns else df.columns[0]
    name_col = "品种名称" if "品种名称" in df.columns else df.columns[1]
    out = pd.DataFrame({
        "code": df[code_col].astype(str).str.zfill(6),
        "name": df[name_col].astype(str),
    })
    return out.drop_duplicates("code").reset_index(drop=True)


def get_trade_calendar(start: str = DEFAULT_START, end: str = DEFAULT_END) -> pd.DatetimeIndex:
    """获取交易日历（A 股实际开市日）。

    用真实交易日历而不是自然日，能避免把周末、节假日当成「数据缺失」。
    """
    ak = _import_akshare()
    df = ak.tool_trade_date_hist_sina()
    col = "trade_date" if "trade_date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[col])
    mask = (dates >= pd.to_datetime(start)) & (dates <= pd.to_datetime(end))
    return pd.DatetimeIndex(dates[mask].sort_values())


# 东财（push2）接口一旦被限流，后续请求会持续失败，
# 用一个模块级标志切到备用源，避免每只股票都白等一轮重试。
_EASTMONEY_BLOCKED = False


def _to_sina_symbol(code: str) -> str:
    """把 6 位代码转成新浪格式：6/9 开头 -> sh，0/3 开头 -> sz，其他 -> bj。"""
    if code.startswith(("6", "9")):
        return f"sh{code}"
    if code.startswith(("0", "3")):
        return f"sz{code}"
    return f"bj{code}"


def _download_eastmoney(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """主数据源：东方财富（ak.stock_zh_a_hist）。"""
    ak = _import_akshare()
    raw = ak.stock_zh_a_hist(
        symbol=code, period="daily", start_date=start, end_date=end, adjust=adjust
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = raw.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in raw.columns})
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])

    for col in PANEL_COLUMNS:  # 接口字段变动时补空列，保证结构一致
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PANEL_COLUMNS]
    for col in PANEL_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # 关键：不使用接口给的涨跌幅，改用前复权收盘价重算。
    # 接口字段按「未调整除权」的价格计算，高分红/送转股票在除息日会出现
    # 超过涨跌停幅度的"假暴跌"（实测 601225 陕西煤业有 484 个这样的日期），
    # 而前复权后的价格序列本身是连续的，两者口径不一致会污染因子和回测。
    df["pct_chg"] = df["close"].pct_change() * 100
    df["pct_chg"] = df["pct_chg"].fillna(0.0)
    return df


def _download_sina(code: str, start: str, end: str, adjust: str) -> pd.DataFrame:
    """备用数据源：新浪（ak.stock_zh_a_daily）。

    与东财的差别：turnover 是小数（需乘 100），没有涨跌幅字段（需自己算）。
    """
    ak = _import_akshare()
    raw = ak.stock_zh_a_daily(
        symbol=_to_sina_symbol(code), start_date=start, end_date=end, adjust=adjust
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=PANEL_COLUMNS)

    df = raw.copy()
    df["code"] = code
    df["date"] = pd.to_datetime(df["date"])
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") * 100  # 小数 -> %
    for col in PANEL_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    df = df[PANEL_COLUMNS]
    for col in PANEL_COLUMNS[2:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)

    # 与东财分支统一口径：用前复权收盘价重算涨跌幅
    df["pct_chg"] = df["close"].pct_change() * 100
    df["pct_chg"] = df["pct_chg"].fillna(0.0)
    return df


def download_stock_daily(
    code: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    adjust: str = "qfq",
    retries: int = 3,
    pause: float = 0.5,
) -> pd.DataFrame:
    """下载单只股票日线行情并统一列名，新浪为主源、东财为后备。

    为什么新浪优先：东财的前复权（qfq）序列在部分高分红/送转标的上
    复权因子断裂，会产生大量超过涨跌停的"假暴跌"（实测 600039 四川路桥
    386 天、601088 中国神华 309 天），而新浪的同标的前复权序列连续，
    越界 0 条。这是数据源的质量问题，靠事后计算补不回来。

    Parameters
    ----------
    adjust : {"qfq", "hfq", ""}
        qfq = 前复权（推荐，价格连续、便于计算收益率）；
        hfq = 后复权；"" = 不复权（有除权跳空，不要用）。

    Returns
    -------
    DataFrame
        列见 PANEL_COLUMNS，按日期升序；无数据时返回空表。
    """
    global _EASTMONEY_BLOCKED
    last_err: Exception | None = None

    # 1) 主源：新浪
    for attempt in range(retries):
        try:
            df = _download_sina(code, start, end, adjust)
            if df is not None and not df.empty:
                return df
        except Exception as exc:
            last_err = exc
            time.sleep(pause * (attempt + 1))

    # 2) 备源：东方财富（新浪缺该标的或无数据时兜底）
    if not _EASTMONEY_BLOCKED:
        for attempt in range(retries):
            try:
                return _download_eastmoney(code, start, end, adjust)
            except Exception as exc:
                last_err = exc
                msg = f"{type(exc).__name__} {exc}"
                if "限流" in msg or "push2" in msg.lower() or "ProxyError" in msg:
                    _EASTMONEY_BLOCKED = True
                    print("[提示] 东财接口被限流，跳过备用源。")
                    break
                time.sleep(pause * (attempt + 1))

    raise RuntimeError(f"{code} 下载失败（新浪+东财均不可用）：{last_err}")


def download_panel(
    codes: list[str],
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    adjust: str = "qfq",
    overwrite: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    """批量下载并落地缓存，返回合并后的面板数据与失败清单。

    每只股票单独存一个 parquet（data/raw/<code>.parquet），
    这样中断后重跑只补缺失的部分，不必从头来过。
    """
    failed: list[str] = []
    frames: list[pd.DataFrame] = []

    iterator = codes
    if verbose:
        try:
            from tqdm import tqdm
            iterator = tqdm(codes, desc="下载进度", ncols=80)
        except ImportError:
            pass

    for code in iterator:
        cache = _cache_path(code)
        if cache.exists() and not overwrite:
            frames.append(_read_frame(cache))
            continue
        try:
            df = download_stock_daily(code, start=start, end=end, adjust=adjust)
            if df.empty:
                failed.append(code)
                continue
            _save_frame(df, cache)
            frames.append(df)
        except Exception:
            failed.append(code)

    if not frames:
        return pd.DataFrame(columns=PANEL_COLUMNS), failed

    panel = pd.concat(frames, ignore_index=True)
    panel = panel.drop_duplicates(["date", "code"]).sort_values(["date", "code"])
    _save_frame(panel, DATA_DIR / f"panel{CACHE_SUFFIX}")
    return panel.reset_index(drop=True), failed


def load_panel() -> pd.DataFrame:
    """从本地缓存加载面板数据（长表：date × code）。

    .gitignore 排除了 data/，所以这份数据只存在于你本机；
    换电脑后需要重新执行 scripts/download_data.py 生成。
    """
    path = DATA_DIR / f"panel{CACHE_SUFFIX}"
    if not path.exists():
        raise FileNotFoundError(
            f"未找到 {path}，请先运行：python scripts/download_data.py"
        )
    return _read_frame(path)


def to_wide(panel: pd.DataFrame, field: str = "close") -> pd.DataFrame:
    """把长表转成宽表（index=date, columns=code），便于向量化计算。"""
    wide = panel.pivot(index="date", columns="code", values=field)
    return wide.sort_index()


def filter_untradable(panel: pd.DataFrame, min_volume: float = 0.0) -> pd.DataFrame:
    """剔除停牌与无效交易日。

    A 股停牌时接口有时仍返回一行（成交量为 0 / 价格前值），
    这类样本若进入因子计算会制造虚假的 0 收益，必须剔除。
    """
    df = panel.copy()
    df = df[df["volume"] > min_volume]
    df = df[df["close"].notna()]
    return df
