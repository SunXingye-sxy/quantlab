"""图表输出：净值曲线、分组收益对比。

标签一律用英文。原因：matplotlib 默认字体（DejaVu Sans）不含中文字形，
直接写中文会在图上显示成一堆方块，还会刷一屏 "Glyph missing" 警告。
指定系统字体虽然能修好，但每台机器字体名不同（Windows 是 Microsoft YaHei，
macOS 是 PingFang，Linux 不一定有），为了几个标签去脆脆地依赖系统字体不值得。
图是给自己和面试官看的，英文足够。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境也能出图
import matplotlib.pyplot as plt
import pandas as pd

# 回测代码里多空组合叫「多空(高-低)」，画图时统一转成英文
LABEL_MAP = {"多空(高-低)": "Long-Short"}


def _en(label: str) -> str:
    return LABEL_MAP.get(label, label)


def plot_navs(
    navs: dict[str, pd.Series],
    path: Path,
    title: str = "Cumulative NAV",
    emphasize: str | None = None,
) -> Path:
    """画多条净值曲线。emphasize 指定的那条加粗、加深。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 5.5))

    for name, nav in navs.items():
        if name == emphasize:
            ax.plot(nav.index, nav.values, label=_en(name),
                    lw=2.0, color="#c0392b", zorder=5)
        else:
            ax.plot(nav.index, nav.values, label=_en(name), lw=1.0, alpha=0.75)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("NAV (start = 1.0)")
    ax.axhline(1.0, color="grey", lw=0.8, ls="--")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_group_bar(
    summary: pd.DataFrame,
    path: Path,
    column: str = "年化收益",
    title: str = "Annual return by group",
) -> Path:
    """把五组的分组绩效画成柱状图，直观看出单调性。

    只画 G1..Gn 各组，跳过「多空」行（它是派生指标，混在柱状图里会误解）。
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))

    values = summary[column].astype(float)
    values = values[~values.index.astype(str).str.contains("多空")]
    colors = ["#c0392b" if v < 0 else "#2980b9" for v in values]

    ax.bar(values.index.astype(str), values.values, color=colors)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"{title} ({column})")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path
