"""深色交易終端風格：配色、CSS、Plotly 樣式、KPI 卡片。"""
from __future__ import annotations

import html

import plotly.graph_objects as go

# 深色背景下驗證過的類別色（依固定順序使用，不循環）
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9", "#e66767", "#008300"]
BLUE, ORANGE = SERIES[0], SERIES[1]
UP, DOWN = "#0ca30c", "#e66767"              # 漲 / 跌（搭配 ▲▼ 符號，不只靠顏色）
INK, INK2, MUTED = "#f2f1ec", "#c3c2b7", "#898781"
GRID, AXIS = "#2c2c2a", "#383835"
SURFACE, PAGE = "#161615", "#0d0d0d"
DIVERGING = [[0.0, "#e66767"], [0.5, "#383835"], [1.0, "#3987e5"]]   # 紅 ↔ 灰 ↔ 藍

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;600&family=Noto+Sans+TC:wght@400;500;700&display=swap');

html, body, .stApp, [data-testid="stSidebar"], button, input, textarea, select {
  font-family: 'Inter', 'Noto Sans TC', -apple-system, sans-serif !important;
}
.stApp { background: #0d0d0d; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"], footer, #MainMenu { display: none !important; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

/* 側欄 */
[data-testid="stSidebar"] { background: #121211; border-right: 1px solid #242422; }
[data-testid="stSidebar"] h3 {
  font-size: .72rem; letter-spacing: .12em; text-transform: uppercase;
  color: #898781; font-weight: 600; margin: 1.2rem 0 .3rem;
}

/* 頂部品牌列 */
.topbar { display:flex; align-items:baseline; gap:.9rem; margin-bottom:.4rem; flex-wrap:wrap; }
.brand { font-weight:700; font-size:1.35rem; letter-spacing:.06em; color:#f2f1ec; }
.brand b { color:#3987e5; font-weight:700; }
.topbar .sub { color:#898781; font-size:.85rem; }
.page-title { font-size:1.6rem; font-weight:600; color:#f2f1ec; margin:.2rem 0 .1rem; }
.page-ref { color:#898781; font-size:.85rem; margin-bottom:1rem; }
.data-chip {
  display:inline-block; font-family:'JetBrains Mono', monospace; font-size:.75rem; color:#c3c2b7;
  background:#1a1a19; border:1px solid #2c2c2a; border-radius:999px; padding:.15rem .6rem; margin:0 .3rem .3rem 0;
}

/* KPI 卡片 */
.kpi-grid { display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:.75rem; margin:.4rem 0 1rem; }
.kpi { background:#161615; border:1px solid #2a2a28; border-radius:12px; padding:.85rem 1rem .8rem; }
.kpi-label { color:#898781; font-size:.72rem; letter-spacing:.06em; text-transform:uppercase; margin-bottom:.35rem; }
.kpi-value { font-family:'JetBrains Mono', monospace; font-size:1.55rem; font-weight:600; color:#f2f1ec; line-height:1.15; }
.kpi-delta { font-family:'JetBrains Mono', monospace; font-size:.78rem; margin-top:.3rem; color:#898781; }
.kpi-delta.up { color:#0ca30c; } .kpi-delta.down { color:#e66767; }

/* 狀態列 */
.status { border-radius:10px; padding:.55rem .9rem; font-size:.85rem; margin:0 0 1rem; border:1px solid; }
.status.ok { background:rgba(12,163,12,.08); border-color:rgba(12,163,12,.35); color:#8fdc8f; }
.status.bad { background:rgba(230,103,103,.08); border-color:rgba(230,103,103,.4); color:#f0a3a3; }

/* 控制面板、分頁 */
[data-testid="stVerticalBlockBorderWrapper"] { border-color:#2a2a28 !important; border-radius:12px !important; background:#121211; }
button[data-baseweb="tab"] { font-size:.9rem; padding:.5rem .2rem; }
button[data-baseweb="tab"][aria-selected="true"] { color:#f2f1ec; }
[data-baseweb="tab-highlight"] { background:#3987e5 !important; }
[data-baseweb="tab-border"] { background:#242422 !important; }
[data-testid="stExpander"] details { border-color:#2a2a28; border-radius:12px; }
[data-testid="stDataFrame"] { border:1px solid #2a2a28; border-radius:10px; }
.hint { color:#898781; font-size:.82rem; line-height:1.6; }
</style>
"""


def esc(s) -> str:
    return html.escape(str(s))


def kpi_cards(items: list[dict]) -> str:
    """items: [{label, value, delta?, trend? ('up'|'down'|None)}]"""
    cards = []
    for it in items:
        delta = ""
        if it.get("delta"):
            trend = it.get("trend")
            icon = "▲ " if trend == "up" else "▼ " if trend == "down" else ""
            delta = f'<div class="kpi-delta {trend or ""}">{icon}{esc(it["delta"])}</div>'
        cards.append(f'<div class="kpi"><div class="kpi-label">{esc(it["label"])}</div>'
                     f'<div class="kpi-value">{esc(it["value"])}</div>{delta}</div>')
    return f'<div class="kpi-grid">{"".join(cards)}</div>'


def status(ok: bool, text: str) -> str:
    return f'<div class="status {"ok" if ok else "bad"}">{"✓" if ok else "✕"}&nbsp; {esc(text)}</div>'


def chips(texts: list[str]) -> str:
    return "".join(f'<span class="data-chip">{esc(t)}</span>' for t in texts)


def style(fig: go.Figure, height: int = 360, title: str | None = None, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, template="plotly_dark",
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
        font=dict(family="Inter, Noto Sans TC, sans-serif", color=INK2, size=12),
        title=dict(text=title, font=dict(size=14, color=INK), x=0.01, y=0.97) if title else None,
        margin=dict(l=12, r=12, t=48 if title else 16, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="#222221", bordercolor=AXIS, font=dict(color=INK, family="JetBrains Mono, monospace")),
        showlegend=legend,
        legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom",
                    bgcolor="rgba(0,0,0,0)", font=dict(color=INK2)),
    )
    fig.update_xaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, zerolinecolor=AXIS, tickfont=dict(color=MUTED))
    return fig
