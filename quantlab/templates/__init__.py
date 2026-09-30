"""內建策略範本（純文字載入，使用者可以在介面上修改）。"""
from pathlib import Path

_DIR = Path(__file__).parent
TEMPLATES = {
    "均線交叉": "ma_cross.py",
    "RSI 均值回歸": "rsi_reversion.py",
    "布林通道均值回歸": "bollinger_reversion.py",
    "唐奇安通道突破（海龜）": "donchian_breakout.py",
    "MACD 趨勢 + 移動停損": "macd_trend.py",
}


def load(name: str) -> str:
    return (_DIR / TEMPLATES[name]).read_text(encoding="utf-8")
