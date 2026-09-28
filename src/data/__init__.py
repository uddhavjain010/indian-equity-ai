"""Data provider factory and accessors."""
from typing import Optional
from src.data.base_provider import BaseDataProvider
from src.data.yfinance_provider import YFinanceProvider
from src.data.upstox_provider import UpstoxProvider
from config.settings import AppConfig

def get_data_provider(provider_type: Optional[str] = None) -> BaseDataProvider:
    """Return an instance of the configured data provider.

    Args:
        provider_type: Optional override ('default', 'yfinance', or 'upstox').

    Returns:
        BaseDataProvider instance.
    """
    cfg = AppConfig()
    selected = (provider_type or cfg.provider).lower()

    if selected == "upstox":
        upstox = UpstoxProvider()
        if upstox.is_configured():
            return upstox
        # If not configured, fall back gracefully to YFinance
        return YFinanceProvider()

    return YFinanceProvider()

__all__ = ["BaseDataProvider", "YFinanceProvider", "UpstoxProvider", "get_data_provider"]
