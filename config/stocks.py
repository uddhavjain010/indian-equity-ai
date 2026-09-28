"""Stock Universe Configuration for Indian Equities (NSE/BSE).

Provides curated Nifty 50 constituents with company metadata, sector mapping,
and multi-exchange ticker resolution.
"""
from typing import Dict, List, Optional, Any

# Curated Nifty 50 Universe (symbol conforms to yfinance NSE format with .NS suffix)
NIFTY_50_UNIVERSE: Dict[str, Dict[str, Any]] = {
    "RELIANCE.NS": {
        "symbol": "RELIANCE.NS",
        "bse_symbol": "500325.BO",
        "company_name": "Reliance Industries Ltd",
        "exchange": "NSE",
        "sector": "Energy & Petrochemicals",
    },
    "TCS.NS": {
        "symbol": "TCS.NS",
        "bse_symbol": "532540.BO",
        "company_name": "Tata Consultancy Services Ltd",
        "exchange": "NSE",
        "sector": "Information Technology",
    },
    "HDFCBANK.NS": {
        "symbol": "HDFCBANK.NS",
        "bse_symbol": "500180.BO",
        "company_name": "HDFC Bank Ltd",
        "exchange": "NSE",
        "sector": "Financial Services (Banking)",
    },
    "INFY.NS": {
        "symbol": "INFY.NS",
        "bse_symbol": "500209.BO",
        "company_name": "Infosys Ltd",
        "exchange": "NSE",
        "sector": "Information Technology",
    },
    "ICICIBANK.NS": {
        "symbol": "ICICIBANK.NS",
        "bse_symbol": "532174.BO",
        "company_name": "ICICI Bank Ltd",
        "exchange": "NSE",
        "sector": "Financial Services (Banking)",
    },
    "HINDUNILVR.NS": {
        "symbol": "HINDUNILVR.NS",
        "bse_symbol": "500696.BO",
        "company_name": "Hindustan Unilever Ltd",
        "exchange": "NSE",
        "sector": "Fast Moving Consumer Goods",
    },
    "ITC.NS": {
        "symbol": "ITC.NS",
        "bse_symbol": "500875.BO",
        "company_name": "ITC Ltd",
        "exchange": "NSE",
        "sector": "Consumer Goods & Conglomerate",
    },
    "SBIN.NS": {
        "symbol": "SBIN.NS",
        "bse_symbol": "500112.BO",
        "company_name": "State Bank of India",
        "exchange": "NSE",
        "sector": "Financial Services (Public Banking)",
    },
    "BHARTIARTL.NS": {
        "symbol": "BHARTIARTL.NS",
        "bse_symbol": "532454.BO",
        "company_name": "Bharti Airtel Ltd",
        "exchange": "NSE",
        "sector": "Telecommunications",
    },
    "LT.NS": {
        "symbol": "LT.NS",
        "bse_symbol": "500510.BO",
        "company_name": "Larsen & Toubro Ltd",
        "exchange": "NSE",
        "sector": "Engineering & Construction",
    },
    "KOTAKBANK.NS": {
        "symbol": "KOTAKBANK.NS",
        "bse_symbol": "500247.BO",
        "company_name": "Kotak Mahindra Bank Ltd",
        "exchange": "NSE",
        "sector": "Financial Services (Banking)",
    },
    "AXISBANK.NS": {
        "symbol": "AXISBANK.NS",
        "bse_symbol": "532215.BO",
        "company_name": "Axis Bank Ltd",
        "exchange": "NSE",
        "sector": "Financial Services (Banking)",
    },
    "BAJFINANCE.NS": {
        "symbol": "BAJFINANCE.NS",
        "bse_symbol": "500034.BO",
        "company_name": "Bajaj Finance Ltd",
        "exchange": "NSE",
        "sector": "Financial Services (NBFC)",
    },
    "MARUTI.NS": {
        "symbol": "MARUTI.NS",
        "bse_symbol": "532500.BO",
        "company_name": "Maruti Suzuki India Ltd",
        "exchange": "NSE",
        "sector": "Automobile",
    },
    "TATAMOTORS.NS": {
        "symbol": "TATAMOTORS.NS",
        "bse_symbol": "500570.BO",
        "company_name": "Tata Motors Ltd",
        "exchange": "NSE",
        "sector": "Automobile",
    },
    "SUNPHARMA.NS": {
        "symbol": "SUNPHARMA.NS",
        "bse_symbol": "524715.BO",
        "company_name": "Sun Pharmaceutical Industries Ltd",
        "exchange": "NSE",
        "sector": "Healthcare & Pharmaceuticals",
    },
    "TITAN.NS": {
        "symbol": "TITAN.NS",
        "bse_symbol": "500114.BO",
        "company_name": "Titan Company Ltd",
        "exchange": "NSE",
        "sector": "Consumer Durables",
    },
    "ASIANPAINT.NS": {
        "symbol": "ASIANPAINT.NS",
        "bse_symbol": "500820.BO",
        "company_name": "Asian Paints Ltd",
        "exchange": "NSE",
        "sector": "Paints & Chemicals",
    },
    "NTPC.NS": {
        "symbol": "NTPC.NS",
        "bse_symbol": "532555.BO",
        "company_name": "NTPC Ltd",
        "exchange": "NSE",
        "sector": "Power & Utilities",
    },
    "POWERGRID.NS": {
        "symbol": "POWERGRID.NS",
        "bse_symbol": "532898.BO",
        "company_name": "Power Grid Corporation of India Ltd",
        "exchange": "NSE",
        "sector": "Power & Utilities",
    },
    "TATASTEEL.NS": {
        "symbol": "TATASTEEL.NS",
        "bse_symbol": "500470.BO",
        "company_name": "Tata Steel Ltd",
        "exchange": "NSE",
        "sector": "Metals & Mining",
    },
    "JSWSTEEL.NS": {
        "symbol": "JSWSTEEL.NS",
        "bse_symbol": "500228.BO",
        "company_name": "JSW Steel Ltd",
        "exchange": "NSE",
        "sector": "Metals & Mining",
    },
    "COALINDIA.NS": {
        "symbol": "COALINDIA.NS",
        "bse_symbol": "533278.BO",
        "company_name": "Coal India Ltd",
        "exchange": "NSE",
        "sector": "Mining & Energy",
    },
    "ADANIENT.NS": {
        "symbol": "ADANIENT.NS",
        "bse_symbol": "512599.BO",
        "company_name": "Adani Enterprises Ltd",
        "exchange": "NSE",
        "sector": "Metals, Mining & Conglomerate",
    },
    "ADANIPORTS.NS": {
        "symbol": "ADANIPORTS.NS",
        "bse_symbol": "532921.BO",
        "company_name": "Adani Ports & Special Economic Zone Ltd",
        "exchange": "NSE",
        "sector": "Infrastructure & Logistics",
    },
    "WIPRO.NS": {
        "symbol": "WIPRO.NS",
        "bse_symbol": "507685.BO",
        "company_name": "Wipro Ltd",
        "exchange": "NSE",
        "sector": "Information Technology",
    },
    "HCLTECH.NS": {
        "symbol": "HCLTECH.NS",
        "bse_symbol": "532281.BO",
        "company_name": "HCL Technologies Ltd",
        "exchange": "NSE",
        "sector": "Information Technology",
    },
    "ULTRACEMCO.NS": {
        "symbol": "ULTRACEMCO.NS",
        "bse_symbol": "532538.BO",
        "company_name": "UltraTech Cement Ltd",
        "exchange": "NSE",
        "sector": "Materials & Cement",
    },
    "M&M.NS": {
        "symbol": "M&M.NS",
        "bse_symbol": "500520.BO",
        "company_name": "Mahindra & Mahindra Ltd",
        "exchange": "NSE",
        "sector": "Automobile",
    },
    "BAJAJFINSV.NS": {
        "symbol": "BAJAJFINSV.NS",
        "bse_symbol": "532978.BO",
        "company_name": "Bajaj Finserv Ltd",
        "exchange": "NSE",
        "sector": "Financial Services",
    },
    "ONGC.NS": {
        "symbol": "ONGC.NS",
        "bse_symbol": "500312.BO",
        "company_name": "Oil & Natural Gas Corporation Ltd",
        "exchange": "NSE",
        "sector": "Energy & Exploration",
    },
    "NESTLEIND.NS": {
        "symbol": "NESTLEIND.NS",
        "bse_symbol": "500790.BO",
        "company_name": "Nestle India Ltd",
        "exchange": "NSE",
        "sector": "Consumer Goods (FMCG)",
    },
}

def get_stock_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Retrieve metadata for a given symbol (NSE or BSE symbol format)."""
    sym = symbol.strip().upper()
    if sym in NIFTY_50_UNIVERSE:
        return NIFTY_50_UNIVERSE[sym]
    # Check if BSE symbol or symbol without extension was passed
    for nse_sym, data in NIFTY_50_UNIVERSE.items():
        if data.get("bse_symbol") == sym or nse_sym.split(".")[0] == sym.split(".")[0]:
            return data
    return None

def get_all_symbols() -> List[str]:
    """Return all configured NSE stock symbols."""
    return list(NIFTY_50_UNIVERSE.keys())

def get_display_name(symbol: str) -> str:
    """Return formatted display name for UI dropdowns."""
    info = get_stock_info(symbol)
    if info:
        return f"{info['symbol']} — {info['company_name']} ({info['sector']})"
    return symbol
