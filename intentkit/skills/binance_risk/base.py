"""Base class for Binance Risk Assessment tools."""

import logging
import math
import statistics
from typing import Any

import httpx
from pydantic import BaseModel, Field

from intentkit.skills.base import IntentKitSkill

BINANCE_FAPI_BASE_URL = "https://fapi.binance.com"
BINANCE_API_BASE_URL = "https://api.binance.com"

logger = logging.getLogger(__name__)


class BinanceRiskBaseTool(IntentKitSkill):
    """Base class for Binance Risk Assessment tools.

    Provides shared utilities for fetching market data from Binance
    public API endpoints and computing risk metrics.
    """

    category: str = "binance_risk"

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 168,
    ) -> list[list[Any]]:
        """Fetch kline/candlestick data from Binance Futures API.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT).
            interval: Kline interval (e.g., 1h, 4h, 1d).
            limit: Number of klines to fetch (max 1500).

        Returns:
            List of kline data arrays.
        """
        url = f"{BINANCE_FAPI_BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": limit}
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_ticker_price(self, symbol: str) -> float:
        """Fetch current mark price for a futures symbol.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT).

        Returns:
            Current mark price as float.
        """
        url = f"{BINANCE_FAPI_BASE_URL}/fapi/v1/premiumIndex"
        params = {"symbol": symbol.upper()}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return float(data["markPrice"])

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        """Fetch current funding rate and next funding time.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT).

        Returns:
            Dict with lastFundingRate, markPrice, nextFundingTime.
        """
        url = f"{BINANCE_FAPI_BASE_URL}/fapi/v1/premiumIndex"
        params = {"symbol": symbol.upper()}
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params=params)
        response.raise_for_status()
        return response.json()

    async def fetch_exchange_info_symbol(self, symbol: str) -> dict[str, Any] | None:
        """Fetch exchange info for a specific futures symbol.

        Args:
            symbol: Trading pair symbol (e.g., BTCUSDT).

        Returns:
            Symbol info dict or None if not found.
        """
        url = f"{BINANCE_FAPI_BASE_URL}/fapi/v1/exchangeInfo"
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
        response.raise_for_status()
        data = response.json()
        for s in data.get("symbols", []):
            if s["symbol"] == symbol.upper():
                return s
        return None

    def compute_volatility(self, klines: list[list[Any]]) -> dict[str, float]:
        """Compute volatility metrics from kline data.

        Uses log returns to calculate annualized volatility, max drawdown,
        and average true range percentage.

        Args:
            klines: List of kline arrays from Binance API.

        Returns:
            Dict with volatility_pct, annualized_volatility_pct,
            max_drawdown_pct, and atr_pct.
        """
        closes = [float(k[4]) for k in klines]
        highs = [float(k[2]) for k in klines]
        lows = [float(k[3]) for k in klines]

        if len(closes) < 2:
            return {
                "volatility_pct": 0.0,
                "annualized_volatility_pct": 0.0,
                "max_drawdown_pct": 0.0,
                "atr_pct": 0.0,
            }

        # Log returns for volatility
        log_returns = []
        for i in range(1, len(closes)):
            if closes[i - 1] > 0:
                log_returns.append(math.log(closes[i] / closes[i - 1]))

        hourly_vol = statistics.stdev(log_returns) if len(log_returns) > 1 else 0.0
        # Annualize assuming hourly data (8760 hours/year)
        annualized_vol = hourly_vol * math.sqrt(8760)

        # Max drawdown
        peak = closes[0]
        max_dd = 0.0
        for price in closes:
            if price > peak:
                peak = price
            dd = (peak - price) / peak if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Average True Range as percentage of price
        atr_values = []
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            atr_values.append(tr)
        avg_atr = statistics.mean(atr_values) if atr_values else 0.0
        atr_pct = (avg_atr / closes[-1] * 100) if closes[-1] > 0 else 0.0

        return {
            "volatility_pct": round(hourly_vol * 100, 4),
            "annualized_volatility_pct": round(annualized_vol * 100, 2),
            "max_drawdown_pct": round(max_dd * 100, 2),
            "atr_pct": round(atr_pct, 4),
        }

    def compute_liquidation_distance(
        self,
        entry_price: float,
        leverage: int,
        side: str,
    ) -> dict[str, float]:
        """Calculate estimated liquidation price and distance.

        Uses simplified formula: maintenance margin ~0.4% for most pairs.

        Args:
            entry_price: Position entry price.
            leverage: Leverage multiplier (e.g., 10).
            side: Position side, 'long' or 'short'.

        Returns:
            Dict with liquidation_price, distance_pct, and distance_usd.
        """
        maintenance_margin_rate = 0.004  # 0.4% typical for major pairs
        if side.lower() == "long":
            liq_price = entry_price * (1 - (1 / leverage) + maintenance_margin_rate)
            distance_pct = ((entry_price - liq_price) / entry_price) * 100
        else:
            liq_price = entry_price * (1 + (1 / leverage) - maintenance_margin_rate)
            distance_pct = ((liq_price - entry_price) / entry_price) * 100

        return {
            "liquidation_price": round(liq_price, 4),
            "distance_pct": round(distance_pct, 2),
            "distance_usd": round(abs(entry_price - liq_price), 4),
        }

    def compute_risk_score(
        self,
        annualized_vol: float,
        funding_rate: float,
        leverage: int,
        liq_distance_pct: float,
    ) -> dict[str, Any]:
        """Compute a composite risk score from 0-100.

        Higher score = higher risk. Components:
        - Volatility risk (0-35): based on annualized volatility
        - Leverage risk (0-30): exponential scaling with leverage
        - Funding risk (0-15): based on funding rate magnitude
        - Liquidation proximity (0-20): inverse of liquidation distance

        Args:
            annualized_vol: Annualized volatility percentage.
            funding_rate: Current funding rate (decimal, e.g., 0.0001).
            leverage: Leverage multiplier.
            liq_distance_pct: Distance to liquidation as percentage.

        Returns:
            Dict with total score, component scores, and risk level.
        """
        # Volatility component (0-35)
        # 50% annualized vol = 17.5, 100% = 35
        vol_score = min(35.0, (annualized_vol / 100.0) * 35.0)

        # Leverage component (0-30)
        # 1x = 0, 10x = 15, 20x = 22.5, 50x = 28.5, 125x = 30
        lev_score = min(30.0, 30.0 * (1 - math.exp(-leverage / 20.0)))

        # Funding rate component (0-15)
        # |0.01%| = 1.5, |0.1%| = 15
        abs_funding = abs(funding_rate) * 100  # convert to percentage
        funding_score = min(15.0, abs_funding * 150.0)

        # Liquidation proximity component (0-20)
        # 1% distance = 20, 5% = 10, 10% = 5, 50%+ = ~0
        if liq_distance_pct > 0:
            liq_score = min(20.0, 20.0 * math.exp(-liq_distance_pct / 5.0))
        else:
            liq_score = 20.0

        total = round(vol_score + lev_score + funding_score + liq_score, 1)

        # Risk level classification
        if total >= 75:
            level = "EXTREME"
        elif total >= 55:
            level = "HIGH"
        elif total >= 35:
            level = "MODERATE"
        elif total >= 15:
            level = "LOW"
        else:
            level = "MINIMAL"

        return {
            "total_score": total,
            "risk_level": level,
            "components": {
                "volatility": round(vol_score, 1),
                "leverage": round(lev_score, 1),
                "funding_rate": round(funding_score, 1),
                "liquidation_proximity": round(liq_score, 1),
            },
        }


# Response models

class RiskAssessmentResult(BaseModel):
    """Full risk assessment result for a trading position."""

    symbol: str = Field(description="Trading pair symbol")
    current_price: float = Field(description="Current mark price")
    position_size_usd: float = Field(description="Position size in USD")
    leverage: int = Field(description="Leverage multiplier")
    side: str = Field(description="Position side (long/short)")
    volatility: dict[str, float] = Field(
        description="Volatility metrics including annualized vol and max drawdown"
    )
    funding_rate: dict[str, Any] = Field(
        description="Current funding rate info and annualized cost"
    )
    liquidation: dict[str, float] = Field(
        description="Liquidation price and distance metrics"
    )
    risk_score: dict[str, Any] = Field(
        description="Composite risk score with component breakdown"
    )
    recommendations: list[str] = Field(
        description="Risk management recommendations"
    )


class PositionSizeResult(BaseModel):
    """Recommended position size based on risk parameters."""

    symbol: str = Field(description="Trading pair symbol")
    current_price: float = Field(description="Current mark price")
    account_balance_usd: float = Field(description="Account balance in USD")
    risk_per_trade_pct: float = Field(description="Risk percentage per trade")
    leverage: int = Field(description="Leverage multiplier")
    stop_loss_pct: float = Field(description="Stop loss percentage")
    recommended_size_usd: float = Field(
        description="Recommended position size in USD"
    )
    recommended_quantity: float = Field(
        description="Recommended quantity of the asset"
    )
    max_loss_usd: float = Field(description="Maximum loss in USD at stop loss")
    risk_reward_note: str = Field(description="Risk/reward context note")
