"""Skill for calculating optimal position size based on risk parameters."""

import logging

from langchain_core.tools import ArgsSchema
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from intentkit.skills.binance_risk.base import BinanceRiskBaseTool, PositionSizeResult

logger = logging.getLogger(__name__)


class CalculatePositionSizeInput(BaseModel):
    """Input for CalculatePositionSize tool."""

    symbol: str = Field(
        ...,
        description="Binance Futures trading pair (e.g., BTCUSDT, ETHUSDT)",
    )
    account_balance_usd: float = Field(
        ...,
        description="Total account balance in USD",
        gt=0,
    )
    risk_per_trade_pct: float = Field(
        default=1.0,
        description="Percentage of account to risk per trade (e.g., 1.0 for 1%)",
        gt=0,
        le=100,
    )
    leverage: int = Field(
        default=1,
        description="Leverage multiplier (1-125, default 1)",
        ge=1,
        le=125,
    )
    stop_loss_pct: float = Field(
        default=0.0,
        description=(
            "Stop loss distance as percentage from entry (e.g., 2.0 for 2%). "
            "If 0, an ATR-based stop loss will be suggested automatically."
        ),
        ge=0,
        le=100,
    )


class CalculatePositionSize(BinanceRiskBaseTool):
    """Calculate the optimal position size for a Binance Futures trade.

    Uses the fixed-percentage risk model: given an account balance, risk
    tolerance, leverage, and stop loss distance, computes the maximum
    position size that limits losses to the specified risk amount.

    If no stop loss is provided, automatically suggests one based on
    the asset's recent Average True Range (ATR).
    """

    name: str = "binance_risk_calculate_size"
    description: str = (
        "Calculate optimal position size for a Binance Futures trade. "
        "Takes symbol, account balance, risk percentage per trade, "
        "leverage, and optional stop loss percentage. Returns recommended "
        "position size in USD and quantity, with the maximum loss amount. "
        "If no stop loss is given, suggests one based on recent volatility."
    )
    args_schema: ArgsSchema | None = CalculatePositionSizeInput

    async def _arun(
        self,
        symbol: str,
        account_balance_usd: float,
        risk_per_trade_pct: float = 1.0,
        leverage: int = 1,
        stop_loss_pct: float = 0.0,
        **kwargs,
    ) -> PositionSizeResult:
        """Calculate recommended position size.

        Args:
            symbol: Binance Futures trading pair.
            account_balance_usd: Total account balance in USD.
            risk_per_trade_pct: Max risk per trade as percentage of balance.
            leverage: Leverage multiplier.
            stop_loss_pct: Stop loss distance in percent. 0 = auto-suggest.

        Returns:
            PositionSizeResult with sizing recommendation.
        """
        context = self.get_context()

        # Rate limit: 10 calculations per minute per user
        await self.user_rate_limit_by_skill(10, 60)

        symbol = symbol.upper()

        # Fetch current price
        current_price = await self.fetch_ticker_price(symbol)

        # If no stop loss provided, compute from ATR
        if stop_loss_pct <= 0:
            klines = await self.fetch_klines(symbol, interval="1h", limit=168)
            volatility = self.compute_volatility(klines)
            # Use 2x ATR as a reasonable stop loss
            stop_loss_pct = max(volatility["atr_pct"] * 2, 0.5)
            auto_sl = True
        else:
            auto_sl = False

        # Fixed-percentage position sizing formula:
        # max_loss = account_balance * (risk_per_trade_pct / 100)
        # position_size = max_loss / (stop_loss_pct / 100)
        max_loss_usd = account_balance_usd * (risk_per_trade_pct / 100.0)
        stop_loss_decimal = stop_loss_pct / 100.0

        if stop_loss_decimal <= 0:
            raise ToolException(
                "Stop loss percentage must be greater than 0. "
                "Cannot calculate position size without a defined risk limit."
            )

        # Position size in USD (notional value, not margin)
        recommended_size_usd = max_loss_usd / stop_loss_decimal

        # Cap at maximum leverage allows
        max_notional = account_balance_usd * leverage
        if recommended_size_usd > max_notional:
            recommended_size_usd = max_notional
            # Recalculate actual max loss with capped size
            max_loss_usd = recommended_size_usd * stop_loss_decimal

        # Quantity in base asset
        recommended_quantity = recommended_size_usd / current_price if current_price > 0 else 0.0

        # Risk/reward context
        if auto_sl:
            note = (
                f"Stop loss auto-set to {stop_loss_pct:.2f}% based on 2x ATR. "
                "Adjust based on your trade thesis and support/resistance levels."
            )
        else:
            effective_risk_pct = (max_loss_usd / account_balance_usd) * 100
            note = (
                f"With {stop_loss_pct:.2f}% stop loss and {leverage}x leverage, "
                f"you risk {effective_risk_pct:.2f}% of your account "
                f"(${max_loss_usd:.2f}) on this trade."
            )

        return PositionSizeResult(
            symbol=symbol,
            current_price=round(current_price, 4),
            account_balance_usd=account_balance_usd,
            risk_per_trade_pct=risk_per_trade_pct,
            leverage=leverage,
            stop_loss_pct=round(stop_loss_pct, 2),
            recommended_size_usd=round(recommended_size_usd, 2),
            recommended_quantity=round(recommended_quantity, 6),
            max_loss_usd=round(max_loss_usd, 2),
            risk_reward_note=note,
        )
