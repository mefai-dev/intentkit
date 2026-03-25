"""Skill for assessing risk of a Binance Futures trading position."""

import logging
from typing import Any

from langchain_core.tools import ArgsSchema
from langchain_core.tools.base import ToolException
from pydantic import BaseModel, Field

from intentkit.skills.binance_risk.base import (
    BinanceRiskBaseTool,
    RiskAssessmentResult,
)

logger = logging.getLogger(__name__)


class AssessPositionRiskInput(BaseModel):
    """Input for AssessPositionRisk tool."""

    symbol: str = Field(
        ...,
        description="Binance Futures trading pair (e.g., BTCUSDT, ETHUSDT)",
    )
    position_size_usd: float = Field(
        ...,
        description="Position size in USD (e.g., 1000)",
        gt=0,
    )
    leverage: int = Field(
        default=1,
        description="Leverage multiplier (1-125, default 1)",
        ge=1,
        le=125,
    )
    side: str = Field(
        default="long",
        description="Position side: 'long' or 'short'",
    )


class AssessPositionRisk(BinanceRiskBaseTool):
    """Assess the risk of a Binance Futures position.

    Analyzes a trading pair and position to provide:
    - Volatility metrics (hourly, annualized, max drawdown, ATR)
    - Funding rate impact (current rate, annualized cost)
    - Liquidation distance and estimated liquidation price
    - Composite risk score (0-100) with component breakdown
    - Actionable risk management recommendations
    """

    name: str = "binance_risk_assess_position"
    description: str = (
        "Assess the risk of a Binance Futures trading position. "
        "Takes a symbol (e.g., BTCUSDT), position size in USD, leverage, "
        "and side (long/short). Returns volatility metrics, funding rate "
        "impact, liquidation distance, a risk score from 0-100, and "
        "risk management recommendations."
    )
    args_schema: ArgsSchema | None = AssessPositionRiskInput

    async def _arun(
        self,
        symbol: str,
        position_size_usd: float,
        leverage: int = 1,
        side: str = "long",
        **kwargs,
    ) -> RiskAssessmentResult:
        """Run the position risk assessment.

        Args:
            symbol: Binance Futures trading pair.
            position_size_usd: Position size in USD.
            leverage: Leverage multiplier.
            side: Position side ('long' or 'short').

        Returns:
            RiskAssessmentResult with full risk breakdown.
        """
        context = self.get_context()

        # Rate limit: 5 assessments per minute per user
        await self.user_rate_limit_by_skill(5, 60)

        symbol = symbol.upper()
        side = side.lower()
        if side not in ("long", "short"):
            raise ToolException("Side must be 'long' or 'short'")

        # Fetch market data in parallel-safe manner
        klines = await self.fetch_klines(symbol, interval="1h", limit=168)
        funding_data = await self.fetch_funding_rate(symbol)
        current_price = float(funding_data["markPrice"])
        current_funding = float(funding_data["lastFundingRate"])

        # Compute volatility
        volatility = self.compute_volatility(klines)

        # Compute liquidation distance
        liquidation = self.compute_liquidation_distance(
            entry_price=current_price,
            leverage=leverage,
            side=side,
        )

        # Funding rate analysis
        # Funding is paid every 8h, so 3x per day, 1095x per year
        annualized_funding_pct = current_funding * 100 * 1095
        # Cost for this position over 24h
        daily_funding_cost = abs(current_funding) * position_size_usd * 3
        funding_info: dict[str, Any] = {
            "current_rate": round(current_funding, 6),
            "current_rate_pct": round(current_funding * 100, 4),
            "annualized_pct": round(annualized_funding_pct, 2),
            "daily_cost_usd": round(daily_funding_cost, 2),
            "direction": "longs pay shorts" if current_funding > 0 else "shorts pay longs",
        }

        # Compute risk score
        risk_score = self.compute_risk_score(
            annualized_vol=volatility["annualized_volatility_pct"],
            funding_rate=current_funding,
            leverage=leverage,
            liq_distance_pct=liquidation["distance_pct"],
        )

        # Generate recommendations
        recommendations = self._generate_recommendations(
            risk_score=risk_score,
            volatility=volatility,
            leverage=leverage,
            liquidation=liquidation,
            funding_info=funding_info,
            side=side,
            position_size_usd=position_size_usd,
        )

        return RiskAssessmentResult(
            symbol=symbol,
            current_price=current_price,
            position_size_usd=position_size_usd,
            leverage=leverage,
            side=side,
            volatility=volatility,
            funding_rate=funding_info,
            liquidation=liquidation,
            risk_score=risk_score,
            recommendations=recommendations,
        )

    def _generate_recommendations(
        self,
        risk_score: dict[str, Any],
        volatility: dict[str, float],
        leverage: int,
        liquidation: dict[str, float],
        funding_info: dict[str, Any],
        side: str,
        position_size_usd: float,
    ) -> list[str]:
        """Generate actionable risk management recommendations.

        Args:
            risk_score: Computed risk score breakdown.
            volatility: Volatility metrics.
            leverage: Current leverage.
            liquidation: Liquidation distance info.
            funding_info: Funding rate info.
            side: Position side.
            position_size_usd: Position size in USD.

        Returns:
            List of recommendation strings.
        """
        recs = []
        level = risk_score["risk_level"]

        # Leverage warnings
        if leverage >= 50:
            recs.append(
                f"Leverage of {leverage}x is extremely high. "
                "Consider reducing to 10-20x to increase liquidation distance."
            )
        elif leverage >= 20:
            recs.append(
                f"Leverage of {leverage}x is aggressive. "
                "A 5% adverse move would result in significant losses."
            )

        # Liquidation proximity
        if liquidation["distance_pct"] < 2.0:
            recs.append(
                f"Liquidation is only {liquidation['distance_pct']}% away at "
                f"${liquidation['liquidation_price']:.2f}. "
                "Consider reducing position size or leverage immediately."
            )
        elif liquidation["distance_pct"] < 5.0:
            recs.append(
                f"Liquidation distance of {liquidation['distance_pct']}% is tight. "
                "Set a stop loss at least 1% above liquidation price."
            )

        # Volatility-based stop loss suggestion
        atr = volatility["atr_pct"]
        if atr > 0:
            suggested_sl = round(atr * 2, 2)
            recs.append(
                f"Based on current ATR ({atr:.2f}%), consider a stop loss "
                f"of at least {suggested_sl}% to avoid noise-triggered exits."
            )

        # Funding rate impact
        if abs(funding_info["current_rate_pct"]) > 0.03:
            direction = "long" if funding_info["current_rate"] > 0 else "short"
            if direction == side:
                recs.append(
                    f"Funding rate of {funding_info['current_rate_pct']:.4f}% "
                    f"works against your {side} position. "
                    f"Daily cost: ${funding_info['daily_cost_usd']:.2f}. "
                    "Consider timing entry near funding payment."
                )
            else:
                recs.append(
                    f"Funding rate of {funding_info['current_rate_pct']:.4f}% "
                    f"favors your {side} position. "
                    f"You earn approximately ${funding_info['daily_cost_usd']:.2f}/day."
                )

        # High volatility warning
        if volatility["annualized_volatility_pct"] > 100:
            recs.append(
                "Annualized volatility exceeds 100%. "
                "This pair is highly volatile; reduce position size accordingly."
            )

        # Max drawdown context
        if volatility["max_drawdown_pct"] > 10:
            recs.append(
                f"Recent max drawdown was {volatility['max_drawdown_pct']:.1f}% "
                f"(7-day window). At {leverage}x leverage, this would mean a "
                f"{volatility['max_drawdown_pct'] * leverage:.1f}% loss on margin."
            )

        # Overall risk level advice
        if level == "EXTREME":
            recs.append(
                "Overall risk is EXTREME. This setup has a high probability of "
                "liquidation. Strongly consider reducing leverage and size."
            )
        elif level == "HIGH":
            recs.append(
                "Overall risk is HIGH. Ensure you have a clear exit strategy "
                "and do not risk more than 1-2% of your account on this trade."
            )
        elif level == "MODERATE":
            recs.append(
                "Risk level is MODERATE. Standard risk management applies. "
                "Use stop losses and monitor funding rate changes."
            )

        return recs
