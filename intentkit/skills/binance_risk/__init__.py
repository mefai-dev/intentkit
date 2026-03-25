"""Binance smart risk assessment skills for position sizing and risk management."""

import logging
from typing import TypedDict

from intentkit.skills.base import SkillConfig, SkillState
from intentkit.skills.binance_risk.base import BinanceRiskBaseTool
from intentkit.skills.binance_risk.assess_position_risk import AssessPositionRisk
from intentkit.skills.binance_risk.calculate_position_size import CalculatePositionSize

# Cache skills at the system level, because they are stateless
_cache: dict[str, BinanceRiskBaseTool] = {}

logger = logging.getLogger(__name__)


class SkillStates(TypedDict):
    binance_risk_assess_position: SkillState
    binance_risk_calculate_size: SkillState


_SKILL_NAME_TO_CLASS_MAP: dict[str, type[BinanceRiskBaseTool]] = {
    "binance_risk_assess_position": AssessPositionRisk,
    "binance_risk_calculate_size": CalculatePositionSize,
}


class Config(SkillConfig):
    """Configuration for Binance Risk Assessment skills."""

    enabled: bool
    states: SkillStates


async def get_skills(
    config: "Config",
    is_private: bool,
    **_,
) -> list[BinanceRiskBaseTool]:
    """Get all Binance Risk Assessment skills.

    Args:
        config: The configuration for Binance Risk skills.
        is_private: Whether to include private skills.

    Returns:
        A list of Binance Risk Assessment skills.
    """

    available_skills = []

    # Include skills based on their state
    for skill_name, state in config["states"].items():
        if state == "disabled":
            continue
        elif state == "public" or (state == "private" and is_private):
            available_skills.append(skill_name)

    # Get each skill using the cached getter
    result = []
    for name in available_skills:
        skill = get_binance_risk_skill(name)
        if skill:
            result.append(skill)
    return result


def get_binance_risk_skill(
    name: str,
) -> BinanceRiskBaseTool | None:
    """Get a Binance Risk skill by name.

    Args:
        name: The name of the skill to get

    Returns:
        The requested Binance Risk skill
    """

    # Return from cache immediately if already exists
    if name in _cache:
        return _cache[name]

    skill_class = _SKILL_NAME_TO_CLASS_MAP.get(name)
    if not skill_class:
        logger.warning("Unknown Binance Risk skill: %s", name)
        return None

    _cache[name] = skill_class()
    return _cache[name]


def available() -> bool:
    """Check if this skill category is available based on system config.

    This skill uses public Binance API endpoints (no API key required
    for market data), so it is always available.
    """
    return True
