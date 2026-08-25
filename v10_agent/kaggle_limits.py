"""
Kaggle Competition Limits Module for V10 Agent.
Reads and validates environment variables for competition constraints.
Ref: Spec 8.4 (Competition Limits)
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompetitionLimits:
    """
    Immutable container for Kaggle competition limits.
    
    Attributes are read from environment variables with safe defaults
    from V10.0 specification.
    """
    max_actions_per_game: int
    max_level_attempts: int
    game_wall_clock_limit_seconds: int
    competition_wall_clock_limit_seconds: int
    
    def __post_init__(self):
        # Validate positive values
        if self.max_actions_per_game <= 0:
            raise ValueError("max_actions_per_game must be positive")
        if self.max_level_attempts <= 0:
            raise ValueError("max_level_attempts must be positive")
        if self.game_wall_clock_limit_seconds <= 0:
            raise ValueError("game_wall_clock_limit_seconds must be positive")
        if self.competition_wall_clock_limit_seconds <= 0:
            raise ValueError("competition_wall_clock_limit_seconds must be positive")


# V10.0 Specification Defaults
DEFAULT_MAX_ACTIONS_PER_GAME = 50
DEFAULT_MAX_LEVEL_ATTEMPTS = 3
DEFAULT_GAME_WALL_CLOCK_LIMIT_SECONDS = 300  # 5 minutes per game
DEFAULT_COMPETITION_WALL_CLOCK_LIMIT_SECONDS = 7200  # 2 hours total


def _safe_int_env(var_name: str, default: int) -> int:
    """
    Safely reads an integer from environment variable.
    Returns default if not set or invalid.
    """
    value = os.environ.get(var_name)
    if value is None:
        return default
    
    try:
        return int(value)
    except (ValueError, TypeError):
        # Log warning in real implementation
        return default


def get_competition_limits() -> CompetitionLimits:
    """
    Reads competition limits from Kaggle environment variables.
    
    Environment Variables:
    - LCLD_MAX_ACTIONS_PER_GAME
    - LCLD_MAX_LEVEL_ATTEMPTS
    - LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS
    - LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS
    
    Returns:
        CompetitionLimits with validated values.
        
    Note:
        If environment variables are missing or invalid, safe defaults
        from V10.0 specification are used.
    """
    max_actions = _safe_int_env(
        "LCLD_MAX_ACTIONS_PER_GAME",
        DEFAULT_MAX_ACTIONS_PER_GAME
    )
    
    max_attempts = _safe_int_env(
        "LCLD_MAX_LEVEL_ATTEMPTS",
        DEFAULT_MAX_LEVEL_ATTEMPTS
    )
    
    game_limit = _safe_int_env(
        "LCLD_GAME_WALL_CLOCK_LIMIT_SECONDS",
        DEFAULT_GAME_WALL_CLOCK_LIMIT_SECONDS
    )
    
    comp_limit = _safe_int_env(
        "LCLD_COMPETITION_WALL_CLOCK_LIMIT_SECONDS",
        DEFAULT_COMPETITION_WALL_CLOCK_LIMIT_SECONDS
    )
    
    return CompetitionLimits(
        max_actions_per_game=max_actions,
        max_level_attempts=max_attempts,
        game_wall_clock_limit_seconds=game_limit,
        competition_wall_clock_limit_seconds=comp_limit
    )


def validate_limits(limits: CompetitionLimits) -> tuple[bool, list[str]]:
    """
    Validates that limits are within acceptable ranges.
    
    Args:
        limits: CompetitionLimits to validate
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    # Check for unusually low values
    if limits.max_actions_per_game < 10:
        warnings.append(f"max_actions_per_game ({limits.max_actions_per_game}) is very low")
    
    if limits.max_level_attempts < 1:
        warnings.append(f"max_level_attempts ({limits.max_level_attempts}) is too low")
    
    if limits.game_wall_clock_limit_seconds < 60:
        warnings.append(f"game_wall_clock_limit_seconds ({limits.game_wall_clock_limit_seconds}) is very short")
    
    if limits.competition_wall_clock_limit_seconds < 300:
        warnings.append(f"competition_wall_clock_limit_seconds ({limits.competition_wall_clock_limit_seconds}) is very short")
    
    # Check for unusually high values (might indicate misconfiguration)
    if limits.max_actions_per_game > 1000:
        warnings.append(f"max_actions_per_game ({limits.max_actions_per_game}) is very high")
    
    if limits.competition_wall_clock_limit_seconds > 86400:
        warnings.append(f"competition_wall_clock_limit_seconds ({limits.competition_wall_clock_limit_seconds}) exceeds 24h")
    
    return len(warnings) == 0, warnings


if __name__ == "__main__":
    # Demo usage
    limits = get_competition_limits()
    print(f"Competition Limits:")
    print(f"  Max Actions/Game: {limits.max_actions_per_game}")
    print(f"  Max Level Attempts: {limits.max_level_attempts}")
    print(f"  Game Time Limit: {limits.game_wall_clock_limit_seconds}s")
    print(f"  Total Time Limit: {limits.competition_wall_clock_limit_seconds}s")
    
    is_valid, warnings = validate_limits(limits)
    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  - {w}")
    else:
        print("\nAll limits validated successfully.")
