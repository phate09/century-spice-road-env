"""
Century: Spice Road - Jumanji Environment Registration

Register the environment with Jumanji for easy instantiation.
"""

from typing import Dict, Any, Optional

from century_env.env import CenturySpiceRoad
from century_env.constants import MIN_PLAYERS, MAX_PLAYERS


# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "num_players": 4,
}


def make_century_spice_road(
    num_players: int = 4,
    **kwargs
) -> CenturySpiceRoad:
    """Create a Century: Spice Road environment.

    Args:
        num_players: Number of players (2-5)
        **kwargs: Additional configuration (currently unused)

    Returns:
        CenturySpiceRoad environment instance
    """
    return CenturySpiceRoad(num_players=num_players)


def register_century_spice_road():
    """Register Century: Spice Road with Jumanji registry.

    After calling this function, you can create the environment via:
        import jumanji
        env = jumanji.make("CenturySpiceRoad-v0")

    Note: Jumanji registration requires the environment to be importable
    from the jumanji.environments module. For standalone use, just import
    CenturySpiceRoad directly from century_env.
    """
    try:
        from jumanji.registration import register

        register(
            id="CenturySpiceRoad-v0",
            entry_point="century_env.env:CenturySpiceRoad",
            kwargs=DEFAULT_CONFIG,
        )
    except ImportError:
        # Jumanji registration not available (older version or missing)
        pass


# Convenience aliases for different player counts
def make_2p() -> CenturySpiceRoad:
    """Create a 2-player environment."""
    return make_century_spice_road(num_players=2)


def make_3p() -> CenturySpiceRoad:
    """Create a 3-player environment."""
    return make_century_spice_road(num_players=3)


def make_4p() -> CenturySpiceRoad:
    """Create a 4-player environment."""
    return make_century_spice_road(num_players=4)


def make_5p() -> CenturySpiceRoad:
    """Create a 5-player environment."""
    return make_century_spice_road(num_players=5)
