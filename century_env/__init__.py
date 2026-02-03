"""
Century: Spice Road - Jumanji RL Environment

A JAX-based reinforcement learning environment for the Century: Spice Road board game,
compatible with the Jumanji framework.
"""

# Constants
from century_env.constants import (
    # Player
    MAX_PLAYERS,
    MIN_PLAYERS,
    # Cards
    MAX_PLAYER_CARDS,
    NUM_STARTING_CARDS,
    NUM_DECK_TRADER_CARDS,
    NUM_ALL_TRADER_CARDS,
    NUM_SCORING_CARDS,
    MAX_SCORED_CARDS,
    # Spices
    NUM_SPICE_TYPES,
    CARAVAN_LIMIT,
    SPICE_YELLOW,
    SPICE_RED,
    SPICE_GREEN,
    SPICE_BROWN,
    SPICE_NAMES,
    SPICE_VALUES,
    # Card types
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
    # Market
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    # Action space
    NUM_ACTION_TYPES,
    NUM_CONTINUE_FLAGS,
    # Game setup
    STARTING_SPICES,
    SCORING_CARDS_TO_WIN,
)

# Types
from century_env.types import (
    State,
    Observation,
    Phase,
    ActionType,
    Action,
    ActionMask,
)

# Card data
from century_env.cards import (
    ALL_TRADER_CARDS,
    DECK_TRADER_CARDS,
    STARTING_CARDS,
    SCORING_CARDS,
    get_card_type,
    get_card_upgrades,
    get_card_input,
    get_card_output,
    get_scoring_points,
    get_scoring_requirements,
    can_afford_scoring_card,
    can_afford_exchange,
)

# Environment
from century_env.env import CenturySpiceRoad

# Rendering (for debugging - not JIT-compatible)
from century_env.render import (
    render_state,
    render_observation,
    render_action,
    render_caravan,
)

__version__ = "0.1.0"
