"""
Century: Spice Road - Game Constants

All game constants in one place for clean imports and single source of truth.
"""

import jax.numpy as jnp

# =============================================================================
# PLAYER CONSTANTS
# =============================================================================

MAX_PLAYERS = 5
MIN_PLAYERS = 2

# =============================================================================
# CARD CONSTANTS
# =============================================================================

MAX_PLAYER_CARDS = 25  # Maximum cards a player can hold in hand
NUM_STARTING_CARDS = 2  # Cards dealt to each player at game start
NUM_DECK_TRADER_CARDS = 43  # Shuffled trader cards (excluding starting cards)
NUM_ALL_TRADER_CARDS = 45  # Total trader cards (including starting cards)
NUM_SCORING_CARDS = 36  # Total scoring cards
MAX_SCORED_CARDS = 10  # Maximum scoring cards per player

# =============================================================================
# SPICE CONSTANTS
# =============================================================================

NUM_SPICE_TYPES = 4
CARAVAN_LIMIT = 10  # Maximum total spices in caravan

# Spice indices
SPICE_YELLOW = 0  # Turmeric (lowest value)
SPICE_RED = 1     # Saffron
SPICE_GREEN = 2   # Cardamom
SPICE_BROWN = 3   # Cinnamon (highest value)

SPICE_NAMES = ['Yellow (Turmeric)', 'Red (Saffron)', 'Green (Cardamom)', 'Brown (Cinnamon)']
SPICE_VALUES = jnp.array([1, 2, 3, 4], dtype=jnp.int32)

# =============================================================================
# CARD TYPE CONSTANTS
# =============================================================================

CARD_TYPE_SPICE = 0      # Gain spices
CARD_TYPE_CONVERSION = 1  # Upgrade spices
CARD_TYPE_EXCHANGE = 2    # Trade spices

# =============================================================================
# MARKET CONSTANTS
# =============================================================================

NUM_MARKET_SLOTS = 6      # Trader cards in market row
NUM_SCORING_SLOTS = 5     # Scoring cards in scoring row
MAX_SPICES_PER_MARKET = 20  # Maximum spices that can accumulate on a market card

# =============================================================================
# ACTION SPACE CONSTANTS
# =============================================================================

NUM_ACTION_TYPES = 4      # Play, Acquire, Rest, Score
NUM_CONTINUE_FLAGS = 2    # AGAIN=0, DONE=1

# Action head sizes for multi-discrete action space
ACTION_HEAD_ACTION_TYPE = 4   # Play/Acquire/Rest/Score
ACTION_HEAD_CARD_IDX = 25     # Hand card selection
ACTION_HEAD_MARKET_POS = 6    # Market slot selection
ACTION_HEAD_SCORING_IDX = 5   # Scoring card selection
ACTION_HEAD_SPICE_TYPE = 4    # Spice selection
ACTION_HEAD_CONTINUE = 2      # AGAIN/DONE

# =============================================================================
# STARTING SPICES BY PLAYER POSITION
# =============================================================================

STARTING_SPICES = jnp.array([
    [3, 0, 0, 0],  # Player 0 (start): 3 yellow
    [4, 0, 0, 0],  # Player 1: 4 yellow
    [4, 0, 0, 0],  # Player 2: 4 yellow
    [3, 1, 0, 0],  # Player 3: 3 yellow + 1 red
    [3, 1, 0, 0],  # Player 4: 3 yellow + 1 red
], dtype=jnp.int32)

# =============================================================================
# GAME END CONSTANTS
# =============================================================================

# Index by num_players: [0, 1, 2, 3, 4, 5]
# 2-3 players: 6th scoring card triggers end
# 4-5 players: 5th scoring card triggers end
SCORING_CARDS_TO_WIN = jnp.array([0, 0, 6, 6, 5, 5], dtype=jnp.int32)

# =============================================================================
# COIN CONSTANTS
# =============================================================================

# Coins per player count: gold and silver both get 2 * num_players
def get_initial_coins(num_players: int) -> int:
    """Return initial gold/silver coin count based on player count."""
    return 2 * num_players
