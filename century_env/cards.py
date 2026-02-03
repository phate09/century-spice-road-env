"""
Century: Spice Road - Card Data

Card encodings extracted from official card list spreadsheet.

Trader Card Format (10 dimensions):
    [card_type, num_upgrades, in_y, in_r, in_g, in_b, out_y, out_r, out_g, out_b]

    card_type: 0=Spice (gain), 1=Conversion (upgrade), 2=Exchange (trade)
    num_upgrades: For conversion cards, how many upgrades allowed
    in_*: Input spices required (for exchange cards)
    out_*: Output spices produced (for spice/exchange cards)

Scoring Card Format (5 dimensions):
    [points, req_y, req_r, req_g, req_b]

    points: Victory points awarded
    req_*: Required spices to claim

Spice Values: Y=Yellow(Turmeric)=1, R=Red(Saffron)=2, G=Green(Cardamom)=3, B=Brown(Cinnamon)=4
"""

import jax.numpy as jnp

# Import constants from centralized location
from century_env.constants import (
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
    SPICE_YELLOW,
    SPICE_RED,
    SPICE_GREEN,
    SPICE_BROWN,
    SPICE_NAMES,
    SPICE_VALUES,
)

# =============================================================================
# STARTING CARDS (each player gets one of each)
# =============================================================================

# Card indices 0 and 1 in ALL_TRADER_CARDS are the starting cards
STARTING_CARD_2_TURMERIC_IDX = 0
STARTING_CARD_CONVERSION_IDX = 1

# =============================================================================
# ALL TRADER CARDS - 45 total (including starting cards at indices 0-1)
# =============================================================================

# Format: [type, upgrades, in_y, in_r, in_g, in_b, out_y, out_r, out_g, out_b]
# Type: 0=Spice, 1=Conversion, 2=Exchange

_ALL_TRADER_CARD_DATA = [
    # === STARTING CARDS (indices 0-1) ===
    [0, 0, 0, 0, 0, 0, 2, 0, 0, 0],  # 0: Obtain YY (Starting: 2 Turmeric)
    [1, 2, 0, 0, 0, 0, 0, 0, 0, 0],  # 1: Upgrade 2x (Starting: Conversion)

    # === DECK CARDS (indices 2-44) - 43 cards shuffled for market ===
    [2, 0, 3, 0, 0, 0, 0, 0, 0, 1],  # 2: YYY → B
    [2, 0, 0, 1, 0, 0, 3, 0, 0, 0],  # 3: R → YYY
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 4: Obtain YR
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # 5: Obtain G
    [0, 0, 0, 0, 0, 0, 3, 0, 0, 0],  # 6: Obtain YYY
    [1, 3, 0, 0, 0, 0, 0, 0, 0, 0],  # 7: Upgrade 3x
    [2, 0, 0, 0, 2, 0, 2, 3, 0, 0],  # 8: GG → RRRYY
    [2, 0, 0, 0, 2, 0, 2, 1, 0, 1],  # 9: GG → BRYY
    [2, 0, 0, 0, 0, 1, 3, 0, 1, 0],  # 10: B → GYYY
    [2, 0, 0, 2, 0, 0, 3, 0, 1, 0],  # 11: RR → GYYY
    [2, 0, 0, 3, 0, 0, 2, 0, 2, 0],  # 12: RRR → GGYY
    [2, 0, 0, 0, 0, 1, 2, 2, 0, 0],  # 13: B → RRYY
    [2, 0, 4, 0, 0, 0, 0, 0, 2, 0],  # 14: YYYY → GG
    [0, 0, 0, 0, 0, 0, 2, 1, 0, 0],  # 15: Obtain RYY
    [0, 0, 0, 0, 0, 0, 4, 0, 0, 0],  # 16: Obtain YYYY
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 17: Obtain B
    [0, 0, 0, 0, 0, 0, 0, 2, 0, 0],  # 18: Obtain RR
    [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # 19: Obtain YG
    [2, 0, 2, 0, 0, 0, 0, 0, 1, 0],  # 20: YY → G
    [2, 0, 1, 1, 0, 0, 0, 0, 0, 1],  # 21: RY → B
    [2, 0, 0, 0, 1, 0, 0, 2, 0, 0],  # 22: G → RR
    [2, 0, 0, 2, 0, 0, 2, 0, 0, 1],  # 23: RR → BYY
    [2, 0, 3, 0, 0, 0, 0, 1, 1, 0],  # 24: YYY → RG
    [2, 0, 0, 0, 2, 0, 0, 2, 0, 1],  # 25: GG → BRR
    [2, 0, 0, 3, 0, 0, 1, 0, 1, 1],  # 26: RRR → BGY
    [2, 0, 0, 0, 0, 1, 0, 3, 0, 0],  # 27: B → RRR
    [2, 0, 0, 3, 0, 0, 0, 0, 0, 2],  # 28: RRR → BB
    [2, 0, 0, 0, 0, 1, 1, 1, 1, 0],  # 29: B → GRY
    [2, 0, 0, 0, 1, 0, 1, 2, 0, 0],  # 30: G → RRY
    [2, 0, 0, 0, 1, 0, 4, 1, 0, 0],  # 31: G → RYYYY
    [2, 0, 5, 0, 0, 0, 0, 0, 0, 2],  # 32: YYYYY → BB
    [2, 0, 4, 0, 0, 0, 0, 0, 1, 1],  # 33: YYYY → GB
    [2, 0, 0, 0, 0, 2, 0, 3, 2, 0],  # 34: BB → GGRRR
    [2, 0, 0, 0, 0, 2, 1, 1, 3, 0],  # 35: BB → GGGRY
    [2, 0, 5, 0, 0, 0, 0, 0, 3, 0],  # 36: YYYYY → GGG
    [2, 0, 2, 0, 1, 0, 0, 0, 0, 2],  # 37: GYY → BB
    [2, 0, 0, 0, 3, 0, 0, 0, 0, 3],  # 38: GGG → BBB
    [2, 0, 0, 3, 0, 0, 0, 0, 3, 0],  # 39: RRR → GGG
    [2, 0, 3, 0, 0, 0, 0, 3, 0, 0],  # 40: YYY → RRR
    [2, 0, 2, 0, 0, 0, 0, 2, 0, 0],  # 41: YY → RR
    [2, 0, 0, 0, 2, 0, 0, 0, 0, 2],  # 42: GG → BB
    [2, 0, 0, 2, 0, 0, 0, 0, 2, 0],  # 43: RR → GG
    [2, 0, 0, 0, 0, 1, 0, 0, 2, 0],  # 44: B → GG
]

ALL_TRADER_CARDS = jnp.array(_ALL_TRADER_CARD_DATA, dtype=jnp.int32)
NUM_ALL_TRADER_CARDS = len(_ALL_TRADER_CARD_DATA)  # 45

# Starting cards (given to each player at setup)
STARTING_CARDS = ALL_TRADER_CARDS[:2]  # Indices 0-1
NUM_STARTING_CARDS = 2

# Deck cards (shuffled for market)
DECK_TRADER_CARDS = ALL_TRADER_CARDS[2:]  # Indices 2-44
NUM_DECK_TRADER_CARDS = NUM_ALL_TRADER_CARDS - NUM_STARTING_CARDS  # 43


# =============================================================================
# SCORING CARDS (Point Cards) - 36 cards
# =============================================================================

# Format: [points, req_y, req_r, req_g, req_b]

_SCORING_CARD_DATA = [
    # Basic combinations (2 spice types)
    [6, 2, 2, 0, 0],    # 0: YYRR = 6pts
    [7, 3, 2, 0, 0],    # 1: YYYRR = 7pts
    [8, 0, 4, 0, 0],    # 2: RRRR = 8pts
    [8, 2, 0, 2, 0],    # 3: YYGG = 8pts
    [8, 2, 3, 0, 0],    # 4: YYRRR = 8pts
    [9, 3, 0, 2, 0],    # 5: YYYGG = 9pts
    [10, 0, 2, 2, 0],   # 6: RRGG = 10pts
    [10, 0, 5, 0, 0],   # 7: RRRRR = 10pts
    [10, 2, 0, 0, 2],   # 8: YYBB = 10pts
    [11, 2, 0, 3, 0],   # 9: YYGGG = 11pts
    [11, 3, 0, 0, 2],   # 10: YYYBB = 11pts
    [12, 0, 0, 4, 0],   # 11: GGGG = 12pts
    [12, 0, 2, 0, 2],   # 12: RRBB = 12pts
    [12, 0, 3, 2, 0],   # 13: RRRGG = 12pts
    [13, 0, 2, 3, 0],   # 14: RRGGG = 13pts
    [14, 0, 0, 2, 2],   # 15: GGBB = 14pts
    [14, 0, 3, 0, 2],   # 16: RRRBB = 14pts
    [14, 2, 0, 0, 3],   # 17: YYBBB = 14pts
    [15, 0, 0, 5, 0],   # 18: GGGGG = 15pts
    [16, 0, 0, 0, 4],   # 19: BBBB = 16pts
    [16, 0, 2, 0, 3],   # 20: RRBBB = 16pts
    [17, 0, 0, 3, 2],   # 21: GGGBB = 17pts
    [18, 0, 0, 2, 3],   # 22: GGBBB = 18pts
    [20, 0, 0, 0, 5],   # 23: BBBBB = 20pts

    # Mixed combinations (3 spice types)
    [9, 2, 1, 0, 1],    # 24: YYRB = 9pts
    [12, 0, 2, 1, 1],   # 25: RRGB = 12pts
    [12, 1, 0, 2, 1],   # 26: YGGB = 12pts
    [13, 2, 2, 2, 0],   # 27: YYRRGG = 13pts
    [15, 2, 2, 0, 2],   # 28: YYRRBB = 15pts
    [17, 2, 0, 2, 2],   # 29: YYGGBB = 17pts
    [19, 0, 2, 2, 2],   # 30: RRGGBB = 19pts

    # Rainbow combinations (4 spice types)
    [12, 1, 1, 1, 1],   # 31: YRGB = 12pts
    [14, 3, 1, 1, 1],   # 32: YYYRGB = 14pts
    [16, 1, 3, 1, 1],   # 33: YRRRGB = 16pts
    [18, 1, 1, 3, 1],   # 34: YRGGGB = 18pts
    [20, 1, 1, 1, 3],   # 35: YRGBBB = 20pts
]

SCORING_CARDS = jnp.array(_SCORING_CARD_DATA, dtype=jnp.int32)
NUM_SCORING_CARDS = len(_SCORING_CARD_DATA)  # 36


# =============================================================================
# CARD DESCRIPTION STRINGS (for rendering/debugging)
# =============================================================================

_TRADER_CARD_DESCRIPTIONS = [
    "Obtain YY (Starting: 2 Turmeric)",
    "Upgrade 2x (Starting: Conversion)",
    "YYY → B",
    "R → YYY",
    "Obtain YR",
    "Obtain G",
    "Obtain YYY",
    "Upgrade 3x",
    "GG → RRRYY",
    "GG → BRYY",
    "B → GYYY",
    "RR → GYYY",
    "RRR → GGYY",
    "B → RRYY",
    "YYYY → GG",
    "Obtain RYY",
    "Obtain YYYY",
    "Obtain B",
    "Obtain RR",
    "Obtain YG",
    "YY → G",
    "RY → B",
    "G → RR",
    "RR → BYY",
    "YYY → RG",
    "GG → BRR",
    "RRR → BGY",
    "B → RRR",
    "RRR → BB",
    "B → GRY",
    "G → RRY",
    "G → RYYYY",
    "YYYYY → BB",
    "YYYY → GB",
    "BB → GGRRR",
    "BB → GGGRY",
    "YYYYY → GGG",
    "GYY → BB",
    "GGG → BBB",
    "RRR → GGG",
    "YYY → RRR",
    "YY → RR",
    "GG → BB",
    "RR → GG",
    "B → GG",
]

_SCORING_CARD_DESCRIPTIONS = [
    "YYRR = 6pts",
    "YYYRR = 7pts",
    "RRRR = 8pts",
    "YYGG = 8pts",
    "YYRRR = 8pts",
    "YYYGG = 9pts",
    "RRGG = 10pts",
    "RRRRR = 10pts",
    "YYBB = 10pts",
    "YYGGG = 11pts",
    "YYYBB = 11pts",
    "GGGG = 12pts",
    "RRBB = 12pts",
    "RRRGG = 12pts",
    "RRGGG = 13pts",
    "GGBB = 14pts",
    "RRRBB = 14pts",
    "YYBBB = 14pts",
    "GGGGG = 15pts",
    "BBBB = 16pts",
    "RRBBB = 16pts",
    "GGGBB = 17pts",
    "GGBBB = 18pts",
    "BBBBB = 20pts",
    "YYRB = 9pts",
    "RRGB = 12pts",
    "YGGB = 12pts",
    "YYRRGG = 13pts",
    "YYRRBB = 15pts",
    "YYGGBB = 17pts",
    "RRGGBB = 19pts",
    "YRGB = 12pts",
    "YYYRGB = 14pts",
    "YRRRGB = 16pts",
    "YRGGGB = 18pts",
    "YRGBBB = 20pts",
]


def get_trader_card_description(card_idx: int) -> str:
    """Get human-readable description of a trader card."""
    if 0 <= card_idx < len(_TRADER_CARD_DESCRIPTIONS):
        return _TRADER_CARD_DESCRIPTIONS[card_idx]
    return f"Unknown card {card_idx}"


def get_scoring_card_description(card_idx: int) -> str:
    """Get human-readable description of a scoring card."""
    if 0 <= card_idx < len(_SCORING_CARD_DESCRIPTIONS):
        return _SCORING_CARD_DESCRIPTIONS[card_idx]
    return f"Unknown card {card_idx}"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_card_type(card: jnp.ndarray) -> jnp.ndarray:
    """Get the type of a trader card (0=Spice, 1=Conversion, 2=Exchange)."""
    return card[0]


def get_card_upgrades(card: jnp.ndarray) -> jnp.ndarray:
    """Get number of upgrades for a conversion card."""
    return card[1]


def get_card_input(card: jnp.ndarray) -> jnp.ndarray:
    """Get input spices for an exchange card [y, r, g, b]."""
    return card[2:6]


def get_card_output(card: jnp.ndarray) -> jnp.ndarray:
    """Get output spices for a spice/exchange card [y, r, g, b]."""
    return card[6:10]


def get_scoring_points(card: jnp.ndarray) -> jnp.ndarray:
    """Get victory points for a scoring card."""
    return card[0]


def get_scoring_requirements(card: jnp.ndarray) -> jnp.ndarray:
    """Get required spices for a scoring card [y, r, g, b]."""
    return card[1:5]


def can_afford_scoring_card(caravan: jnp.ndarray, card: jnp.ndarray) -> jnp.ndarray:
    """Check if caravan can afford the requirements of a scoring card."""
    requirements = get_scoring_requirements(card)
    return jnp.all(caravan >= requirements)


def can_afford_exchange(caravan: jnp.ndarray, card: jnp.ndarray) -> jnp.ndarray:
    """Check if caravan can afford one execution of an exchange card."""
    input_spices = get_card_input(card)
    return jnp.all(caravan >= input_spices)


# =============================================================================
# VALIDATION
# =============================================================================

def _validate_cards():
    """Validate card data integrity."""
    # Check all trader cards
    assert ALL_TRADER_CARDS.shape == (NUM_ALL_TRADER_CARDS, 10), \
        f"Trader cards shape mismatch: {ALL_TRADER_CARDS.shape}"

    for i, card in enumerate(ALL_TRADER_CARDS):
        card_type = int(card[0])
        assert card_type in [0, 1, 2], f"Invalid card type {card_type} for trader card {i}"

        if card_type == CARD_TYPE_SPICE:
            # Spice cards should have output but no input
            assert jnp.sum(card[2:6]) == 0, f"Spice card {i} has input spices"
            assert jnp.sum(card[6:10]) > 0, f"Spice card {i} has no output spices"

        elif card_type == CARD_TYPE_CONVERSION:
            # Conversion cards should have upgrades, no input/output
            assert card[1] > 0, f"Conversion card {i} has no upgrades"
            assert jnp.sum(card[2:10]) == 0, f"Conversion card {i} has input/output spices"

        elif card_type == CARD_TYPE_EXCHANGE:
            # Exchange cards should have both input and output
            assert jnp.sum(card[2:6]) > 0, f"Exchange card {i} has no input spices"
            assert jnp.sum(card[6:10]) > 0, f"Exchange card {i} has no output spices"

    # Check scoring cards
    assert SCORING_CARDS.shape == (NUM_SCORING_CARDS, 5), \
        f"Scoring cards shape mismatch: {SCORING_CARDS.shape}"

    for i, card in enumerate(SCORING_CARDS):
        assert card[0] > 0, f"Scoring card {i} has no points"
        assert jnp.sum(card[1:5]) > 0, f"Scoring card {i} has no requirements"

    # Check deck cards count
    assert DECK_TRADER_CARDS.shape[0] == NUM_DECK_TRADER_CARDS == 43, \
        f"Deck trader cards count mismatch: {DECK_TRADER_CARDS.shape[0]}"

    print(f"Validated {NUM_ALL_TRADER_CARDS} total trader cards:")
    print(f"  - {NUM_STARTING_CARDS} starting cards (indices 0-1)")
    print(f"  - {NUM_DECK_TRADER_CARDS} deck cards (indices 2-44)")
    print(f"Validated {NUM_SCORING_CARDS} scoring cards")


if __name__ == "__main__":
    _validate_cards()

    spice_count = int(jnp.sum(ALL_TRADER_CARDS[:, 0] == 0))
    conversion_count = int(jnp.sum(ALL_TRADER_CARDS[:, 0] == 1))
    exchange_count = int(jnp.sum(ALL_TRADER_CARDS[:, 0] == 2))

    print(f"\nTrader Card Breakdown:")
    print(f"  - Spice cards: {spice_count}")
    print(f"  - Conversion cards: {conversion_count}")
    print(f"  - Exchange cards: {exchange_count}")

    print(f"\nScoring Card Stats:")
    print(f"  - Point range: {int(SCORING_CARDS[:, 0].min())} - {int(SCORING_CARDS[:, 0].max())}")

    print(f"\nStarting Cards:")
    for i in range(NUM_STARTING_CARDS):
        print(f"  - {get_trader_card_description(i)}: {ALL_TRADER_CARDS[i].tolist()}")
