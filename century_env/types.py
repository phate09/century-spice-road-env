"""
Century: Spice Road - Type Definitions

State, Observation, and Phase dataclasses using chex for JAX compatibility.
"""

import enum
from typing import Tuple

import chex
import jax.numpy as jnp

from century_env.constants import (
    MAX_PLAYERS,
    MAX_PLAYER_CARDS,
    MAX_SCORED_CARDS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    NUM_SPICE_TYPES,
    NUM_DECK_TRADER_CARDS,
    NUM_SCORING_CARDS,
)


class Phase(enum.IntEnum):
    """Game phase state machine.

    Turn flow:
    CHOOSE_ACTION -> [Play|Acquire|Rest|Score]
        Play -> SELECT_CARD -> EXECUTE_CARD (loop) -> [DISCARD_OVERFLOW?] -> done
        Acquire -> SELECT_MARKET_POS -> PLACE_SPICE (loop) -> [DISCARD_OVERFLOW?] -> done
        Rest -> done (no overflow possible)
        Score -> SELECT_SCORING_CARD -> done (removes spices, no overflow)
    """
    CHOOSE_ACTION = 0
    SELECT_CARD = 1
    EXECUTE_CARD = 2
    SELECT_MARKET_POS = 3
    PLACE_SPICE = 4
    SELECT_SCORING_CARD = 5
    DISCARD_OVERFLOW = 6


class ActionType(enum.IntEnum):
    """Top-level action types available during CHOOSE_ACTION phase."""
    PLAY = 0      # Play a card from hand
    ACQUIRE = 1   # Acquire a card from market
    REST = 2      # Return played cards to hand
    SCORE = 3     # Claim a scoring card


@chex.dataclass(frozen=True)
class State:
    """Complete game state.

    All arrays are padded to maximum sizes for JAX compatibility.
    Grouped by domain for clarity.
    """
    # === Game Flow ===
    current_player: chex.Array  # int32 scalar - whose turn
    phase: chex.Array  # int32 scalar - current phase
    num_players: chex.Array  # int32 scalar - active players (2-5)
    game_triggered: chex.Array  # bool scalar - end game triggered
    trigger_player: chex.Array  # int32 scalar - who triggered end (-1 if not triggered)

    # === Per-Player State (all shape (MAX_PLAYERS, ...)) ===
    hands: chex.Array  # (5, 25, 10) - trader cards in each player's hand
    hand_sizes: chex.Array  # (5,) - number of cards in each hand
    played_piles: chex.Array  # (5, 25, 10) - played cards per player
    played_sizes: chex.Array  # (5,) - number of played cards per player
    caravans: chex.Array  # (5, 4) - spices per player [y, r, g, b]
    scored_cards: chex.Array  # (5, 10, 5) - scoring cards claimed per player
    scored_counts: chex.Array  # (5,) - number of scoring cards per player
    gold_coins: chex.Array  # (5,) - gold coins per player
    silver_coins: chex.Array  # (5,) - silver coins per player

    # === Market State ===
    market_cards: chex.Array  # (6, 10) - trader cards in market row
    market_spices: chex.Array  # (6, 4) - spices placed on each market card
    market_size: chex.Array  # int32 scalar - valid cards in market (may be < 6 when deck empty)
    scoring_row: chex.Array  # (5, 5) - scoring cards available
    scoring_row_size: chex.Array  # int32 scalar - valid scoring cards (may be < 5)
    trader_deck: chex.Array  # (43, 10) - remaining trader deck
    trader_deck_size: chex.Array  # int32 scalar - cards remaining in trader deck
    scoring_deck: chex.Array  # (36, 5) - remaining scoring deck
    scoring_deck_size: chex.Array  # int32 scalar - cards remaining in scoring deck
    gold_remaining: chex.Array  # int32 scalar - gold coins in supply
    silver_remaining: chex.Array  # int32 scalar - silver coins in supply

    # === Action State (for multi-step actions) ===
    selected_card_idx: chex.Array  # int32 scalar - selected hand card index
    selected_card: chex.Array  # (10,) - the selected card data
    remaining_upgrades: chex.Array  # int32 scalar - upgrades left for conversion
    acquire_target_position: chex.Array  # int32 scalar - market position being acquired
    spices_placed_count: chex.Array  # int32 scalar - spices placed so far during acquire

    # === PRNG ===
    key: chex.PRNGKey  # Must be split before any stochastic operation


@chex.dataclass(frozen=True)
class Observation:
    """Ego-centric observation for a player.

    Current player always sees themselves as "player 0".
    Opponents are rotated so opp[0] is the next player clockwise.
    """
    # === My State ===
    my_hand: chex.Array  # (25, 10) - my trader cards
    my_hand_size: chex.Array  # int32 scalar - cards in my hand
    my_played: chex.Array  # (25, 10) - my played cards
    my_played_size: chex.Array  # int32 scalar - my played card count
    my_caravan: chex.Array  # (4,) - my spices [y, r, g, b]
    my_scored: chex.Array  # (10, 5) - my scoring cards
    my_scored_count: chex.Array  # int32 scalar - my scoring card count
    my_gold: chex.Array  # int32 scalar - my gold coins
    my_silver: chex.Array  # int32 scalar - my silver coins

    # === Opponent State (rotated, up to 4 opponents) ===
    opp_active: chex.Array  # (4,) - which opponents are active players
    opp_hand_sizes: chex.Array  # (4,) - cards in each opponent's hand
    opp_played: chex.Array  # (4, 25, 10) - opponents' played cards (public)
    opp_played_sizes: chex.Array  # (4,) - opponents' played card counts
    opp_caravans: chex.Array  # (4, 4) - opponents' spices (public)
    opp_scored: chex.Array  # (4, 10, 5) - opponents' scoring cards (public)
    opp_scored_counts: chex.Array  # (4,) - opponents' scoring card counts
    opp_gold: chex.Array  # (4,) - opponents' gold coins
    opp_silver: chex.Array  # (4,) - opponents' silver coins

    # === Market State (public) ===
    market_cards: chex.Array  # (6, 10) - market trader cards
    market_spices: chex.Array  # (6, 4) - spices on market cards
    market_size: chex.Array  # int32 scalar - valid market cards
    scoring_row: chex.Array  # (5, 5) - scoring cards available
    scoring_row_size: chex.Array  # int32 scalar - valid scoring cards

    # === Supply Info (public) ===
    gold_remaining: chex.Array  # int32 scalar - gold in supply
    silver_remaining: chex.Array  # int32 scalar - silver in supply
    trader_deck_size: chex.Array  # int32 scalar - trader deck remaining
    scoring_deck_size: chex.Array  # int32 scalar - scoring deck remaining

    # === Game Info ===
    num_players: chex.Array  # int32 scalar - total players
    current_phase: chex.Array  # int32 scalar - current game phase

    # === Action State (for multi-step actions) ===
    selected_card: chex.Array  # (10,) - currently selected card (for execute phase)
    remaining_upgrades: chex.Array  # int32 scalar - upgrades left
    acquire_target_position: chex.Array  # int32 scalar - market position being acquired
    spices_placed_count: chex.Array  # int32 scalar - spices placed during acquire

    # === Action Mask ===
    action_mask: Tuple[chex.Array, ...]  # 6-tuple of boolean masks for each action head


# Type aliases for clarity
Action = chex.Array  # Shape (6,) multi-discrete action
ActionMask = Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]
