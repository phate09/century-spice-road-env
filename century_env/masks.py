"""
Century: Spice Road - Legal Action Masking

Compute which actions are legal in each game phase.
All functions must be jittable.
"""

from typing import Tuple

import jax.numpy as jnp
from jax import lax

from century_env.constants import (
    NUM_ACTION_TYPES,
    MAX_PLAYER_CARDS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    NUM_SPICE_TYPES,
    NUM_CONTINUE_FLAGS,
    CARAVAN_LIMIT,
    SPICE_BROWN,
)
from century_env.cards import (
    get_card_type,
    get_scoring_requirements,
    can_afford_scoring_card,
    can_afford_exchange,
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)
from century_env.types import State, Phase, ActionType
from century_env.mechanics import can_apply_conversion, can_apply_exchange, caravan_total


def get_action_mask(state: State) -> Tuple[jnp.ndarray, ...]:
    """Get legal action masks for all 6 action heads.

    Returns tuple of 6 boolean arrays:
        - action_type_mask: shape (4,) - Play/Acquire/Rest/Score
        - card_idx_mask: shape (25,) - hand card selection
        - market_pos_mask: shape (6,) - market position selection
        - scoring_idx_mask: shape (5,) - scoring card selection
        - spice_type_mask: shape (4,) - spice selection
        - continue_mask: shape (2,) - AGAIN(0)/DONE(1)
    """
    phase = state.phase

    # Dispatch based on phase
    masks = lax.switch(
        phase,
        [
            _mask_choose_action,     # Phase 0: CHOOSE_ACTION
            _mask_select_card,       # Phase 1: SELECT_CARD
            _mask_execute_card,      # Phase 2: EXECUTE_CARD
            _mask_select_market,     # Phase 3: SELECT_MARKET_POS
            _mask_place_spice,       # Phase 4: PLACE_SPICE
            _mask_select_scoring,    # Phase 5: SELECT_SCORING_CARD
            _mask_discard_overflow,  # Phase 6: DISCARD_OVERFLOW
        ],
        state
    )

    return masks


def _mask_choose_action(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for CHOOSE_ACTION phase - which of 4 actions are legal."""
    player = state.current_player
    caravan = state.caravans[player]
    hand_size = state.hand_sizes[player]
    played_size = state.played_sizes[player]

    # Play: legal if has cards in hand
    can_play = hand_size > 0

    # Acquire: always legal (position 0 is free)
    can_acquire = jnp.bool_(True)

    # Rest: legal only if has played cards
    can_rest = played_size > 0

    # Score: legal if any scoring card is affordable
    can_score = _any_scoring_affordable(state, player)

    action_type_mask = jnp.array([can_play, can_acquire, can_rest, can_score])

    # Other masks are all False (not used in this phase)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_select_card(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for SELECT_CARD phase - which hand cards can be played."""
    player = state.current_player
    hand_size = state.hand_sizes[player]

    # Cards 0..hand_size-1 are valid
    card_indices = jnp.arange(MAX_PLAYER_CARDS)
    card_idx_mask = card_indices < hand_size

    # Only card_idx head is active
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_execute_card(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for EXECUTE_CARD phase - depends on card type.

    Spice cards: auto-execute, only DONE valid
    Conversion cards: select spice to upgrade, AGAIN if upgrades remain
    Exchange cards: AGAIN if can afford another, DONE always valid
    """
    player = state.current_player
    caravan = state.caravans[player]
    card = state.selected_card
    card_type = get_card_type(card)
    remaining = state.remaining_upgrades

    # Spice card: just DONE
    spice_mask = _make_spice_execute_mask(caravan)

    # Conversion card: select upgradeable spice, AGAIN/DONE
    conversion_mask = _make_conversion_execute_mask(caravan, remaining)

    # Exchange card: AGAIN/DONE based on affordability
    exchange_mask = _make_exchange_execute_mask(caravan, card)

    # Select mask based on card type
    masks = lax.switch(
        card_type,
        [
            lambda: spice_mask,
            lambda: conversion_mask,
            lambda: exchange_mask,
        ]
    )

    return masks


def _make_spice_execute_mask(caravan: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Mask for executing a spice card - only DONE is valid."""
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)
    # Only DONE (index 1) is valid
    continue_mask = jnp.array([False, True])

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _make_conversion_execute_mask(caravan: jnp.ndarray,
                                   remaining: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Mask for executing a conversion card.

    spice_type: non-brown spices with count > 0
    continue: AGAIN if upgrades remain AND can upgrade, DONE always
    """
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)

    # Spice mask: non-brown with count > 0
    has_spice = caravan > 0
    not_brown = jnp.array([True, True, True, False])
    spice_type_mask = has_spice & not_brown

    # Continue mask: AGAIN if remaining > 0 and can upgrade, DONE always
    can_again = (remaining > 0) & can_apply_conversion(caravan)
    continue_mask = jnp.array([can_again, True])

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _make_exchange_execute_mask(caravan: jnp.ndarray,
                                 card: jnp.ndarray) -> Tuple[jnp.ndarray, ...]:
    """Mask for executing an exchange card.

    continue: AGAIN if can afford another exchange, DONE always
    """
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)

    # Continue mask: AGAIN if can afford, DONE always
    can_again = can_afford_exchange(caravan, card)
    continue_mask = jnp.array([can_again, True])

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_select_market(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for SELECT_MARKET_POS phase - which positions are affordable.

    Position N requires N spices in caravan (position 0 is free).
    """
    player = state.current_player
    caravan = state.caravans[player]
    total_spices = caravan_total(caravan)
    market_size = state.market_size

    # Position i requires i spices
    positions = jnp.arange(NUM_MARKET_SLOTS)
    can_afford = positions <= total_spices

    # Also must be a valid position (within market size)
    is_valid = positions < market_size

    market_pos_mask = can_afford & is_valid

    # Only market_pos head is active
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_place_spice(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for PLACE_SPICE phase - which spice types can be placed.

    Only non-zero spices in caravan can be placed.
    """
    player = state.current_player
    caravan = state.caravans[player]

    # Spices with count > 0
    spice_type_mask = caravan > 0

    # Only spice_type head is active
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_select_scoring(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for SELECT_SCORING_CARD phase - which scoring cards affordable."""
    player = state.current_player
    caravan = state.caravans[player]

    # Check each scoring card
    scoring_idx_mask = _get_affordable_scoring_mask(state, caravan)

    # Only scoring_idx head is active
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    spice_type_mask = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _mask_discard_overflow(state: State) -> Tuple[jnp.ndarray, ...]:
    """Mask for DISCARD_OVERFLOW phase - which spices can be discarded.

    Only non-zero spices can be discarded.
    """
    player = state.current_player
    caravan = state.caravans[player]

    # Spices with count > 0
    spice_type_mask = caravan > 0

    # Only spice_type head is active
    action_type_mask = jnp.zeros(NUM_ACTION_TYPES, dtype=jnp.bool_)
    card_idx_mask = jnp.zeros(MAX_PLAYER_CARDS, dtype=jnp.bool_)
    market_pos_mask = jnp.zeros(NUM_MARKET_SLOTS, dtype=jnp.bool_)
    scoring_idx_mask = jnp.zeros(NUM_SCORING_SLOTS, dtype=jnp.bool_)
    continue_mask = jnp.zeros(NUM_CONTINUE_FLAGS, dtype=jnp.bool_)

    return (action_type_mask, card_idx_mask, market_pos_mask,
            scoring_idx_mask, spice_type_mask, continue_mask)


def _any_scoring_affordable(state: State, player: jnp.ndarray) -> jnp.ndarray:
    """Check if any scoring card is affordable by player."""
    caravan = state.caravans[player]
    mask = _get_affordable_scoring_mask(state, caravan)
    return jnp.any(mask)


def _get_affordable_scoring_mask(state: State, caravan: jnp.ndarray) -> jnp.ndarray:
    """Get mask of which scoring row cards are affordable.

    Args:
        state: Current game state
        caravan: Player's caravan, shape (4,)

    Returns:
        Boolean array, shape (NUM_SCORING_SLOTS,)
    """
    scoring_row = state.scoring_row
    scoring_row_size = state.scoring_row_size

    def check_card(i):
        card = scoring_row[i]
        is_valid = i < scoring_row_size
        affordable = can_afford_scoring_card(caravan, card)
        return is_valid & affordable

    indices = jnp.arange(NUM_SCORING_SLOTS)
    return lax.map(check_card, indices)
