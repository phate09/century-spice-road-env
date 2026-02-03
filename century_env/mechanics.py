"""
Century: Spice Road - Core Game Mechanics

Pure JAX functions for card execution. All functions are jittable.
"""

import jax.numpy as jnp
from jax import lax

from century_env.constants import (
    SPICE_BROWN,
    NUM_SPICE_TYPES,
)
from century_env.cards import (
    get_card_type,
    get_card_input,
    get_card_output,
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)


def add_spices_to_caravan(caravan: jnp.ndarray, spices: jnp.ndarray) -> jnp.ndarray:
    """Add spices to caravan.

    Args:
        caravan: Current caravan state, shape (4,) [y, r, g, b]
        spices: Spices to add, shape (4,) [y, r, g, b]

    Returns:
        Updated caravan, shape (4,)

    Note: Does NOT enforce caravan limit - caller must handle overflow.
    """
    return caravan + spices


def remove_spices_from_caravan(caravan: jnp.ndarray, spices: jnp.ndarray) -> jnp.ndarray:
    """Remove spices from caravan.

    Args:
        caravan: Current caravan state, shape (4,) [y, r, g, b]
        spices: Spices to remove, shape (4,) [y, r, g, b]

    Returns:
        Updated caravan, shape (4,)

    Note: Assumes caller has verified sufficient spices exist.
    """
    return caravan - spices


def apply_spice_card(caravan: jnp.ndarray, card: jnp.ndarray) -> jnp.ndarray:
    """Apply a spice (gain) card to caravan.

    Spice cards add output spices with no cost.

    Args:
        caravan: Current caravan state, shape (4,)
        card: Trader card data, shape (10,)

    Returns:
        Updated caravan, shape (4,)
    """
    output = get_card_output(card)
    return add_spices_to_caravan(caravan, output)


def apply_conversion(caravan: jnp.ndarray, spice_idx: jnp.ndarray) -> jnp.ndarray:
    """Apply one upgrade step to a single spice.

    Upgrades: Yellow -> Red -> Green -> Brown
    Brown cannot be upgraded (returns caravan unchanged).

    Args:
        caravan: Current caravan state, shape (4,)
        spice_idx: Index of spice to upgrade (0-3)

    Returns:
        Updated caravan, shape (4,)
    """
    # If brown selected, no-op
    is_brown = spice_idx == SPICE_BROWN

    # Create decrement for source spice
    decrement = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.int32).at[spice_idx].set(1)

    # Create increment for target spice (next in sequence)
    target_idx = jnp.minimum(spice_idx + 1, SPICE_BROWN)
    increment = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.int32).at[target_idx].set(1)

    # Apply both (but not if brown)
    new_caravan = lax.cond(
        is_brown,
        lambda c: c,
        lambda c: c - decrement + increment,
        caravan
    )

    return new_caravan


def can_apply_conversion(caravan: jnp.ndarray) -> jnp.ndarray:
    """Check if any upgradeable spice exists (non-brown with count > 0).

    Args:
        caravan: Current caravan state, shape (4,)

    Returns:
        Boolean scalar - True if upgrade possible
    """
    # Sum of yellow, red, green (indices 0, 1, 2)
    upgradeable = jnp.sum(caravan[:3])
    return upgradeable > 0


def apply_exchange(caravan: jnp.ndarray, card: jnp.ndarray) -> jnp.ndarray:
    """Execute one exchange: subtract input spices, add output spices.

    Args:
        caravan: Current caravan state, shape (4,)
        card: Exchange card data, shape (10,)

    Returns:
        Updated caravan, shape (4,)
    """
    input_spices = get_card_input(card)
    output_spices = get_card_output(card)

    new_caravan = remove_spices_from_caravan(caravan, input_spices)
    new_caravan = add_spices_to_caravan(new_caravan, output_spices)

    return new_caravan


def can_apply_exchange(caravan: jnp.ndarray, card: jnp.ndarray) -> jnp.ndarray:
    """Check if exchange card can be applied (have required input spices).

    Args:
        caravan: Current caravan state, shape (4,)
        card: Exchange card data, shape (10,)

    Returns:
        Boolean scalar - True if exchange possible
    """
    input_spices = get_card_input(card)
    return jnp.all(caravan >= input_spices)


def caravan_total(caravan: jnp.ndarray) -> jnp.ndarray:
    """Get total spice count in caravan.

    Args:
        caravan: Current caravan state, shape (4,)

    Returns:
        int32 scalar - total spices
    """
    return jnp.sum(caravan)


def has_overflow(caravan: jnp.ndarray, limit: int = 10) -> jnp.ndarray:
    """Check if caravan exceeds limit.

    Args:
        caravan: Current caravan state, shape (4,)
        limit: Maximum allowed spices (default 10)

    Returns:
        Boolean scalar - True if overflow
    """
    return caravan_total(caravan) > limit


def discard_spice(caravan: jnp.ndarray, spice_idx: jnp.ndarray) -> jnp.ndarray:
    """Discard one spice of specified type.

    Args:
        caravan: Current caravan state, shape (4,)
        spice_idx: Index of spice to discard (0-3)

    Returns:
        Updated caravan, shape (4,)
    """
    decrement = jnp.zeros(NUM_SPICE_TYPES, dtype=jnp.int32).at[spice_idx].set(1)
    return caravan - decrement
