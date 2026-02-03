"""Tests for game mechanics functions."""

import jax.numpy as jnp
import pytest

from century_env.mechanics import (
    add_spices_to_caravan,
    remove_spices_from_caravan,
    apply_spice_card,
    apply_conversion,
    can_apply_conversion,
    apply_exchange,
    can_apply_exchange,
    caravan_total,
    has_overflow,
    discard_spice,
)
from century_env.constants import (
    SPICE_YELLOW,
    SPICE_RED,
    SPICE_GREEN,
    SPICE_BROWN,
    CARAVAN_LIMIT,
)


class TestAddRemoveSpices:
    def test_add_spices(self):
        caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
        spices = jnp.array([1, 1, 1, 0], dtype=jnp.int32)
        result = add_spices_to_caravan(caravan, spices)
        assert jnp.array_equal(result, jnp.array([3, 2, 1, 0]))

    def test_remove_spices(self):
        caravan = jnp.array([3, 2, 1, 0], dtype=jnp.int32)
        spices = jnp.array([1, 1, 1, 0], dtype=jnp.int32)
        result = remove_spices_from_caravan(caravan, spices)
        assert jnp.array_equal(result, jnp.array([2, 1, 0, 0]))

    def test_add_empty_spices(self):
        caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
        spices = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        result = add_spices_to_caravan(caravan, spices)
        assert jnp.array_equal(result, caravan)


class TestApplySpiceCard:
    def test_apply_obtain_yellow(self):
        caravan = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        # Spice card: Obtain YY [0, 0, 0, 0, 0, 0, 2, 0, 0, 0]
        card = jnp.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0], dtype=jnp.int32)
        result = apply_spice_card(caravan, card)
        assert jnp.array_equal(result, jnp.array([2, 0, 0, 0]))

    def test_apply_obtain_mixed(self):
        caravan = jnp.array([1, 0, 0, 0], dtype=jnp.int32)
        # Spice card: Obtain YR [0, 0, 0, 0, 0, 0, 1, 1, 0, 0]
        card = jnp.array([0, 0, 0, 0, 0, 0, 1, 1, 0, 0], dtype=jnp.int32)
        result = apply_spice_card(caravan, card)
        assert jnp.array_equal(result, jnp.array([2, 1, 0, 0]))


class TestApplyConversion:
    def test_upgrade_yellow_to_red(self):
        caravan = jnp.array([2, 0, 0, 0], dtype=jnp.int32)
        result = apply_conversion(caravan, jnp.int32(SPICE_YELLOW))
        assert jnp.array_equal(result, jnp.array([1, 1, 0, 0]))

    def test_upgrade_red_to_green(self):
        caravan = jnp.array([0, 2, 0, 0], dtype=jnp.int32)
        result = apply_conversion(caravan, jnp.int32(SPICE_RED))
        assert jnp.array_equal(result, jnp.array([0, 1, 1, 0]))

    def test_upgrade_green_to_brown(self):
        caravan = jnp.array([0, 0, 2, 0], dtype=jnp.int32)
        result = apply_conversion(caravan, jnp.int32(SPICE_GREEN))
        assert jnp.array_equal(result, jnp.array([0, 0, 1, 1]))

    def test_upgrade_brown_unchanged(self):
        """Brown cannot be upgraded - should return unchanged."""
        caravan = jnp.array([0, 0, 0, 2], dtype=jnp.int32)
        result = apply_conversion(caravan, jnp.int32(SPICE_BROWN))
        assert jnp.array_equal(result, caravan)

    def test_upgrade_preserves_other_spices(self):
        caravan = jnp.array([2, 1, 1, 1], dtype=jnp.int32)
        result = apply_conversion(caravan, jnp.int32(SPICE_YELLOW))
        assert jnp.array_equal(result, jnp.array([1, 2, 1, 1]))


class TestCanApplyConversion:
    def test_can_upgrade_with_yellow(self):
        caravan = jnp.array([2, 0, 0, 0], dtype=jnp.int32)
        assert bool(can_apply_conversion(caravan))

    def test_can_upgrade_with_red(self):
        caravan = jnp.array([0, 2, 0, 0], dtype=jnp.int32)
        assert bool(can_apply_conversion(caravan))

    def test_can_upgrade_with_green(self):
        caravan = jnp.array([0, 0, 2, 0], dtype=jnp.int32)
        assert bool(can_apply_conversion(caravan))

    def test_cannot_upgrade_only_brown(self):
        caravan = jnp.array([0, 0, 0, 2], dtype=jnp.int32)
        assert not bool(can_apply_conversion(caravan))

    def test_cannot_upgrade_empty(self):
        caravan = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        assert not bool(can_apply_conversion(caravan))


class TestApplyExchange:
    def test_simple_exchange(self):
        caravan = jnp.array([3, 0, 0, 0], dtype=jnp.int32)
        # Exchange: YYY -> B [2, 0, 3, 0, 0, 0, 0, 0, 0, 1]
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        result = apply_exchange(caravan, card)
        assert jnp.array_equal(result, jnp.array([0, 0, 0, 1]))

    def test_exchange_with_leftover(self):
        caravan = jnp.array([5, 0, 0, 0], dtype=jnp.int32)
        # Exchange: YYY -> B
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        result = apply_exchange(caravan, card)
        assert jnp.array_equal(result, jnp.array([2, 0, 0, 1]))


class TestCanApplyExchange:
    def test_can_afford_exchange(self):
        caravan = jnp.array([3, 0, 0, 0], dtype=jnp.int32)
        # Exchange: YYY -> B
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        assert bool(can_apply_exchange(caravan, card))

    def test_cannot_afford_exchange(self):
        caravan = jnp.array([2, 0, 0, 0], dtype=jnp.int32)
        # Exchange: YYY -> B (need 3 yellow)
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        assert not bool(can_apply_exchange(caravan, card))

    def test_can_afford_with_exact_spices(self):
        caravan = jnp.array([2, 2, 0, 0], dtype=jnp.int32)
        # Exchange: YY RR -> G [2, 0, 2, 2, 0, 0, 0, 0, 1, 0]
        card = jnp.array([2, 0, 2, 2, 0, 0, 0, 0, 1, 0], dtype=jnp.int32)
        assert bool(can_apply_exchange(caravan, card))


class TestCaravanTotal:
    def test_empty_caravan(self):
        caravan = jnp.array([0, 0, 0, 0], dtype=jnp.int32)
        assert int(caravan_total(caravan)) == 0

    def test_mixed_caravan(self):
        caravan = jnp.array([3, 2, 1, 1], dtype=jnp.int32)
        assert int(caravan_total(caravan)) == 7

    def test_full_caravan(self):
        caravan = jnp.array([4, 3, 2, 1], dtype=jnp.int32)
        assert int(caravan_total(caravan)) == 10


class TestHasOverflow:
    def test_no_overflow_at_limit(self):
        caravan = jnp.array([4, 3, 2, 1], dtype=jnp.int32)  # Total = 10
        assert not bool(has_overflow(caravan, CARAVAN_LIMIT))

    def test_overflow_above_limit(self):
        caravan = jnp.array([4, 3, 2, 2], dtype=jnp.int32)  # Total = 11
        assert bool(has_overflow(caravan, CARAVAN_LIMIT))

    def test_no_overflow_below_limit(self):
        caravan = jnp.array([3, 2, 1, 0], dtype=jnp.int32)  # Total = 6
        assert not bool(has_overflow(caravan, CARAVAN_LIMIT))


class TestDiscardSpice:
    def test_discard_yellow(self):
        caravan = jnp.array([3, 2, 1, 0], dtype=jnp.int32)
        result = discard_spice(caravan, jnp.int32(SPICE_YELLOW))
        assert jnp.array_equal(result, jnp.array([2, 2, 1, 0]))

    def test_discard_brown(self):
        caravan = jnp.array([3, 2, 1, 1], dtype=jnp.int32)
        result = discard_spice(caravan, jnp.int32(SPICE_BROWN))
        assert jnp.array_equal(result, jnp.array([3, 2, 1, 0]))
