"""Tests for game constants."""

import jax.numpy as jnp
import pytest

from century_env.constants import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    MAX_PLAYER_CARDS,
    NUM_STARTING_CARDS,
    NUM_DECK_TRADER_CARDS,
    NUM_ALL_TRADER_CARDS,
    NUM_SCORING_CARDS,
    MAX_SCORED_CARDS,
    NUM_SPICE_TYPES,
    CARAVAN_LIMIT,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    STARTING_SPICES,
    SCORING_CARDS_TO_WIN,
    SPICE_YELLOW,
    SPICE_RED,
    SPICE_GREEN,
    SPICE_BROWN,
)


class TestPlayerConstants:
    def test_player_bounds(self):
        assert MIN_PLAYERS == 2
        assert MAX_PLAYERS == 5
        assert MIN_PLAYERS < MAX_PLAYERS


class TestCardConstants:
    def test_trader_card_counts(self):
        assert NUM_STARTING_CARDS == 2
        assert NUM_DECK_TRADER_CARDS == 43
        assert NUM_ALL_TRADER_CARDS == NUM_STARTING_CARDS + NUM_DECK_TRADER_CARDS

    def test_scoring_card_count(self):
        assert NUM_SCORING_CARDS == 36

    def test_player_card_limits(self):
        assert MAX_PLAYER_CARDS == 25
        assert MAX_SCORED_CARDS == 10


class TestSpiceConstants:
    def test_spice_count(self):
        assert NUM_SPICE_TYPES == 4

    def test_caravan_limit(self):
        assert CARAVAN_LIMIT == 10

    def test_spice_indices(self):
        assert SPICE_YELLOW == 0
        assert SPICE_RED == 1
        assert SPICE_GREEN == 2
        assert SPICE_BROWN == 3


class TestMarketConstants:
    def test_market_sizes(self):
        assert NUM_MARKET_SLOTS == 6
        assert NUM_SCORING_SLOTS == 5


class TestStartingSpices:
    def test_shape(self):
        assert STARTING_SPICES.shape == (5, 4)

    def test_player_0_spices(self):
        # Player 0: 3 yellow
        assert jnp.array_equal(STARTING_SPICES[0], jnp.array([3, 0, 0, 0]))

    def test_player_1_spices(self):
        # Player 1: 4 yellow
        assert jnp.array_equal(STARTING_SPICES[1], jnp.array([4, 0, 0, 0]))

    def test_player_2_spices(self):
        # Player 2: 4 yellow
        assert jnp.array_equal(STARTING_SPICES[2], jnp.array([4, 0, 0, 0]))

    def test_player_3_spices(self):
        # Player 3: 3 yellow + 1 red
        assert jnp.array_equal(STARTING_SPICES[3], jnp.array([3, 1, 0, 0]))

    def test_player_4_spices(self):
        # Player 4: 3 yellow + 1 red
        assert jnp.array_equal(STARTING_SPICES[4], jnp.array([3, 1, 0, 0]))

    def test_total_starting_spices(self):
        # All starting spices are within caravan limit
        totals = jnp.sum(STARTING_SPICES, axis=1)
        assert jnp.all(totals <= CARAVAN_LIMIT)


class TestScoringThresholds:
    def test_shape(self):
        assert SCORING_CARDS_TO_WIN.shape == (6,)

    def test_2_player_threshold(self):
        assert int(SCORING_CARDS_TO_WIN[2]) == 6

    def test_3_player_threshold(self):
        assert int(SCORING_CARDS_TO_WIN[3]) == 6

    def test_4_player_threshold(self):
        assert int(SCORING_CARDS_TO_WIN[4]) == 5

    def test_5_player_threshold(self):
        assert int(SCORING_CARDS_TO_WIN[5]) == 5

    def test_unused_indices_are_zero(self):
        assert int(SCORING_CARDS_TO_WIN[0]) == 0
        assert int(SCORING_CARDS_TO_WIN[1]) == 0
