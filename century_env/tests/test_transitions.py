"""Tests for state transitions."""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad, Phase, ActionType
from century_env.tests.conftest import (
    make_action, play_action, acquire_action, rest_action, score_action,
    execute_done_action, execute_upgrade_action, place_spice_action,
)


class TestPlayCardTransition:
    def test_play_moves_card_to_played_pile(self, env_4p, prng_key):
        """Playing a card should move it from hand to played pile."""
        state, _ = env_4p.reset(prng_key)

        initial_hand_size = int(state.hand_sizes[0])
        initial_played_size = int(state.played_sizes[0])

        # Play card at index 0
        action = play_action(card_idx=0)
        state, _ = env_4p.step(state, action)

        # Card moved to played pile (still in EXECUTE phase)
        assert int(state.hand_sizes[0]) == initial_hand_size - 1
        assert int(state.played_sizes[0]) == initial_played_size + 1
        assert int(state.phase) == Phase.EXECUTE_CARD

    def test_execute_spice_card_adds_spices(self, env_4p, prng_key):
        """Executing a spice card should add output spices to caravan."""
        state, _ = env_4p.reset(prng_key)

        # Player 0 starts with 3 yellow
        initial_caravan = state.caravans[0].copy()

        # Card 0 is "Obtain YY" - adds 2 yellow
        action = play_action(card_idx=0)
        state, _ = env_4p.step(state, action)

        # Execute (DONE)
        action = execute_done_action()
        state, _ = env_4p.step(state, action)

        # Caravan should have 2 more yellow
        expected = initial_caravan.at[0].add(2)
        assert jnp.array_equal(state.caravans[0], expected)


class TestRestTransition:
    def test_rest_returns_played_to_hand(self, env_4p, prng_key):
        """Rest should return all played cards to hand."""
        state, _ = env_4p.reset(prng_key)

        # Play a card first
        action = play_action(card_idx=0)
        state, _ = env_4p.step(state, action)
        action = execute_done_action()
        state, _ = env_4p.step(state, action)

        # Complete other players' turns
        for _ in range(3):
            action = play_action(card_idx=0)
            state, _ = env_4p.step(state, action)
            action = execute_done_action()
            state, _ = env_4p.step(state, action)

        # Back to player 0
        assert int(state.current_player) == 0
        played_before_rest = int(state.played_sizes[0])
        hand_before_rest = int(state.hand_sizes[0])
        assert played_before_rest > 0

        # Rest
        action = rest_action()
        state, _ = env_4p.step(state, action)

        # All played cards should be back in hand
        assert int(state.played_sizes[0]) == 0
        assert int(state.hand_sizes[0]) == hand_before_rest + played_before_rest


class TestAcquireTransition:
    def test_acquire_position_0_takes_card(self, env_4p, prng_key):
        """Acquiring from position 0 should be free and take the card."""
        state, _ = env_4p.reset(prng_key)

        initial_hand_size = int(state.hand_sizes[0])
        card_at_pos_0 = state.market_cards[0].copy()

        # Acquire from position 0
        action = acquire_action(market_pos=0)
        state, _ = env_4p.step(state, action)

        # Hand should have one more card
        assert int(state.hand_sizes[0]) == initial_hand_size + 1
        # The acquired card should be in hand
        new_card = state.hands[0, initial_hand_size]
        assert jnp.array_equal(new_card, card_at_pos_0)

    def test_acquire_position_1_requires_spice_placement(self, env_4p, prng_key):
        """Acquiring from position 1 should require placing 1 spice."""
        state, _ = env_4p.reset(prng_key)

        # Acquire from position 1
        action = acquire_action(market_pos=1)
        state, _ = env_4p.step(state, action)

        # Should be in PLACE_SPICE phase
        assert int(state.phase) == Phase.PLACE_SPICE
        assert int(state.acquire_target_position) == 1
        assert int(state.spices_placed_count) == 0

    def test_acquire_collects_spices_on_card(self, env_4p, prng_key):
        """Acquiring should collect any spices on the target card."""
        state, _ = env_4p.reset(prng_key)

        # Manually place some spices on market card 1
        market_spices = state.market_spices.at[1, 0].set(2)  # 2 yellow
        state = state.replace(market_spices=market_spices)

        initial_caravan = state.caravans[0].copy()

        # Acquire from position 1 (need to place 1 spice first)
        action = acquire_action(market_pos=1)
        state, _ = env_4p.step(state, action)

        # Place a yellow spice on position 0
        action = place_spice_action(spice_type=0)
        state, _ = env_4p.step(state, action)

        # After placing, acquire finalizes
        # Caravan should have: initial - 1 yellow (placed) + 2 yellow (collected)
        # Net: +1 yellow
        expected_yellow = int(initial_caravan[0]) + 1
        assert int(state.caravans[0, 0]) == expected_yellow


class TestScoreTransition:
    def test_score_removes_spices_adds_card(self, env_4p, prng_key):
        """Scoring should remove required spices and add scoring card."""
        state, _ = env_4p.reset(prng_key)

        # Give player 0 enough spices for any scoring card
        caravan = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)
        state = state.replace(caravans=caravans)

        # Get requirements of first scoring card
        scoring_card = state.scoring_row[0]
        requirements = scoring_card[1:5]

        initial_scored = int(state.scored_counts[0])

        # Score card at index 0
        action = score_action(scoring_idx=0)
        state, _ = env_4p.step(state, action)

        # Scored count should increase
        assert int(state.scored_counts[0]) == initial_scored + 1
        # Spices should be removed
        expected_caravan = caravan - requirements
        assert jnp.array_equal(state.caravans[0], expected_caravan)

    def test_score_position_0_awards_gold(self, env_4p, prng_key):
        """Scoring from position 0 should award a gold coin."""
        state, _ = env_4p.reset(prng_key)

        # Give player 0 enough spices
        caravan = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)
        state = state.replace(caravans=caravans)

        initial_gold = int(state.gold_coins[0])
        initial_gold_remaining = int(state.gold_remaining)

        # Score from position 0
        action = score_action(scoring_idx=0)
        state, _ = env_4p.step(state, action)

        # Should have gained gold
        assert int(state.gold_coins[0]) == initial_gold + 1
        assert int(state.gold_remaining) == initial_gold_remaining - 1

    def test_score_position_1_awards_silver(self, env_4p, prng_key):
        """Scoring from position 1 should award a silver coin (if gold available)."""
        state, _ = env_4p.reset(prng_key)

        # Give player 0 enough spices
        caravan = jnp.array([5, 5, 5, 5], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)
        state = state.replace(caravans=caravans)

        initial_silver = int(state.silver_coins[0])

        # Score from position 1
        action = score_action(scoring_idx=1)
        state, _ = env_4p.step(state, action)

        # Should have gained silver
        assert int(state.silver_coins[0]) == initial_silver + 1


class TestAdvanceTurn:
    def test_turn_cycles_players(self, env_4p, prng_key):
        """Turns should cycle through all players."""
        state, _ = env_4p.reset(prng_key)
        assert int(state.current_player) == 0

        # Each player plays their spice card
        for expected_next in [1, 2, 3, 0]:
            action = play_action(card_idx=0)
            state, _ = env_4p.step(state, action)
            action = execute_done_action()
            state, _ = env_4p.step(state, action)

            assert int(state.current_player) == expected_next


class TestGameTrigger:
    def test_scoring_triggers_game_end(self, env_4p, prng_key):
        """Scoring enough cards should trigger game end."""
        state, _ = env_4p.reset(prng_key)

        # Give player 0 lots of spices
        caravan = jnp.array([10, 10, 10, 10], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)

        # Give player 0 almost enough scored cards (4 out of 5 for 4-player)
        scored_counts = state.scored_counts.at[0].set(4)
        state = state.replace(caravans=caravans, scored_counts=scored_counts)

        assert not bool(state.game_triggered)

        # Score the 5th card
        action = score_action(scoring_idx=0)
        state, _ = env_4p.step(state, action)

        # Game should be triggered
        assert bool(state.game_triggered)
        assert int(state.trigger_player) == 0
