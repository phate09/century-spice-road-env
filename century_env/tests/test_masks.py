"""Tests for action masking."""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad, Phase, ActionType
from century_env.masks import get_action_mask


class TestChooseActionMask:
    def test_play_legal_with_cards(self, initial_state_4p):
        """Play should be legal when player has cards in hand."""
        masks = get_action_mask(initial_state_4p)
        action_type_mask = masks[0]
        # Player has 2 starting cards
        assert bool(action_type_mask[ActionType.PLAY])

    def test_acquire_always_legal(self, initial_state_4p):
        """Acquire should always be legal (position 0 is free)."""
        masks = get_action_mask(initial_state_4p)
        action_type_mask = masks[0]
        assert bool(action_type_mask[ActionType.ACQUIRE])

    def test_rest_illegal_without_played_cards(self, initial_state_4p):
        """Rest should be illegal when no cards are played."""
        masks = get_action_mask(initial_state_4p)
        action_type_mask = masks[0]
        # At start, no cards are played
        assert not bool(action_type_mask[ActionType.REST])

    def test_rest_legal_after_playing(self, env_4p, prng_key):
        """Rest should be legal after playing a card."""
        state, _ = env_4p.reset(prng_key)

        # Play a card (card 0 is the starting spice card)
        from century_env.tests.conftest import play_action, execute_done_action
        action = play_action(card_idx=0)
        state, _ = env_4p.step(state, action)

        # Execute the spice card (press DONE)
        action = execute_done_action()
        state, _ = env_4p.step(state, action)

        # Now it's player 1's turn, but player 0 has played a card
        # Let player 1 also play and finish
        for _ in range(3):  # Players 1, 2, 3
            action = play_action(card_idx=0)
            state, _ = env_4p.step(state, action)
            action = execute_done_action()
            state, _ = env_4p.step(state, action)

        # Now back to player 0 who has a played card
        assert int(state.current_player) == 0
        assert int(state.played_sizes[0]) > 0

        masks = get_action_mask(state)
        action_type_mask = masks[0]
        assert bool(action_type_mask[ActionType.REST])


class TestMarketMask:
    def test_position_0_always_affordable(self, initial_state_4p):
        """Position 0 requires 0 spices, always affordable."""
        # Force to SELECT_MARKET_POS phase (we'll test the mask directly)
        state = initial_state_4p.replace(phase=jnp.int32(Phase.SELECT_MARKET_POS))
        masks = get_action_mask(state)
        market_mask = masks[2]
        assert bool(market_mask[0])

    def test_higher_positions_require_spices(self, initial_state_4p):
        """Higher positions require more spices."""
        # Player 0 starts with 3 yellow, can afford positions 0-3
        state = initial_state_4p.replace(phase=jnp.int32(Phase.SELECT_MARKET_POS))
        masks = get_action_mask(state)
        market_mask = masks[2]

        # Can afford 0, 1, 2, 3 (needs 0, 1, 2, 3 spices with 3 yellow)
        assert bool(market_mask[0])
        assert bool(market_mask[1])
        assert bool(market_mask[2])
        assert bool(market_mask[3])
        # Cannot afford 4, 5 (needs 4, 5 spices)
        assert not bool(market_mask[4])
        assert not bool(market_mask[5])


class TestDiscardMask:
    def test_only_nonzero_spices_discardable(self):
        """Can only discard spices you have."""
        from century_env.tests.conftest import prng_key
        from century_env import CenturySpiceRoad

        env = CenturySpiceRoad(num_players=4)
        key = jax.random.PRNGKey(42)
        state, _ = env.reset(key)

        # Set up overflow scenario: player has Y=5, R=4, G=3, B=0
        caravan = jnp.array([5, 4, 3, 0], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)
        state = state.replace(
            caravans=caravans,
            phase=jnp.int32(Phase.DISCARD_OVERFLOW)
        )

        masks = get_action_mask(state)
        spice_mask = masks[4]

        # Can discard Y, R, G but not B
        assert bool(spice_mask[0])  # Yellow
        assert bool(spice_mask[1])  # Red
        assert bool(spice_mask[2])  # Green
        assert not bool(spice_mask[3])  # Brown (have 0)


class TestAtLeastOneActionLegal:
    def test_always_at_least_one_action_legal(self, env_4p, prng_key):
        """At any game state, at least one action should be legal."""
        state, _ = env_4p.reset(prng_key)

        # Play through several random actions
        key = prng_key
        for _ in range(50):
            masks = get_action_mask(state)

            # Check that at least one action is legal across all heads
            # In CHOOSE_ACTION, check action_type_mask
            if int(state.phase) == Phase.CHOOSE_ACTION:
                assert jnp.any(masks[0]), "No legal actions in CHOOSE_ACTION"
            elif int(state.phase) == Phase.EXECUTE_CARD:
                # Either spice_mask or continue_mask should have legal options
                has_spice = jnp.any(masks[4])
                has_continue = jnp.any(masks[5])
                assert has_spice or has_continue, "No legal actions in EXECUTE_CARD"
            elif int(state.phase) == Phase.PLACE_SPICE:
                assert jnp.any(masks[4]), "No legal spices to place"
            elif int(state.phase) == Phase.DISCARD_OVERFLOW:
                assert jnp.any(masks[4]), "No legal spices to discard"

            # Take a random legal action
            key, subkey = jax.random.split(key)
            action = _sample_legal_action(masks, subkey)
            state, timestep = env_4p.step(state, action)

            # Stop if game ended
            if timestep.last():
                break


def _sample_legal_action(masks, key):
    """Sample a random legal action from masks."""
    # Simple implementation: just pick first legal option for each head
    action = jnp.zeros(6, dtype=jnp.int32)

    for i, mask in enumerate(masks):
        if jnp.any(mask):
            # Pick first legal option
            legal_indices = jnp.where(mask, jnp.arange(len(mask)), -1)
            first_legal = jnp.max(jnp.where(legal_indices >= 0, -legal_indices, -1000))
            first_legal = -first_legal
            action = action.at[i].set(first_legal if first_legal >= 0 else 0)

    return action
