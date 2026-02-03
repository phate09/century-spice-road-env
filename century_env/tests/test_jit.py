"""JIT compilation smoke tests.

Verify that all public functions can be JIT-compiled without
ConcretizationTypeError.

Uses pre-compiled JIT functions from conftest to avoid redundant compilation.
"""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad

# Import pre-compiled JIT functions from conftest
from century_env.tests.conftest import (
    jit_apply_spice_card,
    jit_apply_conversion,
    jit_apply_exchange,
    jit_can_apply_conversion,
    jit_can_apply_exchange,
    jit_get_action_mask,
    jit_compute_step_reward,
    jit_compute_final_reward,
    jit_compute_final_scores,
    jit_compute_winner_rewards,
    jit_check_game_triggered,
    jit_check_game_over,
    jit_determine_winner,
)


class TestMechanicsJIT:
    def test_apply_spice_card_jittable(self):
        caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
        card = jnp.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0], dtype=jnp.int32)
        result = jit_apply_spice_card(caravan, card)
        assert result.shape == (4,)

    def test_apply_conversion_jittable(self):
        caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
        result = jit_apply_conversion(caravan, jnp.int32(0))
        assert result.shape == (4,)

    def test_apply_exchange_jittable(self):
        caravan = jnp.array([3, 0, 0, 0], dtype=jnp.int32)
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        result = jit_apply_exchange(caravan, card)
        assert result.shape == (4,)

    def test_can_apply_conversion_jittable(self):
        caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
        result = jit_can_apply_conversion(caravan)
        assert result.dtype == jnp.bool_

    def test_can_apply_exchange_jittable(self):
        caravan = jnp.array([3, 0, 0, 0], dtype=jnp.int32)
        card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        result = jit_can_apply_exchange(caravan, card)
        assert result.dtype == jnp.bool_


class TestMasksJIT:
    def test_get_action_mask_jittable(self, initial_state_4p):
        masks = jit_get_action_mask(initial_state_4p)
        assert len(masks) == 6


class TestRewardsJIT:
    def test_compute_step_reward_jittable(self, initial_state_4p):
        reward = jit_compute_step_reward(initial_state_4p, initial_state_4p, jnp.int32(0))
        assert reward.dtype == jnp.float32

    def test_compute_final_reward_jittable(self, initial_state_4p):
        reward = jit_compute_final_reward(initial_state_4p, jnp.int32(0))
        assert reward.dtype == jnp.float32

    def test_compute_final_scores_jittable(self, initial_state_4p):
        scores = jit_compute_final_scores(initial_state_4p)
        assert scores.shape == (5,)

    def test_compute_winner_rewards_jittable(self, initial_state_4p):
        rewards = jit_compute_winner_rewards(initial_state_4p)
        assert rewards.shape == (5,)


class TestTerminationJIT:
    def test_check_game_triggered_jittable(self, initial_state_4p):
        result = jit_check_game_triggered(initial_state_4p)
        assert result.dtype == jnp.bool_

    def test_check_game_over_jittable(self, initial_state_4p):
        result = jit_check_game_over(initial_state_4p)
        assert result.dtype == jnp.bool_

    def test_determine_winner_jittable(self, initial_state_4p):
        result = jit_determine_winner(initial_state_4p)
        assert result.dtype == jnp.int32


class TestEnvironmentJIT:
    def test_reset_jittable(self, jit_reset_4p):
        key = jax.random.PRNGKey(42)
        state, timestep = jit_reset_4p(key)
        assert state.current_player.dtype == jnp.int32

    def test_step_jittable(self, jit_reset_4p, jit_step_4p, prng_key):
        state, _ = jit_reset_4p(prng_key)
        action = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        next_state, timestep = jit_step_4p(state, action)
        assert next_state.phase.dtype == jnp.int32


class TestVmapOperations:
    def test_vmap_reset_multiple_envs(self, vmap_reset_4p):
        """Test that reset can be vmapped over multiple keys."""
        keys = jax.random.split(jax.random.PRNGKey(42), 10)
        states, timesteps = vmap_reset_4p(keys)

        # Should have 10 independent states
        assert states.current_player.shape == (10,)
        assert states.hands.shape == (10, 5, 25, 10)

    def test_vmap_step_multiple_envs(self, vmap_reset_4p, vmap_step_4p):
        """Test that step can be vmapped over batched states."""
        keys = jax.random.split(jax.random.PRNGKey(42), 10)
        states, _ = vmap_reset_4p(keys)

        # Create batch of actions
        actions = jnp.zeros((10, 6), dtype=jnp.int32)
        actions = actions.at[:, 0].set(0)  # PLAY
        actions = actions.at[:, 5].set(1)  # DONE

        next_states, timesteps = vmap_step_4p(states, actions)

        # Should have 10 independent next states
        assert next_states.phase.shape == (10,)
