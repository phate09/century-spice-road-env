"""
Pytest fixtures for Century: Spice Road tests.

JIT-compiled functions are created at module level to avoid recompilation.
"""

import pytest
import jax
import jax.numpy as jnp

from century_env import CenturySpiceRoad, State, Phase
from century_env.masks import get_action_mask
from century_env.mechanics import (
    apply_spice_card,
    apply_conversion,
    apply_exchange,
    can_apply_conversion,
    can_apply_exchange,
)
from century_env.rewards import (
    compute_step_reward,
    compute_final_reward,
    compute_final_scores,
    compute_winner_rewards,
)
from century_env.termination import (
    check_game_triggered,
    check_game_over,
    determine_winner,
)

# ============================================================================
# Module-level JIT-compiled functions (compiled ONCE for entire test suite)
# ============================================================================

# Mechanics
jit_apply_spice_card = jax.jit(apply_spice_card)
jit_apply_conversion = jax.jit(apply_conversion)
jit_apply_exchange = jax.jit(apply_exchange)
jit_can_apply_conversion = jax.jit(can_apply_conversion)
jit_can_apply_exchange = jax.jit(can_apply_exchange)

# Masks
jit_get_action_mask = jax.jit(get_action_mask)

# Rewards
jit_compute_step_reward = jax.jit(compute_step_reward)
jit_compute_final_reward = jax.jit(compute_final_reward)
jit_compute_final_scores = jax.jit(compute_final_scores)
jit_compute_winner_rewards = jax.jit(compute_winner_rewards)

# Termination
jit_check_game_triggered = jax.jit(check_game_triggered)
jit_check_game_over = jax.jit(check_game_over)
jit_determine_winner = jax.jit(determine_winner)


# ============================================================================
# Session-scoped fixtures (created ONCE for entire test suite)
# ============================================================================

@pytest.fixture(scope="session")
def prng_key():
    """Fresh PRNG key for tests."""
    return jax.random.PRNGKey(42)


@pytest.fixture(scope="session")
def env_4p():
    """4-player environment."""
    return CenturySpiceRoad(num_players=4)


@pytest.fixture(scope="session")
def env_2p():
    """2-player environment."""
    return CenturySpiceRoad(num_players=2)


@pytest.fixture(scope="session")
def env_5p():
    """5-player environment."""
    return CenturySpiceRoad(num_players=5)


@pytest.fixture(scope="session")
def initial_state_4p(env_4p, prng_key):
    """Initial state for 4-player game."""
    state, _ = env_4p.reset(prng_key)
    return state


@pytest.fixture(scope="session")
def initial_state_2p(env_2p, prng_key):
    """Initial state for 2-player game."""
    state, _ = env_2p.reset(prng_key)
    return state


@pytest.fixture(scope="session")
def jit_reset_4p(env_4p):
    """JIT-compiled reset for 4-player environment."""
    @jax.jit
    def _reset(key):
        return env_4p.reset(key)
    return _reset


@pytest.fixture(scope="session")
def jit_step_4p(env_4p):
    """JIT-compiled step for 4-player environment."""
    @jax.jit
    def _step(state, action):
        return env_4p.step(state, action)
    return _step


@pytest.fixture(scope="session")
def vmap_reset_4p(env_4p):
    """Vmapped reset for 4-player environment."""
    @jax.vmap
    def _batch_reset(key):
        return env_4p.reset(key)
    return _batch_reset


@pytest.fixture(scope="session")
def vmap_step_4p(env_4p):
    """Vmapped step for 4-player environment."""
    @jax.vmap
    def _batch_step(state, action):
        return env_4p.step(state, action)
    return _batch_step


@pytest.fixture(scope="session", autouse=True)
def warmup_jit_cache(env_4p, prng_key, jit_reset_4p, jit_step_4p, vmap_reset_4p, vmap_step_4p):
    """Pre-warm JIT cache once before any tests run.

    This compiles all JIT functions with representative inputs so
    individual tests don't pay the compilation cost.
    """
    # Warm up mechanics
    caravan = jnp.array([2, 1, 0, 0], dtype=jnp.int32)
    card = jnp.array([0, 0, 0, 0, 0, 0, 2, 0, 0, 0], dtype=jnp.int32)
    exchange_card = jnp.array([2, 0, 3, 0, 0, 0, 0, 0, 0, 1], dtype=jnp.int32)

    _ = jit_apply_spice_card(caravan, card)
    _ = jit_apply_conversion(caravan, jnp.int32(0))
    _ = jit_apply_exchange(caravan, exchange_card)
    _ = jit_can_apply_conversion(caravan)
    _ = jit_can_apply_exchange(caravan, exchange_card)

    # Warm up environment reset/step (this triggers JIT compilation)
    state, _ = jit_reset_4p(prng_key)
    action = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.int32)
    _ = jit_step_4p(state, action)

    # Warm up vmap operations
    keys = jax.random.split(prng_key, 10)
    states, _ = vmap_reset_4p(keys)
    actions = jnp.zeros((10, 6), dtype=jnp.int32).at[:, 5].set(1)
    _ = vmap_step_4p(states, actions)

    # Warm up masks
    _ = jit_get_action_mask(state)

    # Warm up rewards
    _ = jit_compute_step_reward(state, state, jnp.int32(0))
    _ = jit_compute_final_reward(state, jnp.int32(0))
    _ = jit_compute_final_scores(state)
    _ = jit_compute_winner_rewards(state)

    # Warm up termination
    _ = jit_check_game_triggered(state)
    _ = jit_check_game_over(state)
    _ = jit_determine_winner(state)


def make_action(action_type=0, card_idx=0, market_pos=0,
                scoring_idx=0, spice_type=0, continue_flag=1):
    """Create a multi-discrete action array."""
    return jnp.array([action_type, card_idx, market_pos,
                      scoring_idx, spice_type, continue_flag], dtype=jnp.int32)


def play_action(card_idx=0):
    """Create a PLAY action."""
    return make_action(action_type=0, card_idx=card_idx)


def acquire_action(market_pos=0):
    """Create an ACQUIRE action."""
    return make_action(action_type=1, market_pos=market_pos)


def rest_action():
    """Create a REST action."""
    return make_action(action_type=2)


def score_action(scoring_idx=0):
    """Create a SCORE action."""
    return make_action(action_type=3, scoring_idx=scoring_idx)


def execute_done_action():
    """Create an EXECUTE done action (for spice cards)."""
    return make_action(continue_flag=1)


def execute_upgrade_action(spice_type=0, done=False):
    """Create an EXECUTE upgrade action (for conversion cards)."""
    return make_action(spice_type=spice_type, continue_flag=1 if done else 0)


def place_spice_action(spice_type=0):
    """Create a PLACE_SPICE action."""
    return make_action(spice_type=spice_type)


def discard_action(spice_type=0):
    """Create a DISCARD_OVERFLOW action."""
    return make_action(spice_type=spice_type)
