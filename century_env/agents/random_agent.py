"""
Century: Spice Road - Random Agent

A baseline agent that samples uniformly from legal actions.
"""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax

from century_env.types import Observation, ActionMask


def sample_masked_action(key: chex.PRNGKey, mask: jnp.ndarray) -> jnp.ndarray:
    """Sample a random action from a boolean mask.

    Args:
        key: JAX random key
        mask: Boolean array indicating legal actions

    Returns:
        int32 scalar - index of sampled action
    """
    # Convert mask to probabilities
    mask_float = mask.astype(jnp.float32)
    total = jnp.sum(mask_float)

    # Handle case where no actions are legal (shouldn't happen in valid game)
    total = jnp.maximum(total, 1.0)
    probs = mask_float / total

    # Sample from categorical
    idx = jax.random.choice(key, jnp.arange(len(mask)), p=probs)
    return idx.astype(jnp.int32)


class RandomAgent:
    """Agent that samples uniformly from legal actions.

    This is a baseline agent useful for:
    - Testing environment correctness
    - Benchmarking against learned policies
    - Generating random rollouts for exploration
    """

    def select_action(
        self,
        key: chex.PRNGKey,
        observation: Observation
    ) -> jnp.ndarray:
        """Select a random legal action.

        Args:
            key: JAX random key
            observation: Current observation with action_mask

        Returns:
            Multi-discrete action array, shape (6,)
        """
        masks = observation.action_mask

        # Split key for each action head
        keys = jax.random.split(key, 6)

        # Sample from each head
        action = jnp.array([
            sample_masked_action(keys[0], masks[0]),
            sample_masked_action(keys[1], masks[1]),
            sample_masked_action(keys[2], masks[2]),
            sample_masked_action(keys[3], masks[3]),
            sample_masked_action(keys[4], masks[4]),
            sample_masked_action(keys[5], masks[5]),
        ], dtype=jnp.int32)

        return action

    def select_action_from_mask(
        self,
        key: chex.PRNGKey,
        action_mask: ActionMask
    ) -> jnp.ndarray:
        """Select a random legal action from mask tuple.

        Args:
            key: JAX random key
            action_mask: Tuple of 6 boolean mask arrays

        Returns:
            Multi-discrete action array, shape (6,)
        """
        keys = jax.random.split(key, 6)

        action = jnp.array([
            sample_masked_action(keys[0], action_mask[0]),
            sample_masked_action(keys[1], action_mask[1]),
            sample_masked_action(keys[2], action_mask[2]),
            sample_masked_action(keys[3], action_mask[3]),
            sample_masked_action(keys[4], action_mask[4]),
            sample_masked_action(keys[5], action_mask[5]),
        ], dtype=jnp.int32)

        return action


def run_random_game(env, key: chex.PRNGKey, max_steps: int = 500):
    """Run a complete game with random actions.

    Args:
        env: CenturySpiceRoad environment
        key: JAX random key
        max_steps: Maximum steps before terminating

    Returns:
        Tuple of (final_state, total_steps, done)
    """
    agent = RandomAgent()

    key, reset_key = jax.random.split(key)
    state, timestep = env.reset(reset_key)

    step_count = 0
    done = False

    while not done and step_count < max_steps:
        # Get action mask from timestep
        masks = timestep.observation.action_mask

        # Sample action
        key, action_key = jax.random.split(key)
        action = agent.select_action_from_mask(action_key, masks)

        # Step environment
        state, timestep = env.step(state, action)
        step_count += 1
        done = timestep.last()

    return state, step_count, done


def run_random_games(env, key: chex.PRNGKey, num_games: int = 100,
                     max_steps: int = 500):
    """Run multiple random games and collect statistics.

    Args:
        env: CenturySpiceRoad environment
        key: JAX random key
        num_games: Number of games to run
        max_steps: Maximum steps per game

    Returns:
        Dict with game statistics
    """
    completed = 0
    timed_out = 0
    step_counts = []

    for i in range(num_games):
        key, game_key = jax.random.split(key)
        _, steps, done = run_random_game(env, game_key, max_steps)

        step_counts.append(steps)
        if done:
            completed += 1
        else:
            timed_out += 1

    return {
        "num_games": num_games,
        "completed": completed,
        "timed_out": timed_out,
        "completion_rate": completed / num_games,
        "avg_steps": sum(step_counts) / len(step_counts),
        "min_steps": min(step_counts),
        "max_steps": max(step_counts),
    }
