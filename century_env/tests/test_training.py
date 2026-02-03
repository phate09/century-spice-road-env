"""End-to-end validation tests with random agents."""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad
from century_env.agents import RandomAgent
from century_env.agents.random_agent import run_random_game, run_random_games


class TestRandomSelfPlay:
    def test_random_game_completes(self, env_4p, prng_key):
        """A random game should complete within reasonable steps."""
        state, steps, done = run_random_game(env_4p, prng_key, max_steps=500)

        assert done, f"Game didn't complete after {steps} steps"
        assert steps > 0, "Game completed with 0 steps"

    def test_random_self_play_10_games(self, env_4p, prng_key):
        """Run 10 random games - all should complete legally."""
        stats = run_random_games(env_4p, prng_key, num_games=10, max_steps=500)

        assert stats["completed"] == 10, \
            f"Only {stats['completed']}/10 games completed"
        assert stats["avg_steps"] > 20, \
            f"Average game too short: {stats['avg_steps']} steps"

    def test_random_game_reasonable_length(self, env_4p, prng_key):
        """Random games should have reasonable length distribution."""
        stats = run_random_games(env_4p, prng_key, num_games=5, max_steps=500)

        # Games shouldn't be too short (would indicate bug)
        assert stats["min_steps"] >= 10, \
            f"Some games too short: min={stats['min_steps']}"

        # Games shouldn't time out
        assert stats["timed_out"] == 0, \
            f"{stats['timed_out']} games timed out"


class TestRandomAgentMethods:
    def test_select_action_returns_valid_shape(self, env_4p, prng_key):
        """Agent should return action with correct shape."""
        state, timestep = env_4p.reset(prng_key)
        agent = RandomAgent()

        key = jax.random.PRNGKey(123)
        action = agent.select_action(key, timestep.observation)

        assert action.shape == (6,)
        assert action.dtype == jnp.int32

    def test_select_action_respects_mask(self, env_4p, prng_key):
        """Agent should only select legal actions."""
        state, timestep = env_4p.reset(prng_key)
        agent = RandomAgent()

        # Run several samples
        key = prng_key
        for _ in range(20):
            key, subkey = jax.random.split(key)
            action = agent.select_action(subkey, timestep.observation)
            masks = timestep.observation.action_mask

            # Check that selected action is legal for each head
            for i, mask in enumerate(masks):
                if jnp.any(mask):
                    assert bool(mask[action[i]]), \
                        f"Action head {i} selected illegal action {action[i]}"


class TestMultiPlayerCounts:
    @pytest.mark.parametrize("num_players", [2, 3, 4, 5])
    def test_random_game_all_player_counts(self, num_players, prng_key):
        """Random games should work for all player counts."""
        env = CenturySpiceRoad(num_players=num_players)
        state, steps, done = run_random_game(env, prng_key, max_steps=500)

        assert done, f"Game with {num_players} players didn't complete"
