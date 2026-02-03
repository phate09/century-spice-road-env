"""Integration tests for the CenturySpiceRoad environment."""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad, Phase, STARTING_SPICES
from century_env.constants import NUM_MARKET_SLOTS, NUM_SCORING_SLOTS


class TestEnvironmentInit:
    def test_create_4_player_env(self):
        env = CenturySpiceRoad(num_players=4)
        assert env.num_players == 4

    def test_create_2_player_env(self):
        env = CenturySpiceRoad(num_players=2)
        assert env.num_players == 2

    def test_create_5_player_env(self):
        env = CenturySpiceRoad(num_players=5)
        assert env.num_players == 5

    def test_invalid_player_count(self):
        with pytest.raises(ValueError):
            CenturySpiceRoad(num_players=1)
        with pytest.raises(ValueError):
            CenturySpiceRoad(num_players=6)


class TestReset:
    def test_reset_initializes_valid_state(self, env_4p, prng_key):
        state, timestep = env_4p.reset(prng_key)

        # Check basic game flow
        assert int(state.current_player) == 0
        assert int(state.phase) == Phase.CHOOSE_ACTION
        assert int(state.num_players) == 4
        assert not bool(state.game_triggered)

    def test_reset_deals_starting_cards(self, env_4p, prng_key):
        state, _ = env_4p.reset(prng_key)

        # Each player should have 2 starting cards
        for p in range(4):
            assert int(state.hand_sizes[p]) == 2

    def test_reset_sets_starting_spices(self, env_4p, prng_key):
        state, _ = env_4p.reset(prng_key)

        # Check starting spices match rules
        for p in range(4):
            expected = STARTING_SPICES[p]
            actual = state.caravans[p]
            assert jnp.array_equal(actual, expected)

    def test_reset_sets_up_market(self, env_4p, prng_key):
        state, _ = env_4p.reset(prng_key)

        assert int(state.market_size) == NUM_MARKET_SLOTS
        assert int(state.scoring_row_size) == NUM_SCORING_SLOTS

    def test_reset_sets_coins(self, env_4p, prng_key):
        state, _ = env_4p.reset(prng_key)

        # 4 players: 2 * 4 = 8 coins each
        assert int(state.gold_remaining) == 8
        assert int(state.silver_remaining) == 8


class TestStep:
    def test_step_with_valid_action_succeeds(self, env_4p, prng_key):
        state, timestep = env_4p.reset(prng_key)

        # Play card 0 (should be valid)
        action = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.int32)  # PLAY card_idx=0
        next_state, next_timestep = env_4p.step(state, action)

        # State should have changed
        assert int(next_state.phase) != int(state.phase) or \
               int(next_state.current_player) != int(state.current_player)

    def test_step_returns_timestep(self, env_4p, prng_key):
        state, _ = env_4p.reset(prng_key)

        action = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.int32)
        _, timestep = env_4p.step(state, action)

        assert hasattr(timestep, 'observation')
        assert hasattr(timestep, 'reward')
        assert hasattr(timestep, 'discount')


class TestRandomGameCompletion:
    def test_random_game_completes_legally(self, env_4p, prng_key):
        """A game with random actions should eventually complete."""
        state, timestep = env_4p.reset(prng_key)

        key = prng_key
        max_turns = 500
        turn_count = 0

        while not timestep.last() and turn_count < max_turns:
            # Get action mask
            masks = timestep.observation.action_mask

            # Sample random legal action
            key, subkey = jax.random.split(key)
            action = _sample_legal_action(masks, subkey)

            state, timestep = env_4p.step(state, action)
            turn_count += 1

        # Game should have ended
        assert turn_count < max_turns, f"Game didn't end after {max_turns} turns"


class TestDeterminism:
    def test_same_key_same_result(self, env_4p):
        """Same PRNG key should produce identical results."""
        key = jax.random.PRNGKey(12345)

        state1, ts1 = env_4p.reset(key)
        state2, ts2 = env_4p.reset(key)

        # States should be identical
        assert jnp.array_equal(state1.hands, state2.hands)
        assert jnp.array_equal(state1.market_cards, state2.market_cards)
        assert jnp.array_equal(state1.scoring_row, state2.scoring_row)


class TestEgoCentricObservation:
    def test_observation_rotation(self, env_4p, prng_key):
        """Player N should see rotated view with self as 'player 0'."""
        state, _ = env_4p.reset(prng_key)

        # Get observation for player 0
        obs0 = env_4p._state_to_observation(state, jnp.int32(0))

        # Get observation for player 2
        obs2 = env_4p._state_to_observation(state, jnp.int32(2))

        # Player 2's "my_caravan" should be player 2's actual caravan
        assert jnp.array_equal(obs2.my_caravan, state.caravans[2])

        # Player 2's opponent 0 should be player 3
        # opp_caravans[0] should equal state.caravans[3]
        assert jnp.array_equal(obs2.opp_caravans[0], state.caravans[3])


class TestVariablePlayerCount:
    @pytest.mark.parametrize("num_players", [2, 3, 4, 5])
    def test_all_player_counts(self, num_players, prng_key):
        """All player counts should work correctly."""
        env = CenturySpiceRoad(num_players=num_players)
        state, timestep = env.reset(prng_key)

        assert int(state.num_players) == num_players

        # Play a few turns
        key = prng_key
        for _ in range(10):
            masks = timestep.observation.action_mask
            key, subkey = jax.random.split(key)
            action = _sample_legal_action(masks, subkey)
            state, timestep = env.step(state, action)

            if timestep.last():
                break


class TestEdgeCases:
    def test_empty_trader_deck_handling(self, env_4p, prng_key):
        """Market shouldn't refill when trader deck is empty."""
        state, _ = env_4p.reset(prng_key)

        # Empty the trader deck
        state = state.replace(trader_deck_size=jnp.int32(0))

        # Acquire a card (should work but not refill)
        action = jnp.array([1, 0, 0, 0, 0, 0], dtype=jnp.int32)  # ACQUIRE pos=0
        next_state, _ = env_4p.step(state, action)

        # Market size should decrease by 1
        assert int(next_state.market_size) == int(state.market_size) - 1

    def test_empty_scoring_deck_handling(self, env_4p, prng_key):
        """Scoring row shouldn't refill when deck is empty."""
        state, _ = env_4p.reset(prng_key)

        # Empty the scoring deck and give player enough spices
        caravan = jnp.array([10, 10, 10, 10], dtype=jnp.int32)
        caravans = state.caravans.at[0].set(caravan)
        state = state.replace(
            scoring_deck_size=jnp.int32(0),
            caravans=caravans
        )

        # Score a card
        action = jnp.array([3, 0, 0, 0, 0, 0], dtype=jnp.int32)  # SCORE idx=0
        next_state, _ = env_4p.step(state, action)

        # Scoring row size should decrease
        assert int(next_state.scoring_row_size) == int(state.scoring_row_size) - 1


def _sample_legal_action(masks, key):
    """Sample a random legal action from masks."""
    action = jnp.zeros(6, dtype=jnp.int32)

    for i, mask in enumerate(masks):
        if jnp.any(mask):
            # Get indices of legal actions
            legal = jnp.where(mask, 1, 0)
            probs = legal / jnp.sum(legal)
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(subkey, jnp.arange(len(mask)), p=probs)
            action = action.at[i].set(idx)

    return action
