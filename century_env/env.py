"""
Century: Spice Road - Jumanji Environment

Main environment class implementing the Jumanji Environment interface.
"""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep, StepType, restart, transition, termination

from century_env.constants import (
    MAX_PLAYERS,
    MIN_PLAYERS,
    MAX_PLAYER_CARDS,
    MAX_SCORED_CARDS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    NUM_SPICE_TYPES,
    NUM_DECK_TRADER_CARDS,
    NUM_SCORING_CARDS,
    STARTING_SPICES,
    SCORING_CARDS_TO_WIN,
    CARAVAN_LIMIT,
)
from century_env.types import State, Observation, Phase, ActionType
from century_env.cards import (
    ALL_TRADER_CARDS,
    STARTING_CARDS,
    DECK_TRADER_CARDS,
    SCORING_CARDS,
)
from century_env.specs import observation_spec, action_spec, discount_spec, reward_spec
from century_env.masks import get_action_mask
from century_env.transitions import (
    transition_choose_action,
    transition_execute_card,
    transition_place_spice,
    transition_discard_overflow,
)
from century_env.rewards import (
    compute_step_reward,
    compute_final_reward,
    compute_winner_rewards,
)


class CenturySpiceRoad(Environment):
    """Jumanji environment for Century: Spice Road board game.

    A fully-featured RL environment supporting:
    - 2-5 players
    - Complete game rules
    - Ego-centric observations
    - Multi-discrete action space
    """

    def __init__(self, num_players: int = 4):
        """Initialize the environment.

        Args:
            num_players: Number of players (2-5)
        """
        if not MIN_PLAYERS <= num_players <= MAX_PLAYERS:
            raise ValueError(f"num_players must be {MIN_PLAYERS}-{MAX_PLAYERS}, "
                           f"got {num_players}")
        self._num_players = num_players

    @property
    def num_players(self) -> int:
        """Return the number of players."""
        return self._num_players

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment to initial state.

        Args:
            key: JAX random key

        Returns:
            Tuple of (initial_state, initial_timestep)
        """
        key, deck_key, scoring_key = jax.random.split(key, 3)

        # Shuffle trader deck (excluding starting cards)
        deck_indices = jax.random.permutation(deck_key, NUM_DECK_TRADER_CARDS)
        shuffled_deck = DECK_TRADER_CARDS[deck_indices]

        # Shuffle scoring deck
        scoring_indices = jax.random.permutation(scoring_key, NUM_SCORING_CARDS)
        shuffled_scoring = SCORING_CARDS[scoring_indices]

        # Deal starting cards to each player
        # Each player gets cards at indices 0 and 1 of ALL_TRADER_CARDS
        hands = jnp.zeros((MAX_PLAYERS, MAX_PLAYER_CARDS, 10), dtype=jnp.int32)
        hands = hands.at[:, 0].set(STARTING_CARDS[0])
        hands = hands.at[:, 1].set(STARTING_CARDS[1])
        hand_sizes = jnp.full((MAX_PLAYERS,), 2, dtype=jnp.int32)

        # Set starting spices (varies by player position)
        caravans = STARTING_SPICES.copy()

        # Setup market row (6 cards from top of shuffled deck)
        market_cards = jnp.zeros((NUM_MARKET_SLOTS, 10), dtype=jnp.int32)
        market_cards = market_cards.at[:6].set(shuffled_deck[:6])
        market_spices = jnp.zeros((NUM_MARKET_SLOTS, NUM_SPICE_TYPES), dtype=jnp.int32)
        trader_deck = jnp.zeros((NUM_DECK_TRADER_CARDS, 10), dtype=jnp.int32)
        trader_deck = trader_deck.at[:NUM_DECK_TRADER_CARDS - 6].set(shuffled_deck[6:])
        trader_deck_size = jnp.int32(NUM_DECK_TRADER_CARDS - 6)

        # Setup scoring row (5 cards from top of shuffled scoring deck)
        scoring_row = jnp.zeros((NUM_SCORING_SLOTS, 5), dtype=jnp.int32)
        scoring_row = scoring_row.at[:5].set(shuffled_scoring[:5])
        scoring_deck = jnp.zeros((NUM_SCORING_CARDS, 5), dtype=jnp.int32)
        scoring_deck = scoring_deck.at[:NUM_SCORING_CARDS - 5].set(shuffled_scoring[5:])
        scoring_deck_size = jnp.int32(NUM_SCORING_CARDS - 5)

        # Coins: 2 × num_players each
        initial_coins = 2 * self._num_players

        state = State(
            # Game flow
            current_player=jnp.int32(0),
            phase=jnp.int32(Phase.CHOOSE_ACTION),
            num_players=jnp.int32(self._num_players),
            game_triggered=jnp.bool_(False),
            trigger_player=jnp.int32(-1),
            # Per-player state
            hands=hands,
            hand_sizes=hand_sizes,
            played_piles=jnp.zeros((MAX_PLAYERS, MAX_PLAYER_CARDS, 10), dtype=jnp.int32),
            played_sizes=jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32),
            caravans=caravans,
            scored_cards=jnp.zeros((MAX_PLAYERS, MAX_SCORED_CARDS, 5), dtype=jnp.int32),
            scored_counts=jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32),
            gold_coins=jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32),
            silver_coins=jnp.zeros((MAX_PLAYERS,), dtype=jnp.int32),
            # Market state
            market_cards=market_cards,
            market_spices=market_spices,
            market_size=jnp.int32(NUM_MARKET_SLOTS),
            scoring_row=scoring_row,
            scoring_row_size=jnp.int32(NUM_SCORING_SLOTS),
            trader_deck=trader_deck,
            trader_deck_size=trader_deck_size,
            scoring_deck=scoring_deck,
            scoring_deck_size=scoring_deck_size,
            gold_remaining=jnp.int32(initial_coins),
            silver_remaining=jnp.int32(initial_coins),
            # Action state
            selected_card_idx=jnp.int32(0),
            selected_card=jnp.zeros(10, dtype=jnp.int32),
            remaining_upgrades=jnp.int32(0),
            acquire_target_position=jnp.int32(0),
            spices_placed_count=jnp.int32(0),
            # PRNG
            key=key,
        )

        observation = self._state_to_observation(state, jnp.int32(0))
        timestep = restart(observation, extras={"action_mask": observation.action_mask})

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Take a step in the environment.

        Args:
            state: Current game state
            action: Multi-discrete action array, shape (6,)

        Returns:
            Tuple of (next_state, timestep)
        """
        player = state.current_player

        # Dispatch based on phase
        next_state = lax.switch(
            state.phase,
            [
                lambda s, a: transition_choose_action(s, a),  # CHOOSE_ACTION
                lambda s, a: _transition_select_card(s, a),   # SELECT_CARD (unused)
                lambda s, a: transition_execute_card(s, a),   # EXECUTE_CARD
                lambda s, a: _transition_select_market(s, a), # SELECT_MARKET_POS (unused)
                lambda s, a: transition_place_spice(s, a),    # PLACE_SPICE
                lambda s, a: _transition_select_scoring(s, a),# SELECT_SCORING_CARD (unused)
                lambda s, a: transition_discard_overflow(s, a),# DISCARD_OVERFLOW
            ],
            state, action
        )

        # Compute reward
        reward = compute_step_reward(state, next_state, player)

        # Check for game end
        is_done = self._check_game_over(next_state)

        # Add final rewards if game over
        final_reward = lax.cond(
            is_done,
            lambda: compute_final_reward(next_state, player),
            lambda: jnp.float32(0.0)
        )
        total_reward = reward + final_reward

        # Build observation for next player
        next_player = next_state.current_player
        observation = self._state_to_observation(next_state, next_player)

        # Build timestep
        discount = lax.cond(is_done, lambda: jnp.float32(0.0), lambda: jnp.float32(1.0))

        timestep = lax.cond(
            is_done,
            lambda: termination(
                reward=total_reward,
                observation=observation,
                extras={"action_mask": observation.action_mask}
            ),
            lambda: transition(
                reward=total_reward,
                observation=observation,
                discount=discount,
                extras={"action_mask": observation.action_mask}
            )
        )

        return next_state, timestep

    def _check_game_over(self, state: State) -> jnp.ndarray:
        """Check if the game is over.

        Game ends when:
        - game_triggered is True AND
        - current_player has cycled back to trigger_player AND
        - we're at CHOOSE_ACTION phase (turn just ended)
        """
        triggered = state.game_triggered
        back_to_trigger = state.current_player == state.trigger_player
        at_turn_start = state.phase == Phase.CHOOSE_ACTION

        return triggered & back_to_trigger & at_turn_start

    def _state_to_observation(self, state: State, player: jnp.ndarray) -> Observation:
        """Convert full state to ego-centric observation for a player.

        The player sees themselves as "player 0" and opponents are rotated.
        """
        num_players = state.num_players

        # My state
        my_hand = state.hands[player]
        my_hand_size = state.hand_sizes[player]
        my_played = state.played_piles[player]
        my_played_size = state.played_sizes[player]
        my_caravan = state.caravans[player]
        my_scored = state.scored_cards[player]
        my_scored_count = state.scored_counts[player]
        my_gold = state.gold_coins[player]
        my_silver = state.silver_coins[player]

        # Opponent state (rotated)
        # Opponent 0 is player+1, opponent 1 is player+2, etc.
        def get_opponent_idx(i):
            return (player + 1 + i) % MAX_PLAYERS

        opp_indices = lax.map(get_opponent_idx, jnp.arange(MAX_PLAYERS - 1))

        # Check which opponents are active
        opp_active = opp_indices < num_players

        # Get opponent data
        opp_hand_sizes = state.hand_sizes[opp_indices]
        opp_played = state.played_piles[opp_indices]
        opp_played_sizes = state.played_sizes[opp_indices]
        opp_caravans = state.caravans[opp_indices]
        opp_scored = state.scored_cards[opp_indices]
        opp_scored_counts = state.scored_counts[opp_indices]
        opp_gold = state.gold_coins[opp_indices]
        opp_silver = state.silver_coins[opp_indices]

        # Get action mask
        action_mask = get_action_mask(state)

        return Observation(
            # My state
            my_hand=my_hand,
            my_hand_size=my_hand_size,
            my_played=my_played,
            my_played_size=my_played_size,
            my_caravan=my_caravan,
            my_scored=my_scored,
            my_scored_count=my_scored_count,
            my_gold=my_gold,
            my_silver=my_silver,
            # Opponent state
            opp_active=opp_active,
            opp_hand_sizes=opp_hand_sizes,
            opp_played=opp_played,
            opp_played_sizes=opp_played_sizes,
            opp_caravans=opp_caravans,
            opp_scored=opp_scored,
            opp_scored_counts=opp_scored_counts,
            opp_gold=opp_gold,
            opp_silver=opp_silver,
            # Market
            market_cards=state.market_cards,
            market_spices=state.market_spices,
            market_size=state.market_size,
            scoring_row=state.scoring_row,
            scoring_row_size=state.scoring_row_size,
            # Supply
            gold_remaining=state.gold_remaining,
            silver_remaining=state.silver_remaining,
            trader_deck_size=state.trader_deck_size,
            scoring_deck_size=state.scoring_deck_size,
            # Game info
            num_players=state.num_players,
            current_phase=state.phase,
            # Action state
            selected_card=state.selected_card,
            remaining_upgrades=state.remaining_upgrades,
            acquire_target_position=state.acquire_target_position,
            spices_placed_count=state.spices_placed_count,
            # Action mask
            action_mask=action_mask,
        )

    def observation_spec(self) -> specs.Spec:
        """Return observation specification."""
        return observation_spec()

    def action_spec(self) -> specs.MultiDiscreteArray:
        """Return action specification."""
        return action_spec()

    def discount_spec(self) -> specs.BoundedArray:
        """Return discount specification."""
        return discount_spec()

    def reward_spec(self) -> specs.Array:
        """Return reward specification."""
        return reward_spec()

    def render(self, state: State, mode: str = "human") -> Optional[str]:
        """Render the current state.

        Args:
            state: Current game state
            mode: Render mode ("human" for text, "rgb_array" not supported)

        Returns:
            String representation if mode is "human", None otherwise
        """
        if mode == "human":
            from century_env.render import render_state
            return render_state(state)
        return None

    def close(self) -> None:
        """Clean up environment resources.

        No-op for this environment (no external resources).
        """
        pass


# Placeholder transition functions for unused phases
# (These phases are handled directly in transition_choose_action)

def _transition_select_card(state: State, action: jnp.ndarray) -> State:
    """SELECT_CARD phase - not used (combined with CHOOSE_ACTION)."""
    return state


def _transition_select_market(state: State, action: jnp.ndarray) -> State:
    """SELECT_MARKET_POS phase - not used (combined with CHOOSE_ACTION)."""
    return state


def _transition_select_scoring(state: State, action: jnp.ndarray) -> State:
    """SELECT_SCORING_CARD phase - not used (combined with CHOOSE_ACTION)."""
    return state
