"""
Century: Spice Road - Reward Computation

Incremental and end-game reward calculations.
All functions are jittable.
"""

import jax.numpy as jnp
from jax import lax

from century_env.types import State
from century_env.constants import MAX_PLAYERS


def compute_step_reward(state: State, next_state: State,
                        player: jnp.ndarray) -> jnp.ndarray:
    """Compute incremental reward for a player after a step.

    Rewards:
        - Points from newly scored card
        - Gold coins gained × 3
        - Silver coins gained × 1

    Args:
        state: State before the step
        next_state: State after the step
        player: Player index to compute reward for

    Returns:
        float32 scalar reward
    """
    # Detect if player scored a card
    old_count = state.scored_counts[player]
    new_count = next_state.scored_counts[player]
    scored = new_count > old_count

    # Get points from newly scored card (if any)
    # The new card is at index new_count - 1
    new_card_idx = new_count - 1
    new_card = next_state.scored_cards[player, new_card_idx]
    card_points = new_card[0]  # Points field

    points_reward = lax.cond(
        scored,
        lambda: card_points.astype(jnp.float32),
        lambda: jnp.float32(0.0)
    )

    # Gold coin delta (worth 3 each)
    gold_delta = next_state.gold_coins[player] - state.gold_coins[player]
    gold_reward = gold_delta.astype(jnp.float32) * 3.0

    # Silver coin delta (worth 1 each)
    silver_delta = next_state.silver_coins[player] - state.silver_coins[player]
    silver_reward = silver_delta.astype(jnp.float32)

    return points_reward + gold_reward + silver_reward


def compute_final_reward(state: State, player: jnp.ndarray) -> jnp.ndarray:
    """Compute end-game bonus reward for remaining spices.

    Non-yellow spices in caravan: 1 point each (R + G + B)

    Args:
        state: Final game state
        player: Player index

    Returns:
        float32 scalar reward
    """
    caravan = state.caravans[player]
    # Sum of red, green, brown (indices 1, 2, 3)
    non_yellow = jnp.sum(caravan[1:])
    return non_yellow.astype(jnp.float32)


def compute_final_scores(state: State) -> jnp.ndarray:
    """Compute final scores for all players.

    Final score = sum of:
        - Scoring card points
        - Gold coins × 3
        - Silver coins × 1
        - Non-yellow spices × 1

    Args:
        state: Final game state

    Returns:
        float32 array, shape (MAX_PLAYERS,)
    """
    def player_score(player):
        # Sum scoring card points
        scored = state.scored_cards[player]
        scored_count = state.scored_counts[player]

        def sum_points(carry, i):
            is_valid = i < scored_count
            points = scored[i, 0]
            return carry + lax.cond(is_valid, lambda: points, lambda: jnp.int32(0)), None

        card_points, _ = lax.scan(sum_points, jnp.int32(0), jnp.arange(10))

        # Coins
        gold = state.gold_coins[player]
        silver = state.silver_coins[player]
        coin_points = gold * 3 + silver

        # Non-yellow spices
        caravan = state.caravans[player]
        spice_points = jnp.sum(caravan[1:])

        return (card_points + coin_points + spice_points).astype(jnp.float32)

    scores = lax.map(player_score, jnp.arange(MAX_PLAYERS))
    return scores


def compute_winner_rewards(state: State) -> jnp.ndarray:
    """Compute per-player win/loss rewards.

    Winner gets +1, others get -1.
    Tie-breaker: later turn order wins.

    Args:
        state: Final game state

    Returns:
        float32 array, shape (MAX_PLAYERS,)
    """
    scores = compute_final_scores(state)
    num_players = state.num_players

    # Find max score
    # Mask inactive players with -inf
    active_mask = jnp.arange(MAX_PLAYERS) < num_players
    masked_scores = jnp.where(active_mask, scores, -jnp.inf)
    max_score = jnp.max(masked_scores)

    # Find players with max score
    has_max = (scores == max_score) & active_mask

    # Tie-breaker: later turn order wins
    # Player indices are turn order (0 goes first)
    # So we want the highest index among tied players
    tied_indices = jnp.where(has_max, jnp.arange(MAX_PLAYERS), -1)
    winner = jnp.max(tied_indices)

    # Rewards: +1 for winner, -1 for others (active only)
    is_winner = jnp.arange(MAX_PLAYERS) == winner
    rewards = jnp.where(is_winner, 1.0, -1.0)
    rewards = jnp.where(active_mask, rewards, 0.0)

    return rewards.astype(jnp.float32)
