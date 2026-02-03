"""
Century: Spice Road - Game Termination Logic

Game end detection and winner determination.
All functions are jittable.
"""

import jax.numpy as jnp
from jax import lax

from century_env.types import State, Phase
from century_env.constants import MAX_PLAYERS, SCORING_CARDS_TO_WIN
from century_env.rewards import compute_final_scores


def check_game_triggered(state: State) -> jnp.ndarray:
    """Check if any player has scored enough cards to trigger game end.

    Returns:
        Boolean scalar - True if game end was triggered this turn
    """
    num_players = state.num_players
    threshold = SCORING_CARDS_TO_WIN[num_players]

    # Check if any active player has reached threshold
    def check_player(i):
        is_active = i < num_players
        count = state.scored_counts[i]
        reached = count >= threshold
        return is_active & reached

    any_triggered = jnp.any(lax.map(check_player, jnp.arange(MAX_PLAYERS)))
    return any_triggered


def get_trigger_player(state: State) -> jnp.ndarray:
    """Get the player who triggered the game end.

    Returns the first player (by index) who has reached the scoring threshold.

    Returns:
        int32 scalar - player index, or -1 if not triggered
    """
    num_players = state.num_players
    threshold = SCORING_CARDS_TO_WIN[num_players]

    def find_trigger(carry, i):
        found_idx = carry
        is_active = i < num_players
        count = state.scored_counts[i]
        reached = count >= threshold
        should_update = is_active & reached & (found_idx == -1)
        new_idx = lax.cond(should_update, lambda: i, lambda: found_idx)
        return new_idx, None

    trigger_player, _ = lax.scan(find_trigger, jnp.int32(-1), jnp.arange(MAX_PLAYERS))
    return trigger_player


def check_game_over(state: State) -> jnp.ndarray:
    """Check if the game is over.

    Game ends when:
    - Game was triggered (game_triggered is True)
    - Current turn has returned to the trigger player
    - We're at the start of a turn (CHOOSE_ACTION phase)

    This means the round completes after trigger.

    Returns:
        Boolean scalar - True if game is over
    """
    triggered = state.game_triggered
    current = state.current_player
    trigger = state.trigger_player
    at_turn_start = state.phase == Phase.CHOOSE_ACTION

    # Game is over when we've come back to trigger player
    back_to_trigger = current == trigger

    return triggered & back_to_trigger & at_turn_start


def determine_winner(state: State) -> jnp.ndarray:
    """Determine the winning player.

    Winner is the player with highest final score.
    Tie-breaker: player with later turn order wins (higher index).

    Args:
        state: Final game state

    Returns:
        int32 scalar - winning player index
    """
    scores = compute_final_scores(state)
    num_players = state.num_players

    # Mask inactive players
    active_mask = jnp.arange(MAX_PLAYERS) < num_players
    masked_scores = jnp.where(active_mask, scores, -jnp.inf)

    # Find max score
    max_score = jnp.max(masked_scores)

    # Find all players with max score
    has_max = (scores == max_score) & active_mask

    # Tie-breaker: later turn order (higher index) wins
    # Set non-max players to -1 and take max of indices
    tied_indices = jnp.where(has_max, jnp.arange(MAX_PLAYERS), -1)
    winner = jnp.max(tied_indices)

    return winner.astype(jnp.int32)


def get_player_rankings(state: State) -> jnp.ndarray:
    """Get player rankings (1st, 2nd, etc.) based on final scores.

    Tie-breaker: later turn order ranks higher.

    Args:
        state: Final game state

    Returns:
        int32 array, shape (MAX_PLAYERS,) - rank for each player (1 = winner)
        Inactive players get rank 0.
    """
    scores = compute_final_scores(state)
    num_players = state.num_players

    # Active mask
    active_mask = jnp.arange(MAX_PLAYERS) < num_players

    # For ranking with tie-breaker, we need to sort by (score DESC, index DESC)
    # We can do this by creating a composite key
    # Score is primary (higher = better), index is secondary (higher = better)
    # Multiply score by a large factor and add index

    composite = scores * 1000 + jnp.arange(MAX_PLAYERS).astype(jnp.float32)
    composite = jnp.where(active_mask, composite, -jnp.inf)

    # Get sorted indices (descending)
    sorted_indices = jnp.argsort(-composite)

    # Create ranks
    ranks = jnp.zeros(MAX_PLAYERS, dtype=jnp.int32)

    def assign_rank(rank_val, sorted_idx):
        player_idx = sorted_indices[sorted_idx]
        is_active = player_idx < num_players
        return lax.cond(
            is_active,
            lambda: ranks.at[player_idx].set(rank_val),
            lambda: ranks
        )

    # Assign ranks 1, 2, 3, ...
    def fold_ranks(ranks, i):
        player_idx = sorted_indices[i]
        is_active = active_mask[player_idx]
        rank = i + 1
        new_ranks = lax.cond(
            is_active,
            lambda: ranks.at[player_idx].set(rank),
            lambda: ranks
        )
        return new_ranks, None

    final_ranks, _ = lax.scan(fold_ranks, ranks, jnp.arange(MAX_PLAYERS))

    return final_ranks
