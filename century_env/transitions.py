"""
Century: Spice Road - State Transitions

State update logic for each action type. All functions are jittable.
"""

import jax
import jax.numpy as jnp
from jax import lax

from century_env.constants import (
    MAX_PLAYER_CARDS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    NUM_SPICE_TYPES,
    CARAVAN_LIMIT,
    SCORING_CARDS_TO_WIN,
)
from century_env.types import State, Phase, ActionType
from century_env.cards import (
    get_card_type,
    get_card_upgrades,
    get_scoring_requirements,
    get_scoring_points,
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)
from century_env.mechanics import (
    apply_spice_card,
    apply_conversion,
    apply_exchange,
    caravan_total,
    has_overflow,
    discard_spice,
)


# =============================================================================
# CHOOSE ACTION PHASE
# =============================================================================

def transition_choose_action(state: State, action: jnp.ndarray) -> State:
    """Process CHOOSE_ACTION phase - dispatch to appropriate action handler.

    Action format: [action_type, card_idx, market_pos, scoring_idx, spice_type, continue]
    """
    action_type = action[0]
    card_idx = action[1]
    market_pos = action[2]
    scoring_idx = action[3]

    # Dispatch based on action type
    new_state = lax.switch(
        action_type,
        [
            lambda s: _start_play(s, card_idx),
            lambda s: _start_acquire(s, market_pos),
            lambda s: _do_rest(s),
            lambda s: _start_score(s, scoring_idx),
        ],
        state
    )

    return new_state


def _start_play(state: State, card_idx: jnp.ndarray) -> State:
    """Start Play action - select card and move to execute phase.

    The card_idx comes from CHOOSE_ACTION (combined with Play action type).
    """
    player = state.current_player
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]

    # Get the selected card
    card = hand[card_idx]
    card_type = get_card_type(card)

    # Move card from hand to played pile
    played = state.played_piles[player]
    played_size = state.played_sizes[player]

    # Add card to played pile
    new_played = played.at[played_size].set(card)
    new_played_size = played_size + 1

    # Remove card from hand (shift remaining cards left)
    new_hand = _remove_card_from_array(hand, card_idx, hand_size)
    new_hand_size = hand_size - 1

    # Update hands and played piles
    new_hands = state.hands.at[player].set(new_hand)
    new_hand_sizes = state.hand_sizes.at[player].set(new_hand_size)
    new_played_piles = state.played_piles.at[player].set(new_played)
    new_played_sizes = state.played_sizes.at[player].set(new_played_size)

    # Get number of upgrades for conversion cards
    upgrades = lax.cond(
        card_type == CARD_TYPE_CONVERSION,
        lambda: get_card_upgrades(card),
        lambda: jnp.int32(0)
    )

    return state.replace(
        hands=new_hands,
        hand_sizes=new_hand_sizes,
        played_piles=new_played_piles,
        played_sizes=new_played_sizes,
        selected_card_idx=card_idx,
        selected_card=card,
        remaining_upgrades=upgrades,
        phase=jnp.int32(Phase.EXECUTE_CARD)
    )


def _start_acquire(state: State, market_pos: jnp.ndarray) -> State:
    """Start Acquire action - set target position and maybe place spices."""
    # If position 0, finalize immediately (no spices to place)
    return lax.cond(
        market_pos == 0,
        lambda s: _finalize_acquire(s, jnp.int32(0)),
        lambda s: s.replace(
            acquire_target_position=market_pos,
            spices_placed_count=jnp.int32(0),
            phase=jnp.int32(Phase.PLACE_SPICE)
        ),
        state
    )


def _do_rest(state: State) -> State:
    """Execute Rest action - return all played cards to hand."""
    player = state.current_player
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]
    played = state.played_piles[player]
    played_size = state.played_sizes[player]

    # Copy played cards to hand
    def copy_card(i, h):
        should_copy = i < played_size
        card = played[i]
        dest_idx = hand_size + i
        return lax.cond(
            should_copy,
            lambda: h.at[dest_idx].set(card),
            lambda: h
        )

    new_hand = lax.fori_loop(0, MAX_PLAYER_CARDS, copy_card, hand)
    new_hand_size = hand_size + played_size

    # Clear played pile
    new_played = jnp.zeros_like(played)
    new_played_size = jnp.int32(0)

    # Update state
    new_hands = state.hands.at[player].set(new_hand)
    new_hand_sizes = state.hand_sizes.at[player].set(new_hand_size)
    new_played_piles = state.played_piles.at[player].set(new_played)
    new_played_sizes = state.played_sizes.at[player].set(new_played_size)

    new_state = state.replace(
        hands=new_hands,
        hand_sizes=new_hand_sizes,
        played_piles=new_played_piles,
        played_sizes=new_played_sizes
    )

    return _advance_turn(new_state)


def _start_score(state: State, scoring_idx: jnp.ndarray) -> State:
    """Start Score action - claim scoring card immediately."""
    return _finalize_score(state, scoring_idx)


# =============================================================================
# EXECUTE CARD PHASE
# =============================================================================

def transition_execute_card(state: State, action: jnp.ndarray) -> State:
    """Process EXECUTE_CARD phase based on card type.

    Action format: [action_type, card_idx, market_pos, scoring_idx, spice_type, continue]
    """
    card = state.selected_card
    card_type = get_card_type(card)
    spice_idx = action[4]
    continue_flag = action[5]

    new_state = lax.switch(
        card_type,
        [
            lambda s: _execute_spice(s),
            lambda s: _execute_conversion(s, spice_idx, continue_flag),
            lambda s: _execute_exchange(s, continue_flag),
        ],
        state
    )

    return new_state


def _execute_spice(state: State) -> State:
    """Execute a spice (gain) card."""
    player = state.current_player
    caravan = state.caravans[player]
    card = state.selected_card

    # Apply spice card effect
    new_caravan = apply_spice_card(caravan, card)
    new_caravans = state.caravans.at[player].set(new_caravan)

    new_state = state.replace(caravans=new_caravans)
    return _finish_card_execution(new_state)


def _execute_conversion(state: State, spice_idx: jnp.ndarray,
                        continue_flag: jnp.ndarray) -> State:
    """Execute one upgrade of a conversion card."""
    player = state.current_player
    caravan = state.caravans[player]
    remaining = state.remaining_upgrades

    # Apply upgrade
    new_caravan = apply_conversion(caravan, spice_idx)
    new_caravans = state.caravans.at[player].set(new_caravan)
    new_remaining = remaining - 1

    new_state = state.replace(
        caravans=new_caravans,
        remaining_upgrades=new_remaining
    )

    # DONE (1) = finish, AGAIN (0) = continue
    return lax.cond(
        continue_flag == 1,
        _finish_card_execution,
        lambda s: s,  # Stay in EXECUTE_CARD phase
        new_state
    )


def _execute_exchange(state: State, continue_flag: jnp.ndarray) -> State:
    """Execute one iteration of an exchange card."""
    player = state.current_player
    caravan = state.caravans[player]
    card = state.selected_card

    # Apply exchange
    new_caravan = apply_exchange(caravan, card)
    new_caravans = state.caravans.at[player].set(new_caravan)

    new_state = state.replace(caravans=new_caravans)

    # DONE (1) = finish, AGAIN (0) = continue
    return lax.cond(
        continue_flag == 1,
        _finish_card_execution,
        lambda s: s,  # Stay in EXECUTE_CARD phase
        new_state
    )


def _finish_card_execution(state: State) -> State:
    """Finish card execution - check overflow then advance turn."""
    player = state.current_player
    caravan = state.caravans[player]

    # Check for overflow
    overflow = has_overflow(caravan, CARAVAN_LIMIT)

    return lax.cond(
        overflow,
        lambda s: s.replace(phase=jnp.int32(Phase.DISCARD_OVERFLOW)),
        _advance_turn,
        state
    )


# =============================================================================
# PLACE SPICE PHASE (for Acquire)
# =============================================================================

def transition_place_spice(state: State, action: jnp.ndarray) -> State:
    """Process PLACE_SPICE phase - place one spice on market card."""
    player = state.current_player
    spice_idx = action[4]
    target_pos = state.acquire_target_position
    spices_placed = state.spices_placed_count

    # Remove spice from caravan
    caravan = state.caravans[player]
    new_caravan = discard_spice(caravan, spice_idx)
    new_caravans = state.caravans.at[player].set(new_caravan)

    # Add spice to market card at position (spices_placed)
    # Spices go on cards 0, 1, 2, ... up to target_pos - 1
    card_pos = spices_placed
    market_spices = state.market_spices
    new_market_spices = market_spices.at[card_pos, spice_idx].add(1)

    new_spices_placed = spices_placed + 1

    new_state = state.replace(
        caravans=new_caravans,
        market_spices=new_market_spices,
        spices_placed_count=new_spices_placed
    )

    # If all spices placed, finalize acquire
    return lax.cond(
        new_spices_placed >= target_pos,
        lambda s: _finalize_acquire(s, target_pos),
        lambda s: s,  # Stay in PLACE_SPICE phase
        new_state
    )


def _finalize_acquire(state: State, target_pos: jnp.ndarray) -> State:
    """Finalize acquire - collect spices, take card, refill market."""
    player = state.current_player
    key = state.key

    # Collect spices from target card
    spices_on_card = state.market_spices[target_pos]
    caravan = state.caravans[player]
    new_caravan = caravan + spices_on_card
    new_caravans = state.caravans.at[player].set(new_caravan)

    # Take card into hand
    card = state.market_cards[target_pos]
    hand = state.hands[player]
    hand_size = state.hand_sizes[player]
    new_hand = hand.at[hand_size].set(card)
    new_hand_size = hand_size + 1
    new_hands = state.hands.at[player].set(new_hand)
    new_hand_sizes = state.hand_sizes.at[player].set(new_hand_size)

    # Shift market cards left (remove acquired card)
    market_cards = state.market_cards
    market_spices = state.market_spices
    market_size = state.market_size

    new_market_cards, new_market_spices, new_market_size, new_deck, new_deck_size, new_key = \
        _refill_market(market_cards, market_spices, market_size, target_pos,
                       state.trader_deck, state.trader_deck_size, key)

    new_state = state.replace(
        caravans=new_caravans,
        hands=new_hands,
        hand_sizes=new_hand_sizes,
        market_cards=new_market_cards,
        market_spices=new_market_spices,
        market_size=new_market_size,
        trader_deck=new_deck,
        trader_deck_size=new_deck_size,
        key=new_key
    )

    # Check for overflow then advance turn
    overflow = has_overflow(new_caravan, CARAVAN_LIMIT)

    return lax.cond(
        overflow,
        lambda s: s.replace(phase=jnp.int32(Phase.DISCARD_OVERFLOW)),
        _advance_turn,
        new_state
    )


def _refill_market(market_cards: jnp.ndarray, market_spices: jnp.ndarray,
                   market_size: jnp.ndarray, removed_pos: jnp.ndarray,
                   deck: jnp.ndarray, deck_size: jnp.ndarray,
                   key: jnp.ndarray):
    """Shift market left and refill from deck.

    Returns: (new_market_cards, new_market_spices, new_market_size,
              new_deck, new_deck_size, new_key)
    """
    # Shift cards left from removed position
    def shift_card(i, state):
        cards, spices = state
        should_shift = i >= removed_pos
        src_idx = i + 1
        src_valid = src_idx < NUM_MARKET_SLOTS

        # Get source values (or zeros if invalid)
        src_card = lax.cond(src_valid, lambda: cards[src_idx],
                            lambda: jnp.zeros(10, dtype=jnp.int32))
        src_spice = lax.cond(src_valid, lambda: spices[src_idx],
                             lambda: jnp.zeros(4, dtype=jnp.int32))

        # Update if should shift
        new_cards = lax.cond(should_shift, lambda: cards.at[i].set(src_card),
                             lambda: cards)
        new_spices = lax.cond(should_shift, lambda: spices.at[i].set(src_spice),
                              lambda: spices)

        return (new_cards, new_spices)

    shifted_cards, shifted_spices = lax.fori_loop(
        0, NUM_MARKET_SLOTS - 1, shift_card, (market_cards, market_spices)
    )

    # Clear the last position
    shifted_cards = shifted_cards.at[NUM_MARKET_SLOTS - 1].set(
        jnp.zeros(10, dtype=jnp.int32)
    )
    shifted_spices = shifted_spices.at[NUM_MARKET_SLOTS - 1].set(
        jnp.zeros(4, dtype=jnp.int32)
    )

    # Draw from deck if available
    can_draw = deck_size > 0
    new_card_pos = market_size - 1  # Position where new card goes

    new_cards = lax.cond(
        can_draw,
        lambda: shifted_cards.at[new_card_pos].set(deck[deck_size - 1]),
        lambda: shifted_cards
    )

    new_deck_size = lax.cond(can_draw, lambda: deck_size - 1, lambda: deck_size)
    new_market_size = lax.cond(can_draw, lambda: market_size, lambda: market_size - 1)

    return new_cards, shifted_spices, new_market_size, deck, new_deck_size, key


# =============================================================================
# SCORE ACTION
# =============================================================================

def _finalize_score(state: State, scoring_idx: jnp.ndarray) -> State:
    """Claim a scoring card."""
    player = state.current_player
    key = state.key

    # Get scoring card
    scoring_card = state.scoring_row[scoring_idx]
    requirements = get_scoring_requirements(scoring_card)

    # Remove spices from caravan
    caravan = state.caravans[player]
    new_caravan = caravan - requirements
    new_caravans = state.caravans.at[player].set(new_caravan)

    # Add scoring card to player
    scored = state.scored_cards[player]
    scored_count = state.scored_counts[player]
    new_scored = scored.at[scored_count].set(scoring_card)
    new_scored_count = scored_count + 1
    new_scored_cards = state.scored_cards.at[player].set(new_scored)
    new_scored_counts = state.scored_counts.at[player].set(new_scored_count)

    # Award coins based on position
    gold = state.gold_coins[player]
    silver = state.silver_coins[player]
    gold_remaining = state.gold_remaining
    silver_remaining = state.silver_remaining

    # Position 0: gold if available
    award_gold = (scoring_idx == 0) & (gold_remaining > 0)
    new_gold = lax.cond(award_gold, lambda: gold + 1, lambda: gold)
    new_gold_remaining = lax.cond(award_gold, lambda: gold_remaining - 1,
                                   lambda: gold_remaining)

    # Position 1: silver if gold still available (silver hasn't moved)
    award_silver = (scoring_idx == 1) & (gold_remaining > 0) & (silver_remaining > 0)
    new_silver = lax.cond(award_silver, lambda: silver + 1, lambda: silver)
    new_silver_remaining = lax.cond(award_silver, lambda: silver_remaining - 1,
                                     lambda: silver_remaining)

    # Special: if last gold taken, silver moves to position 0
    # This is handled implicitly - position 0 awards gold first

    new_gold_coins = state.gold_coins.at[player].set(new_gold)
    new_silver_coins = state.silver_coins.at[player].set(new_silver)

    # Shift scoring row and refill
    scoring_row = state.scoring_row
    scoring_row_size = state.scoring_row_size
    scoring_deck = state.scoring_deck
    scoring_deck_size = state.scoring_deck_size

    new_scoring_row, new_scoring_row_size, new_scoring_deck, new_scoring_deck_size = \
        _refill_scoring_row(scoring_row, scoring_row_size, scoring_idx,
                            scoring_deck, scoring_deck_size)

    new_state = state.replace(
        caravans=new_caravans,
        scored_cards=new_scored_cards,
        scored_counts=new_scored_counts,
        gold_coins=new_gold_coins,
        silver_coins=new_silver_coins,
        gold_remaining=new_gold_remaining,
        silver_remaining=new_silver_remaining,
        scoring_row=new_scoring_row,
        scoring_row_size=new_scoring_row_size,
        scoring_deck=new_scoring_deck,
        scoring_deck_size=new_scoring_deck_size
    )

    # Check game end trigger
    threshold = SCORING_CARDS_TO_WIN[state.num_players]
    triggered = new_scored_count >= threshold

    new_state = lax.cond(
        triggered & ~state.game_triggered,
        lambda s: s.replace(game_triggered=jnp.bool_(True),
                            trigger_player=player),
        lambda s: s,
        new_state
    )

    return _advance_turn(new_state)


def _refill_scoring_row(scoring_row: jnp.ndarray, scoring_row_size: jnp.ndarray,
                        removed_pos: jnp.ndarray, deck: jnp.ndarray,
                        deck_size: jnp.ndarray):
    """Shift scoring row left and refill from deck.

    Returns: (new_scoring_row, new_scoring_row_size, new_deck, new_deck_size)
    """
    # Shift cards left
    def shift_card(i, row):
        should_shift = i >= removed_pos
        src_idx = i + 1
        src_valid = src_idx < NUM_SCORING_SLOTS

        src_card = lax.cond(src_valid, lambda: row[src_idx],
                            lambda: jnp.zeros(5, dtype=jnp.int32))

        return lax.cond(should_shift, lambda: row.at[i].set(src_card), lambda: row)

    shifted_row = lax.fori_loop(0, NUM_SCORING_SLOTS - 1, shift_card, scoring_row)

    # Clear last position
    shifted_row = shifted_row.at[NUM_SCORING_SLOTS - 1].set(
        jnp.zeros(5, dtype=jnp.int32)
    )

    # Draw from deck if available
    can_draw = deck_size > 0
    new_card_pos = scoring_row_size - 1

    new_row = lax.cond(
        can_draw,
        lambda: shifted_row.at[new_card_pos].set(deck[deck_size - 1]),
        lambda: shifted_row
    )

    new_deck_size = lax.cond(can_draw, lambda: deck_size - 1, lambda: deck_size)
    new_row_size = lax.cond(can_draw, lambda: scoring_row_size,
                            lambda: scoring_row_size - 1)

    return new_row, new_row_size, deck, new_deck_size


# =============================================================================
# DISCARD OVERFLOW PHASE
# =============================================================================

def transition_discard_overflow(state: State, action: jnp.ndarray) -> State:
    """Process DISCARD_OVERFLOW phase - discard one spice."""
    player = state.current_player
    spice_idx = action[4]

    caravan = state.caravans[player]
    new_caravan = discard_spice(caravan, spice_idx)
    new_caravans = state.caravans.at[player].set(new_caravan)

    new_state = state.replace(caravans=new_caravans)

    # Check if still overflowing
    still_overflow = has_overflow(new_caravan, CARAVAN_LIMIT)

    return lax.cond(
        still_overflow,
        lambda s: s,  # Stay in DISCARD_OVERFLOW
        _advance_turn,
        new_state
    )


# =============================================================================
# TURN ADVANCEMENT
# =============================================================================

def _advance_turn(state: State) -> State:
    """Advance to next player's turn, check game end."""
    current = state.current_player
    num_players = state.num_players

    # Next player (wrapping)
    next_player = (current + 1) % num_players

    new_state = state.replace(
        current_player=next_player,
        phase=jnp.int32(Phase.CHOOSE_ACTION),
        selected_card_idx=jnp.int32(0),
        selected_card=jnp.zeros(10, dtype=jnp.int32),
        remaining_upgrades=jnp.int32(0),
        acquire_target_position=jnp.int32(0),
        spices_placed_count=jnp.int32(0)
    )

    return new_state


# =============================================================================
# UTILITIES
# =============================================================================

def _remove_card_from_array(arr: jnp.ndarray, idx: jnp.ndarray,
                             size: jnp.ndarray) -> jnp.ndarray:
    """Remove card at idx by shifting remaining cards left.

    Args:
        arr: Card array, shape (MAX_PLAYER_CARDS, 10)
        idx: Index to remove
        size: Current valid size

    Returns:
        Updated array with card removed and shifted
    """
    def shift(i, a):
        should_shift = i >= idx
        src_idx = i + 1
        src_valid = src_idx < size

        src_card = lax.cond(src_valid, lambda: a[src_idx],
                            lambda: jnp.zeros(10, dtype=jnp.int32))

        return lax.cond(should_shift, lambda: a.at[i].set(src_card), lambda: a)

    shifted = lax.fori_loop(0, MAX_PLAYER_CARDS - 1, shift, arr)

    # Clear the last position that was valid
    last_idx = size - 1
    return shifted.at[last_idx].set(jnp.zeros(10, dtype=jnp.int32))
