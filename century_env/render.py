"""
Century: Spice Road - Text-based Rendering

For debugging only - these functions use Python strings/loops.
WARNING: Do NOT call from JIT-compiled code - will cause tracing errors.
"""

import jax.numpy as jnp

from century_env.types import State, Observation, Phase
from century_env.constants import (
    SPICE_NAMES,
    SPICE_YELLOW,
    SPICE_RED,
    SPICE_GREEN,
    SPICE_BROWN,
    MAX_PLAYERS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
)
from century_env.cards import (
    get_card_type,
    get_card_upgrades,
    get_card_input,
    get_card_output,
    get_scoring_points,
    get_scoring_requirements,
    get_trader_card_description,
    get_scoring_card_description,
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)


# Spice symbols for compact display
SPICE_CHARS = ['Y', 'R', 'G', 'B']
SPICE_COLORS = ['Yellow', 'Red', 'Green', 'Brown']


def render_state(state: State) -> str:
    """Render full game state as formatted string.

    Args:
        state: Current game state

    Returns:
        Multi-line string representation
    """
    lines = []
    num_players = int(state.num_players)
    current = int(state.current_player)
    phase = Phase(int(state.phase))

    lines.append("=" * 60)
    lines.append(f"CENTURY: SPICE ROAD - Turn: Player {current}")
    lines.append(f"Phase: {phase.name}")
    if state.game_triggered:
        lines.append(f"*** GAME TRIGGERED by Player {int(state.trigger_player)} ***")
    lines.append("=" * 60)

    # Market row
    lines.append("\n--- MARKET ROW ---")
    market_size = int(state.market_size)
    for i in range(market_size):
        card = state.market_cards[i]
        spices = state.market_spices[i]
        card_str = _render_trader_card_inline(card)
        spice_str = render_caravan(spices)
        lines.append(f"  [{i}] {card_str}")
        if jnp.sum(spices) > 0:
            lines.append(f"      Spices on card: {spice_str}")

    # Scoring row
    lines.append("\n--- SCORING ROW ---")
    scoring_size = int(state.scoring_row_size)
    gold_marker = f" [GOLD: {int(state.gold_remaining)}]" if state.gold_remaining > 0 else ""
    silver_marker = f" [SILVER: {int(state.silver_remaining)}]" if state.silver_remaining > 0 else ""
    for i in range(scoring_size):
        card = state.scoring_row[i]
        card_str = _render_scoring_card_inline(card)
        marker = ""
        if i == 0:
            marker = gold_marker
        elif i == 1 and state.gold_remaining > 0:
            marker = silver_marker
        lines.append(f"  [{i}] {card_str}{marker}")

    # Players
    lines.append("\n--- PLAYERS ---")
    for p in range(num_players):
        marker = " <<< CURRENT" if p == current else ""
        hand_size = int(state.hand_sizes[p])
        played_size = int(state.played_sizes[p])
        caravan = state.caravans[p]
        gold = int(state.gold_coins[p])
        silver = int(state.silver_coins[p])
        scored = int(state.scored_counts[p])

        lines.append(f"\nPlayer {p}{marker}")
        lines.append(f"  Caravan: {render_caravan(caravan)} (total: {int(jnp.sum(caravan))})")
        lines.append(f"  Hand: {hand_size} cards, Played: {played_size} cards")
        lines.append(f"  Scored: {scored} cards | Gold: {gold}, Silver: {silver}")

        # Show hand cards for current player
        if p == current:
            lines.append("  Hand cards:")
            for i in range(hand_size):
                card = state.hands[p, i]
                card_str = _render_trader_card_inline(card)
                lines.append(f"    [{i}] {card_str}")

            if played_size > 0:
                lines.append("  Played cards:")
                for i in range(played_size):
                    card = state.played_piles[p, i]
                    card_str = _render_trader_card_inline(card)
                    lines.append(f"    {card_str}")

    # Supply info
    lines.append("\n--- SUPPLY ---")
    lines.append(f"  Trader deck: {int(state.trader_deck_size)} cards")
    lines.append(f"  Scoring deck: {int(state.scoring_deck_size)} cards")
    lines.append(f"  Gold coins: {int(state.gold_remaining)}")
    lines.append(f"  Silver coins: {int(state.silver_remaining)}")

    # Action state (if mid-action)
    if phase == Phase.EXECUTE_CARD:
        lines.append("\n--- ACTION STATE ---")
        lines.append(f"  Selected card: {_render_trader_card_inline(state.selected_card)}")
        lines.append(f"  Remaining upgrades: {int(state.remaining_upgrades)}")
    elif phase == Phase.PLACE_SPICE:
        lines.append("\n--- ACTION STATE ---")
        lines.append(f"  Acquiring from position: {int(state.acquire_target_position)}")
        lines.append(f"  Spices placed: {int(state.spices_placed_count)}")

    lines.append("=" * 60)

    return "\n".join(lines)


def render_observation(obs: Observation) -> str:
    """Render observation as formatted string (ego-centric view).

    Args:
        obs: Player observation

    Returns:
        Multi-line string representation
    """
    lines = []
    phase = Phase(int(obs.current_phase))

    lines.append("=" * 50)
    lines.append("YOUR VIEW (Ego-centric)")
    lines.append(f"Phase: {phase.name}")
    lines.append("=" * 50)

    # My state
    lines.append("\n--- YOUR STATE ---")
    lines.append(f"Caravan: {render_caravan(obs.my_caravan)}")
    lines.append(f"Hand: {int(obs.my_hand_size)} cards")
    lines.append(f"Played: {int(obs.my_played_size)} cards")
    lines.append(f"Scored: {int(obs.my_scored_count)} | Gold: {int(obs.my_gold)}, Silver: {int(obs.my_silver)}")

    # Hand cards
    hand_size = int(obs.my_hand_size)
    if hand_size > 0:
        lines.append("Hand cards:")
        for i in range(hand_size):
            card = obs.my_hand[i]
            card_str = _render_trader_card_inline(card)
            lines.append(f"  [{i}] {card_str}")

    # Opponents
    lines.append("\n--- OPPONENTS ---")
    for i in range(4):
        if obs.opp_active[i]:
            lines.append(f"Opponent {i}: Hand={int(obs.opp_hand_sizes[i])}, "
                        f"Caravan={render_caravan(obs.opp_caravans[i])}, "
                        f"Scored={int(obs.opp_scored_counts[i])}")

    # Market
    lines.append("\n--- MARKET ---")
    market_size = int(obs.market_size)
    for i in range(market_size):
        card = obs.market_cards[i]
        spices = obs.market_spices[i]
        card_str = _render_trader_card_inline(card)
        spice_str = render_caravan(spices) if jnp.sum(spices) > 0 else ""
        extra = f" + {spice_str}" if spice_str else ""
        lines.append(f"  [{i}] {card_str}{extra}")

    # Scoring row
    lines.append("\n--- SCORING ---")
    scoring_size = int(obs.scoring_row_size)
    for i in range(scoring_size):
        card = obs.scoring_row[i]
        card_str = _render_scoring_card_inline(card)
        lines.append(f"  [{i}] {card_str}")

    lines.append("=" * 50)

    return "\n".join(lines)


def render_action(action: jnp.ndarray, phase: Phase) -> str:
    """Render action as human-readable string.

    Args:
        action: Multi-discrete action array, shape (6,)
        phase: Current game phase

    Returns:
        String description of action
    """
    action_type = int(action[0])
    card_idx = int(action[1])
    market_pos = int(action[2])
    scoring_idx = int(action[3])
    spice_type = int(action[4])
    continue_flag = int(action[5])

    if phase == Phase.CHOOSE_ACTION:
        action_names = ["PLAY", "ACQUIRE", "REST", "SCORE"]
        name = action_names[action_type]
        if action_type == 0:  # Play
            return f"{name} card at index {card_idx}"
        elif action_type == 1:  # Acquire
            return f"{name} from market position {market_pos}"
        elif action_type == 2:  # Rest
            return name
        elif action_type == 3:  # Score
            return f"{name} card at index {scoring_idx}"

    elif phase == Phase.EXECUTE_CARD:
        spice = SPICE_CHARS[spice_type]
        cont = "AGAIN" if continue_flag == 0 else "DONE"
        return f"Upgrade {spice}, then {cont}"

    elif phase == Phase.PLACE_SPICE:
        spice = SPICE_CHARS[spice_type]
        return f"Place {spice}"

    elif phase == Phase.DISCARD_OVERFLOW:
        spice = SPICE_CHARS[spice_type]
        return f"Discard {spice}"

    return f"Action: {action.tolist()}"


def render_caravan(caravan: jnp.ndarray) -> str:
    """Render caravan as compact string like 'YYY RR G B'.

    Args:
        caravan: Spice counts, shape (4,)

    Returns:
        String with spice symbols
    """
    parts = []
    for i, count in enumerate(caravan):
        count = int(count)
        if count > 0:
            parts.append(SPICE_CHARS[i] * count)

    return " ".join(parts) if parts else "(empty)"


def render_card(card: jnp.ndarray, is_scoring: bool = False) -> str:
    """Render a card as multi-line description.

    Args:
        card: Card data array
        is_scoring: True if scoring card, False if trader card

    Returns:
        String description
    """
    if is_scoring:
        return _render_scoring_card_full(card)
    else:
        return _render_trader_card_full(card)


def _render_trader_card_inline(card: jnp.ndarray) -> str:
    """Render trader card as single-line string."""
    card_type = int(card[0])

    if card_type == CARD_TYPE_SPICE:
        output = card[6:10]
        output_str = _spices_to_str(output)
        return f"Obtain {output_str}"

    elif card_type == CARD_TYPE_CONVERSION:
        upgrades = int(card[1])
        return f"Upgrade ×{upgrades}"

    elif card_type == CARD_TYPE_EXCHANGE:
        input_spices = card[2:6]
        output_spices = card[6:10]
        input_str = _spices_to_str(input_spices)
        output_str = _spices_to_str(output_spices)
        return f"{input_str} → {output_str}"

    return "Unknown card"


def _render_trader_card_full(card: jnp.ndarray) -> str:
    """Render trader card as multi-line description."""
    lines = []
    card_type = int(card[0])
    type_names = ["Spice", "Conversion", "Exchange"]
    lines.append(f"Type: {type_names[card_type]}")

    if card_type == CARD_TYPE_SPICE:
        output = card[6:10]
        lines.append(f"Produces: {_spices_to_str(output)}")

    elif card_type == CARD_TYPE_CONVERSION:
        upgrades = int(card[1])
        lines.append(f"Upgrades: {upgrades}")

    elif card_type == CARD_TYPE_EXCHANGE:
        input_spices = card[2:6]
        output_spices = card[6:10]
        lines.append(f"Requires: {_spices_to_str(input_spices)}")
        lines.append(f"Produces: {_spices_to_str(output_spices)}")

    return "\n".join(lines)


def _render_scoring_card_inline(card: jnp.ndarray) -> str:
    """Render scoring card as single-line string."""
    points = int(card[0])
    req = card[1:5]
    req_str = _spices_to_str(req)
    return f"{req_str} = {points}pts"


def _render_scoring_card_full(card: jnp.ndarray) -> str:
    """Render scoring card as multi-line description."""
    points = int(card[0])
    req = card[1:5]
    lines = [
        f"Points: {points}",
        f"Requires: {_spices_to_str(req)}"
    ]
    return "\n".join(lines)


def _spices_to_str(spices: jnp.ndarray) -> str:
    """Convert spice array to compact string."""
    parts = []
    for i, count in enumerate(spices):
        count = int(count)
        if count > 0:
            parts.append(SPICE_CHARS[i] * count)

    return "".join(parts) if parts else "(none)"
