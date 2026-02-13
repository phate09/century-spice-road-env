"""Game controller: wraps the JAX environment for the TUI.

Keeps State as JAX internally; all public methods return plain Python.
Handles AI turns via the RandomAgent.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from century_env.env import CenturySpiceRoad
from century_env.types import Phase, ActionType
from century_env.constants import (
    CARAVAN_LIMIT,
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
    NUM_SPICE_TYPES,
    SPICE_BROWN,
)
from century_env.cards import (
    can_afford_scoring_card,
    can_afford_exchange,
    get_card_type,
    get_card_upgrades,
    get_card_input,
)
from century_env.mechanics import caravan_total, can_apply_conversion
from century_env.render import (
    SPICE_CHARS,
    _render_trader_card_inline,
    _render_scoring_card_inline,
    render_caravan as _render_caravan_jax,
    render_action as _render_action_jax,
)
from century_env.rewards import compute_final_scores
from century_env.agents.random_agent import RandomAgent


def _j2p(x: jnp.ndarray) -> Any:
    """JAX array to plain Python scalar or list."""
    a = np.asarray(x)
    return a.item() if a.ndim == 0 else a.tolist()


class GameController:
    """Wraps CenturySpiceRoad for interactive TUI play.

    Player 0 is always the human; players 1..N-1 use RandomAgent.
    """

    def __init__(self, num_players: int = 4, seed: int = 42) -> None:
        self.num_players = num_players
        self.env = CenturySpiceRoad(num_players=num_players)
        self.agent = RandomAgent()
        self._rng = jax.random.PRNGKey(seed)
        self.state = None
        self.timestep = None
        self.game_over = False
        self.log: list[str] = []
        self.history: list[str] = []
        self._checkpoint: tuple | None = None
        self._history_caravan_before: list[int] | None = None
        self._history_player: int = 0

    # ------------------------------------------------------------------
    # Checkpoint (undo support)
    # ------------------------------------------------------------------

    def save_checkpoint(self) -> None:
        """Snapshot current state/timestep so the player can undo."""
        self._checkpoint = (self.state, self.timestep, len(self.history))

    def clear_checkpoint(self) -> None:
        self._checkpoint = None

    @property
    def can_undo(self) -> bool:
        return self._checkpoint is not None

    def undo(self) -> bool:
        """Restore the checkpoint. Returns True on success."""
        if self._checkpoint is None:
            return False
        self.state, self.timestep, hist_len = self._checkpoint
        self.history = self.history[:hist_len]
        self._checkpoint = None
        self._history_caravan_before = None
        self.game_over = False
        self.log.append("Undid action.")
        return True

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def new_game(self, seed: int | None = None) -> None:
        if seed is not None:
            self._rng = jax.random.PRNGKey(seed)
        self._rng, reset_key = jax.random.split(self._rng)
        self.state, self.timestep = self.env.reset(reset_key)
        self.game_over = False
        self.log = ["New game started."]
        self.history = []
        self._history_caravan_before = None

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def step(self, action: list[int]) -> None:
        """Take one env step with the given 6-element action."""
        action_jax = jnp.array(action, dtype=jnp.int32)
        phase = Phase(int(self.state.phase))
        description = _render_action_jax(action_jax, phase)
        player = int(self.state.current_player)
        label = "You" if player == 0 else f"Player {player}"
        self.log.append(f"{label}: {description}")
        if phase == Phase.CHOOSE_ACTION:
            self.history.append(self._build_history_entry(action))
            self._history_caravan_before = np.asarray(
                self.state.caravans[player]
            ).tolist()
            self._history_player = player
        self.state, self.timestep = self.env.step(self.state, action_jax)
        self.game_over = bool(self.timestep.last())
        # Finalize history entry when action completes
        new_phase = Phase(int(self.state.phase))
        if self._history_caravan_before is not None and (
            new_phase == Phase.CHOOSE_ACTION or self.game_over
        ):
            self._finalize_history_entry()

    def step_ai(self) -> None:
        """Play one AI step for the current player."""
        masks = self.timestep.observation.action_mask
        self._rng, key = jax.random.split(self._rng)
        action = self.agent.select_action_from_mask(key, masks)
        self.step(_j2p(action))

    # ------------------------------------------------------------------
    # History helpers
    # ------------------------------------------------------------------

    def _build_history_entry(self, action: list[int]) -> str:
        """Build a rich history entry for a CHOOSE_ACTION step."""
        s = self.state
        player = int(s.current_player)
        action_type = action[0]
        card_idx = action[1]
        market_pos = action[2]
        scoring_idx = action[3]
        label = "You" if player == 0 else f"Player {player}"

        if action_type == 0:  # Play
            card = s.hands[player, card_idx]
            card_str = _render_trader_card_inline(card)
            return f"{label}: PLAY {card_str}"
        elif action_type == 1:  # Acquire
            card = s.market_cards[market_pos]
            card_str = _render_trader_card_inline(card)
            cost_part = f" (cost: {market_pos})" if market_pos > 0 else ""
            return f"{label}: ACQUIRE {card_str}{cost_part}"
        elif action_type == 2:  # Rest
            played_size = int(s.played_sizes[player])
            return f"{label}: REST ({played_size} cards)"
        elif action_type == 3:  # Score
            card = s.scoring_row[scoring_idx]
            card_str = _render_scoring_card_inline(card)
            return f"{label}: SCORE {card_str}"
        return f"{label}: ???"

    def _finalize_history_entry(self) -> None:
        """Append net caravan change to the last history entry."""
        after = np.asarray(self.state.caravans[self._history_player]).tolist()
        before = self._history_caravan_before
        self._history_caravan_before = None
        diff = [after[i] - before[i] for i in range(len(before))]
        if all(d == 0 for d in diff):
            return
        gains = "".join(SPICE_CHARS[i] * d for i, d in enumerate(diff) if d > 0)
        losses = "".join(
            SPICE_CHARS[i] * (-d) for i, d in enumerate(diff) if d < 0
        )
        parts = []
        if gains:
            parts.append(f"+{gains}")
        if losses:
            parts.append(f"-{losses}")
        if parts:
            self.history[-1] += f" ({' '.join(parts)})"

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    @property
    def current_player(self) -> int:
        return int(self.state.current_player)

    @property
    def phase(self) -> Phase:
        return Phase(int(self.state.phase))

    @property
    def is_human_turn(self) -> bool:
        return self.current_player == 0

    def get_action_type_mask(self) -> list[bool]:
        masks = self.timestep.observation.action_mask
        return _j2p(masks[0])

    def get_mask(self, head: int) -> list[bool]:
        masks = self.timestep.observation.action_mask
        return _j2p(masks[head])

    # ------------------------------------------------------------------
    # Manual legality for CHOOSE_ACTION sub-selections
    # ------------------------------------------------------------------

    def get_legal_hand_indices(self) -> list[int]:
        hs = int(self.state.hand_sizes[0])
        return list(range(hs))

    def get_legal_market_positions(self) -> list[int]:
        caravan = self.state.caravans[0]
        total = int(caravan_total(caravan))
        ms = int(self.state.market_size)
        return [i for i in range(ms) if total >= i]

    def get_legal_scoring_indices(self) -> list[int]:
        caravan = self.state.caravans[0]
        sr_size = int(self.state.scoring_row_size)
        result = []
        for i in range(sr_size):
            card = self.state.scoring_row[i]
            if bool(can_afford_scoring_card(caravan, card)):
                result.append(i)
        return result

    # ------------------------------------------------------------------
    # Display data  (all plain Python)
    # ------------------------------------------------------------------

    def get_display_data(self) -> dict[str, Any]:
        s = self.state
        num_p = int(s.num_players)

        # Market row
        market = []
        ms = int(s.market_size)
        for i in range(ms):
            card = s.market_cards[i]
            spices = s.market_spices[i]
            card_str = _render_trader_card_inline(card)
            sp_total = int(jnp.sum(spices))
            sp_str = _render_caravan_jax(spices) if sp_total > 0 else ""
            market.append({"idx": i, "card": card_str, "spices": sp_str})

        # Scoring row
        scoring = []
        sr_size = int(s.scoring_row_size)
        for i in range(sr_size):
            card = s.scoring_row[i]
            scoring.append({
                "idx": i,
                "card": _render_scoring_card_inline(card),
            })

        # Players
        players = []
        for p in range(num_p):
            caravan = s.caravans[p]
            players.append({
                "idx": p,
                "caravan": _j2p(caravan),
                "caravan_total": int(jnp.sum(caravan)),
                "hand_size": int(s.hand_sizes[p]),
                "played_size": int(s.played_sizes[p]),
                "scored_count": int(s.scored_counts[p]),
                "gold": int(s.gold_coins[p]),
                "silver": int(s.silver_coins[p]),
            })

        # Human hand
        hand = []
        hs = int(s.hand_sizes[0])
        for i in range(hs):
            card = s.hands[0, i]
            hand.append({"idx": i, "card": _render_trader_card_inline(card)})

        # Human played pile
        played = []
        ps = int(s.played_sizes[0])
        for i in range(ps):
            card = s.played_piles[0, i]
            played.append(_render_trader_card_inline(card))

        human_caravan = _j2p(s.caravans[0])

        # Phase-specific info
        phase = Phase(int(s.phase))
        action_state: dict[str, Any] = {}
        if phase == Phase.EXECUTE_CARD:
            card = s.selected_card
            ct = int(get_card_type(card))
            action_state["card_type"] = ct
            action_state["card_text"] = _render_trader_card_inline(card)
            action_state["remaining_upgrades"] = int(s.remaining_upgrades)
            action_state["total_upgrades"] = int(get_card_upgrades(s.selected_card))
            if ct == CARD_TYPE_CONVERSION:
                caravan = s.caravans[int(s.current_player)]
                upgradeable = []
                for si in range(NUM_SPICE_TYPES):
                    if si != SPICE_BROWN and int(caravan[si]) > 0:
                        upgradeable.append(si)
                action_state["upgradeable_spices"] = upgradeable
            elif ct == CARD_TYPE_EXCHANGE:
                caravan = s.caravans[int(s.current_player)]
                action_state["can_again"] = bool(can_afford_exchange(caravan, card))
        elif phase == Phase.PLACE_SPICE:
            caravan = s.caravans[int(s.current_player)]
            placeable = [i for i in range(NUM_SPICE_TYPES) if int(caravan[i]) > 0]
            action_state["placeable_spices"] = placeable
            action_state["target_pos"] = int(s.acquire_target_position)
            action_state["placed"] = int(s.spices_placed_count)
        elif phase == Phase.DISCARD_OVERFLOW:
            caravan = s.caravans[int(s.current_player)]
            discardable = [i for i in range(NUM_SPICE_TYPES) if int(caravan[i]) > 0]
            action_state["discardable_spices"] = discardable
            action_state["overflow"] = int(jnp.sum(caravan)) - CARAVAN_LIMIT

        # Gold/silver remaining
        gold_rem = int(s.gold_remaining)
        silver_rem = int(s.silver_remaining)

        return {
            "phase": phase,
            "current_player": int(s.current_player),
            "market": market,
            "scoring": scoring,
            "players": players,
            "hand": hand,
            "played": played,
            "human_caravan": human_caravan,
            "action_state": action_state,
            "gold_remaining": gold_rem,
            "silver_remaining": silver_rem,
            "game_over": self.game_over,
            "game_triggered": bool(s.game_triggered),
        }

    def get_deck_data(self) -> dict[str, Any]:
        """Return full contents of both decks (top-to-bottom)."""
        s = self.state
        td_size = int(s.trader_deck_size)
        sd_size = int(s.scoring_deck_size)

        trader_deck = []
        for i in range(td_size):
            card = s.trader_deck[i]
            trader_deck.append({"idx": i, "card": _render_trader_card_inline(card)})

        scoring_deck = []
        for i in range(sd_size):
            card = s.scoring_deck[i]
            scoring_deck.append({"idx": i, "card": _render_scoring_card_inline(card)})

        return {
            "trader_deck": trader_deck,
            "trader_deck_size": td_size,
            "scoring_deck": scoring_deck,
            "scoring_deck_size": sd_size,
        }

    def get_history(self) -> list[str]:
        return list(self.history)

    def get_final_scores(self) -> list[int]:
        scores = compute_final_scores(self.state)
        num_p = int(self.state.num_players)
        return [int(scores[i]) for i in range(num_p)]
