"""Textual App: layout, key bindings, action wizard, and AI worker."""

from __future__ import annotations

from enum import Enum, auto
from pathlib import Path

from textual.app import App, ComposeResult
from textual.widgets import Static, RichLog, Header
from textual.containers import Container
from textual.worker import Worker

from century_env.types import Phase
from century_env.constants import (
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)
from century_env.tui.game import GameController
from century_env.tui.widgets import (
    ItemClicked,
    MarketPanel,
    ScoringPanel,
    PlayersPanel,
    HandPanel,
    ActionPanel,
)

CSS_PATH = Path(__file__).parent / "styles.tcss"


class WizardState(Enum):
    IDLE = auto()
    PICK_ACTION_TYPE = auto()
    PICK_CARD = auto()
    PICK_MARKET = auto()
    PICK_SCORING = auto()
    PICK_SPICE = auto()
    PICK_CONTINUE = auto()
    AI_THINKING = auto()
    GAME_OVER = auto()


class SpiceRoadApp(App):
    """Interactive TUI for Century: Spice Road."""

    TITLE = "Century: Spice Road"
    CSS_PATH = CSS_PATH

    BINDINGS = [
        ("q", "quit_game", "Quit"),
        ("n", "new_game", "New Game"),
    ]

    def __init__(
        self,
        num_players: int = 4,
        seed: int = 42,
        ai_delay: float = 0.3,
    ) -> None:
        super().__init__()
        self.gc = GameController(num_players=num_players, seed=seed)
        self.ai_delay = ai_delay
        self.wizard = WizardState.IDLE
        # Context for multi-step wizard
        self._spice_callback: str = ""  # "execute_conv" | "place" | "discard"
        self._exchange_first_done: bool = False

    # ------------------------------------------------------------------
    # Compose
    # ------------------------------------------------------------------

    def compose(self) -> ComposeResult:
        yield Static("Century: Spice Road", id="status-bar")
        yield MarketPanel(id="market-panel")
        yield ScoringPanel(id="scoring-panel")
        yield PlayersPanel(id="players-panel")
        yield HandPanel(id="hand-panel")
        yield ActionPanel(id="action-panel")
        yield RichLog(highlight=True, markup=True, id="game-log")

    def on_mount(self) -> None:
        self.gc.new_game()
        self._refresh_all()
        self._enter_human_phase()

    # ------------------------------------------------------------------
    # Display refresh
    # ------------------------------------------------------------------

    def _refresh_all(self) -> None:
        data = self.gc.get_display_data()
        self.query_one("#market-panel", MarketPanel).update_data(
            data["market"], data["gold_remaining"], data["silver_remaining"]
        )
        self.query_one("#scoring-panel", ScoringPanel).update_data(
            data["scoring"], data["gold_remaining"], data["silver_remaining"]
        )
        self.query_one("#players-panel", PlayersPanel).update_data(
            data["players"], data["current_player"]
        )
        self.query_one("#hand-panel", HandPanel).update_data(
            data["hand"], data["played"], data["human_caravan"]
        )
        phase = data["phase"]
        player = data["current_player"]
        triggered = " [TRIGGERED]" if data["game_triggered"] else ""
        bar = f"Century: Spice Road  |  Phase: {phase.name}  |  Player: {player}{triggered}"
        self.query_one("#status-bar", Static).update(bar)

    def _log(self, msg: str) -> None:
        self.query_one("#game-log", RichLog).write(msg)

    def _flush_game_log(self) -> None:
        """Write any pending messages from game controller to the RichLog."""
        log_widget = self.query_one("#game-log", RichLog)
        for msg in self.gc.log:
            log_widget.write(msg)
        self.gc.log.clear()

    # ------------------------------------------------------------------
    # Phase entry
    # ------------------------------------------------------------------

    def _enter_human_phase(self) -> None:
        """Determine what UI to show based on the current env phase."""
        if self.gc.game_over:
            self._show_game_over()
            return

        if not self.gc.is_human_turn:
            self._run_ai()
            return

        phase = self.gc.phase
        action_panel = self.query_one("#action-panel", ActionPanel)

        if phase == Phase.CHOOSE_ACTION:
            self._exchange_first_done = False
            mask = self.gc.get_action_type_mask()
            action_panel.show_choose_action(mask)
            self.wizard = WizardState.PICK_ACTION_TYPE

        elif phase == Phase.EXECUTE_CARD:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            ct = ast["card_type"]
            if ct == CARD_TYPE_SPICE:
                # Auto-submit DONE
                self.gc.step([0, 0, 0, 0, 0, 1])
                self._flush_game_log()
                self._refresh_all()
                self._enter_human_phase()
            elif ct == CARD_TYPE_CONVERSION:
                legal = ast["upgradeable_spices"]
                if not legal:
                    # No upgradeable spices, auto-DONE
                    self.gc.step([0, 0, 0, 0, 0, 1])
                    self._flush_game_log()
                    self._refresh_all()
                    self._enter_human_phase()
                else:
                    self._spice_callback = "execute_conv"
                    remaining = ast["remaining_upgrades"]
                    total = ast["total_upgrades"]
                    action_panel.show_pick_spice(
                        legal,
                        f"Upgrade spice ({remaining} left):",
                        show_done=remaining < total,
                    )
                    self.wizard = WizardState.PICK_SPICE
            elif ct == CARD_TYPE_EXCHANGE:
                if not self._exchange_first_done:
                    self._exchange_first_done = True
                    self.gc.step([0, 0, 0, 0, 0, 0])
                    self._flush_game_log()
                    self._refresh_all()
                    self._enter_human_phase()
                else:
                    can_again = ast.get("can_again", False)
                    action_panel.show_pick_continue(can_again)
                    self.wizard = WizardState.PICK_CONTINUE
                    self._spice_callback = "execute_exch"

        elif phase == Phase.PLACE_SPICE:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            legal = ast["placeable_spices"]
            target = ast["target_pos"]
            placed = ast["placed"]
            self._spice_callback = "place"
            action_panel.show_pick_spice(
                legal, f"Place spice on market ({placed}/{target}):"
            )
            self.wizard = WizardState.PICK_SPICE

        elif phase == Phase.DISCARD_OVERFLOW:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            legal = ast["discardable_spices"]
            overflow = ast["overflow"]
            self._spice_callback = "discard"
            action_panel.show_pick_spice(
                legal, f"Discard spice (overflow by {overflow}):"
            )
            self.wizard = WizardState.PICK_SPICE

        else:
            # Unexpected phase for human – shouldn't happen
            self._log(f"[red]Unexpected phase: {phase.name}[/red]")

    def _show_game_over(self) -> None:
        self.wizard = WizardState.GAME_OVER
        scores = self.gc.get_final_scores()
        self._flush_game_log()
        self._log("[bold red]Game Over![/bold red]")
        for i, score in enumerate(scores):
            label = "You" if i == 0 else f"Player {i}"
            self._log(f"  {label}: {score} pts")
        self.query_one("#action-panel", ActionPanel).show_game_over(scores)

    # ------------------------------------------------------------------
    # AI worker
    # ------------------------------------------------------------------

    def _run_ai(self) -> None:
        self.wizard = WizardState.AI_THINKING
        self.query_one("#action-panel", ActionPanel).show_waiting()
        self.run_worker(self._ai_loop, exclusive=True, thread=True)

    def _ai_loop(self) -> None:
        import time
        while not self.gc.game_over and not self.gc.is_human_turn:
            self.gc.step_ai()
            self.call_from_thread(self._flush_game_log)
            self.call_from_thread(self._refresh_all)
            time.sleep(self.ai_delay)
        self.call_from_thread(self._enter_human_phase)

    # ------------------------------------------------------------------
    # Key handling
    # ------------------------------------------------------------------

    def on_key(self, event) -> None:
        key = event.key

        # Global keys
        if key == "q":
            self.exit()
            return
        if key == "n":
            self.gc.new_game()
            self.query_one("#game-log", RichLog).clear()
            self._flush_game_log()
            self._refresh_all()
            self._enter_human_phase()
            return

        if self.wizard == WizardState.PICK_ACTION_TYPE:
            self._handle_action_type(key)
        elif self.wizard == WizardState.PICK_CARD:
            self._handle_pick_index(key, "card")
        elif self.wizard == WizardState.PICK_MARKET:
            self._handle_pick_index(key, "market")
        elif self.wizard == WizardState.PICK_SCORING:
            self._handle_pick_index(key, "scoring")
        elif self.wizard == WizardState.PICK_SPICE:
            self._handle_pick_spice(key)
        elif self.wizard == WizardState.PICK_CONTINUE:
            self._handle_pick_continue(key)

    # ------------------------------------------------------------------
    # Click handling
    # ------------------------------------------------------------------

    def on_item_clicked(self, message: ItemClicked) -> None:
        pt, idx = message.panel_type, message.index
        if self.wizard == WizardState.PICK_ACTION_TYPE and pt == "action":
            self._do_action_type(idx)
        elif self.wizard == WizardState.PICK_CARD and pt == "hand":
            self._do_pick_index(idx, "card")
        elif self.wizard == WizardState.PICK_MARKET and pt == "market":
            self._do_pick_index(idx, "market")
        elif self.wizard == WizardState.PICK_SCORING and pt == "scoring":
            self._do_pick_index(idx, "scoring")
        elif self.wizard == WizardState.PICK_SPICE and pt == "spice":
            self._do_pick_spice(idx)
        elif self.wizard == WizardState.PICK_SPICE and pt == "done":
            self._do_spice_done()
        elif self.wizard == WizardState.PICK_CONTINUE and pt == "continue":
            if idx == 0:
                self._do_continue_again()
            else:
                self._do_continue_done()

    # ------------------------------------------------------------------
    # Shared action logic (used by both key and click handlers)
    # ------------------------------------------------------------------

    def _do_action_type(self, action_type: int) -> None:
        mask = self.gc.get_action_type_mask()
        if not mask[action_type]:
            self._log("[red]That action is not legal.[/red]")
            return

        data = self.gc.get_display_data()
        action_panel = self.query_one("#action-panel", ActionPanel)

        if action_type == 0:  # Play
            action_panel.show_pick_card(data["hand"])
            self.wizard = WizardState.PICK_CARD
        elif action_type == 1:  # Acquire
            legal = self.gc.get_legal_market_positions()
            action_panel.show_pick_market(legal, data["market"])
            self.wizard = WizardState.PICK_MARKET
        elif action_type == 2:  # Rest
            self.gc.step([2, 0, 0, 0, 0, 0])
            self._flush_game_log()
            self._refresh_all()
            self._enter_human_phase()
        elif action_type == 3:  # Score
            legal = self.gc.get_legal_scoring_indices()
            action_panel.show_pick_scoring(legal, data["scoring"])
            self.wizard = WizardState.PICK_SCORING

    def _do_pick_index(self, idx: int, context: str) -> None:
        if context == "card":
            legal = self.gc.get_legal_hand_indices()
            if idx not in legal:
                self._log("[red]Invalid card index.[/red]")
                return
            self.gc.step([0, idx, 0, 0, 0, 0])
        elif context == "market":
            legal = self.gc.get_legal_market_positions()
            if idx not in legal:
                self._log("[red]Invalid market position.[/red]")
                return
            self.gc.step([1, 0, idx, 0, 0, 0])
        elif context == "scoring":
            legal = self.gc.get_legal_scoring_indices()
            if idx not in legal:
                self._log("[red]Invalid scoring card.[/red]")
                return
            self.gc.step([3, 0, 0, idx, 0, 0])

        self._flush_game_log()
        self._refresh_all()
        self._enter_human_phase()

    def _do_pick_spice(self, spice: int) -> None:
        self.gc.step([0, 0, 0, 0, spice, 0])
        self._flush_game_log()
        self._refresh_all()
        self._enter_human_phase()

    def _do_spice_done(self) -> None:
        self.gc.step([0, 0, 0, 0, 0, 1])
        self._flush_game_log()
        self._refresh_all()
        self._enter_human_phase()

    def _do_continue_again(self) -> None:
        if self._spice_callback == "execute_exch":
            self.gc.step([0, 0, 0, 0, 0, 0])
            self._flush_game_log()
            self._refresh_all()
            self._enter_human_phase()

    def _do_continue_done(self) -> None:
        self.gc.step([0, 0, 0, 0, 0, 1])
        self._flush_game_log()
        self._refresh_all()
        self._enter_human_phase()

    # ------------------------------------------------------------------
    # Key handlers (delegate to shared logic)
    # ------------------------------------------------------------------

    def _handle_action_type(self, key: str) -> None:
        mapping = {"p": 0, "a": 1, "r": 2, "s": 3}
        if key not in mapping:
            return
        self._do_action_type(mapping[key])

    def _handle_pick_index(self, key: str, context: str) -> None:
        if key == "escape":
            self._enter_human_phase()
            return
        idx = _key_to_index(key)
        if idx is None:
            return
        self._do_pick_index(idx, context)

    def _handle_pick_spice(self, key: str) -> None:
        if key == "escape":
            if self._spice_callback in ("execute_conv",):
                self._do_spice_done()
                return
            return
        if key == "d" and self._spice_callback == "execute_conv":
            self._do_spice_done()
            return
        mapping = {"y": 0, "r": 1, "g": 2, "b": 3}
        if key not in mapping:
            return
        self._do_pick_spice(mapping[key])

    def _handle_pick_continue(self, key: str) -> None:
        if key == "d":
            self._do_continue_done()
        elif key == "a":
            self._do_continue_again()
        elif key == "escape":
            self._do_continue_done()

    # ------------------------------------------------------------------
    # Actions (bound)
    # ------------------------------------------------------------------

    def action_quit_game(self) -> None:
        self.exit()

    def action_new_game(self) -> None:
        self.gc.new_game()
        self.query_one("#game-log", RichLog).clear()
        self._flush_game_log()
        self._refresh_all()
        self._enter_human_phase()


def _key_to_index(key: str) -> int | None:
    """Convert key press to numeric index. 0-9 for digits, a-o for 10-24."""
    if key.isdigit():
        return int(key)
    if len(key) == 1 and "a" <= key <= "o":
        return ord(key) - ord("a") + 10
    return None
