"""Main Pygame application — mirrors tui/app.py's wizard state machine."""

from __future__ import annotations

import threading
from enum import Enum, auto
from typing import Any

import pygame

from century_env.types import Phase
from century_env.constants import (
    CARD_TYPE_SPICE,
    CARD_TYPE_CONVERSION,
    CARD_TYPE_EXCHANGE,
)
from century_env.tui.game import GameController
from century_env.gui.constants import (
    WIDTH,
    HEIGHT,
    FPS,
    BG_COLOR,
    PANEL_BG,
    TEXT_COLOR,
    TEXT_DIM,
    HIGHLIGHT_BORDER,
    CURRENT_PLAYER_BG,
    SPICE_COLORS,
    SPICE_LETTERS,
    SPICE_NAMES_SHORT,
    TRADER_CARD_W,
    TRADER_CARD_H,
    SCORING_CARD_W,
    SCORING_CARD_H,
    HAND_CARD_W,
    HAND_CARD_H,
    BTN_W,
    BTN_H,
    GOLD_COLOR,
    SILVER_COLOR,
    HEADER_RECT,
    MARKET_RECT,
    SCORING_RECT,
    PLAYERS_RECT,
    HAND_RECT,
    CARAVAN_RECT,
    ACTION_RECT,
    LOG_RECT,
    PANEL_HDR_H,
    AI_STEP_EVENT,
    AI_DONE_EVENT,
)
from century_env.gui.renderer import (
    draw_trader_card,
    draw_scoring_card,
    draw_coin_badge,
    draw_caravan_cubes,
    draw_button,
    draw_panel_header,
    draw_spice_text,
    draw_wrapped_text,
)


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


# ─────────────────────────────────────────────────────────────────────────────


class SpiceRoadGUI:
    """Pygame GUI for Century: Spice Road."""

    def __init__(
        self,
        num_players: int = 4,
        seed: int = 42,
        ai_delay: float = 0.3,
    ) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Century: Spice Road")
        self.clock = pygame.time.Clock()

        self.gc = GameController(num_players=num_players, seed=seed)
        self.ai_delay = ai_delay
        self.wizard = WizardState.IDLE
        self._spice_callback: str = ""
        self._wizard_data: dict[str, Any] = {}
        self._exchange_first_done: bool = False

        # Clickable regions rebuilt every frame: (Rect, action_kind, index)
        self._clickables: list[tuple[pygame.Rect, str, int]] = []

        # Game log
        self._log_lines: list[str] = []
        self._log_scroll: int = 0

        # Fonts
        self._font_lg = pygame.font.Font(None, 28)
        self._font_md = pygame.font.Font(None, 22)
        self._font_sm = pygame.font.Font(None, 18)
        self._font_xs = pygame.font.Font(None, 15)

    # ── Main loop ───────────────────────────────────────────────────────────

    def run(self) -> None:
        self.gc.new_game()
        self._flush_game_log()
        self._enter_human_phase()

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        running = False
                    elif event.key == pygame.K_n:
                        self._new_game()
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self._handle_click(event.pos)
                elif event.type == pygame.MOUSEWHEEL:
                    if LOG_RECT.collidepoint(pygame.mouse.get_pos()):
                        self._log_scroll = max(0, self._log_scroll - event.y * 3)
                elif event.type == AI_STEP_EVENT:
                    self._flush_game_log()
                elif event.type == AI_DONE_EVENT:
                    self._flush_game_log()
                    self._enter_human_phase()

            self._render_all()
            self.clock.tick(FPS)

        pygame.quit()

    # ── Lifecycle helpers ───────────────────────────────────────────────────

    def _new_game(self) -> None:
        self.gc.new_game()
        self._log_lines.clear()
        self._log_scroll = 0
        self._flush_game_log()
        self._enter_human_phase()

    def _log(self, msg: str) -> None:
        self._log_lines.append(msg)

    def _flush_game_log(self) -> None:
        for msg in self.gc.log:
            self._log_lines.append(msg)
        self.gc.log.clear()

    # ── Phase entry ─────────────────────────────────────────────────────────

    def _enter_human_phase(self) -> None:
        if self.gc.game_over:
            self._show_game_over()
            return
        if not self.gc.is_human_turn:
            self._run_ai()
            return

        phase = self.gc.phase

        if phase == Phase.CHOOSE_ACTION:
            self._exchange_first_done = False
            mask = self.gc.get_action_type_mask()
            self._wizard_data = {"mask": mask}
            self.wizard = WizardState.PICK_ACTION_TYPE

        elif phase == Phase.EXECUTE_CARD:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            ct = ast["card_type"]
            if ct == CARD_TYPE_SPICE:
                self.gc.step([0, 0, 0, 0, 0, 1])
                self._flush_game_log()
                self._enter_human_phase()
            elif ct == CARD_TYPE_CONVERSION:
                legal = ast["upgradeable_spices"]
                if not legal:
                    self.gc.step([0, 0, 0, 0, 0, 1])
                    self._flush_game_log()
                    self._enter_human_phase()
                else:
                    self._spice_callback = "execute_conv"
                    remaining = ast["remaining_upgrades"]
                    total = ast["total_upgrades"]
                    self._wizard_data = {
                        "legal": legal,
                        "prompt": f"Upgrade spice ({remaining} left):",
                        "show_done": remaining < total,
                    }
                    self.wizard = WizardState.PICK_SPICE
            elif ct == CARD_TYPE_EXCHANGE:
                if not self._exchange_first_done:
                    self._exchange_first_done = True
                    self.gc.step([0, 0, 0, 0, 0, 0])
                    self._flush_game_log()
                    self._enter_human_phase()
                else:
                    can_again = ast.get("can_again", False)
                    self._wizard_data = {"can_again": can_again}
                    self.wizard = WizardState.PICK_CONTINUE
                    self._spice_callback = "execute_exch"

        elif phase == Phase.PLACE_SPICE:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            self._spice_callback = "place"
            self._wizard_data = {
                "legal": ast["placeable_spices"],
                "prompt": f"Place spice on market ({ast['placed']}/{ast['target_pos']}):",
                "show_done": False,
            }
            self.wizard = WizardState.PICK_SPICE

        elif phase == Phase.DISCARD_OVERFLOW:
            data = self.gc.get_display_data()
            ast = data["action_state"]
            self._spice_callback = "discard"
            self._wizard_data = {
                "legal": ast["discardable_spices"],
                "prompt": f"Discard spice (overflow by {ast['overflow']}):",
                "show_done": False,
            }
            self.wizard = WizardState.PICK_SPICE

    def _show_game_over(self) -> None:
        self.wizard = WizardState.GAME_OVER
        scores = self.gc.get_final_scores()
        self._wizard_data = {"scores": scores}
        self._flush_game_log()
        self._log("--- GAME OVER ---")
        for i, s in enumerate(scores):
            label = "You" if i == 0 else f"Player {i}"
            self._log(f"  {label}: {s} pts")

    # ── AI ──────────────────────────────────────────────────────────────────

    def _run_ai(self) -> None:
        self.wizard = WizardState.AI_THINKING
        self._wizard_data = {}
        t = threading.Thread(target=self._ai_loop, daemon=True)
        t.start()

    def _ai_loop(self) -> None:
        import time

        while not self.gc.game_over and not self.gc.is_human_turn:
            self.gc.step_ai()
            pygame.event.post(pygame.event.Event(AI_STEP_EVENT))
            time.sleep(self.ai_delay)
        pygame.event.post(pygame.event.Event(AI_DONE_EVENT))

    # ── Click dispatch ──────────────────────────────────────────────────────

    def _handle_click(self, pos: tuple[int, int]) -> None:
        for rect, kind, idx in self._clickables:
            if rect.collidepoint(pos):
                self._dispatch(kind, idx)
                return

    def _dispatch(self, kind: str, idx: int) -> None:
        if kind == "action" and self.wizard == WizardState.PICK_ACTION_TYPE:
            self._do_action_type(idx)
        elif kind == "hand" and self.wizard == WizardState.PICK_CARD:
            self._do_pick_index(idx, "card")
        elif kind == "market" and self.wizard == WizardState.PICK_MARKET:
            self._do_pick_index(idx, "market")
        elif kind == "scoring" and self.wizard == WizardState.PICK_SCORING:
            self._do_pick_index(idx, "scoring")
        elif kind == "spice" and self.wizard == WizardState.PICK_SPICE:
            self._do_pick_spice(idx)
        elif kind == "done" and self.wizard == WizardState.PICK_SPICE:
            self._do_spice_done()
        elif kind == "continue" and self.wizard == WizardState.PICK_CONTINUE:
            if idx == 0:
                self._do_continue_again()
            else:
                self._do_continue_done()
        elif kind == "new_game":
            self._new_game()

    # ── Action handlers (same logic as TUI) ─────────────────────────────────

    def _do_action_type(self, action_type: int) -> None:
        mask = self.gc.get_action_type_mask()
        if not mask[action_type]:
            self._log("That action is not legal.")
            return
        if action_type == 0:  # Play
            legal = self.gc.get_legal_hand_indices()
            self._wizard_data = {"legal": legal}
            self.wizard = WizardState.PICK_CARD
        elif action_type == 1:  # Acquire
            legal = self.gc.get_legal_market_positions()
            self._wizard_data = {"legal": legal}
            self.wizard = WizardState.PICK_MARKET
        elif action_type == 2:  # Rest
            self.gc.step([2, 0, 0, 0, 0, 0])
            self._flush_game_log()
            self._enter_human_phase()
        elif action_type == 3:  # Score
            legal = self.gc.get_legal_scoring_indices()
            self._wizard_data = {"legal": legal}
            self.wizard = WizardState.PICK_SCORING

    def _do_pick_index(self, idx: int, context: str) -> None:
        if context == "card":
            legal = self.gc.get_legal_hand_indices()
            if idx not in legal:
                self._log("Invalid card index.")
                return
            self.gc.step([0, idx, 0, 0, 0, 0])
        elif context == "market":
            legal = self.gc.get_legal_market_positions()
            if idx not in legal:
                self._log("Invalid market position.")
                return
            self.gc.step([1, 0, idx, 0, 0, 0])
        elif context == "scoring":
            legal = self.gc.get_legal_scoring_indices()
            if idx not in legal:
                self._log("Invalid scoring card.")
                return
            self.gc.step([3, 0, 0, idx, 0, 0])
        self._flush_game_log()
        self._enter_human_phase()

    def _do_pick_spice(self, spice: int) -> None:
        self.gc.step([0, 0, 0, 0, spice, 0])
        self._flush_game_log()
        self._enter_human_phase()

    def _do_spice_done(self) -> None:
        self.gc.step([0, 0, 0, 0, 0, 1])
        self._flush_game_log()
        self._enter_human_phase()

    def _do_continue_again(self) -> None:
        if self._spice_callback == "execute_exch":
            self.gc.step([0, 0, 0, 0, 0, 0])
            self._flush_game_log()
            self._enter_human_phase()

    def _do_continue_done(self) -> None:
        self.gc.step([0, 0, 0, 0, 0, 1])
        self._flush_game_log()
        self._enter_human_phase()

    # ── Rendering ───────────────────────────────────────────────────────────

    def _render_all(self) -> None:
        self._clickables.clear()
        self.screen.fill(BG_COLOR)
        data = self.gc.get_display_data()

        self._draw_header(data)
        self._draw_market(data)
        self._draw_scoring(data)
        self._draw_players(data)
        self._draw_hand(data)
        self._draw_caravan_info(data)
        self._draw_action_panel()
        self._draw_log()

        pygame.display.flip()

    # -- header --

    def _draw_header(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, HEADER_RECT)
        phase = data["phase"]
        player = data["current_player"]
        triggered = "  [TRIGGERED]" if data["game_triggered"] else ""
        text = f"Century: Spice Road   |   Phase: {phase.name}   |   Player: {player}{triggered}"
        surf = self._font_md.render(text, True, TEXT_COLOR)
        self.screen.blit(surf, (12, HEADER_RECT.centery - surf.get_height() // 2))

    # -- market row --

    def _draw_market(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, MARKET_RECT)
        draw_panel_header(self.screen, self._font_xs, MARKET_RECT, "Market Row")
        market = data["market"]
        n = len(market)
        if n == 0:
            return

        legal = set(self._wizard_data.get("legal", [])) if self.wizard == WizardState.PICK_MARKET else set()
        gap = 16
        total_w = n * TRADER_CARD_W + (n - 1) * gap
        sx = (WIDTH - total_w) // 2
        cy = MARKET_RECT.y + PANEL_HDR_H + 4

        for i, m in enumerate(market):
            r = pygame.Rect(sx + i * (TRADER_CARD_W + gap), cy, TRADER_CARD_W, TRADER_CARD_H)
            hl = i in legal
            en = (i in legal) if self.wizard == WizardState.PICK_MARKET else True
            draw_trader_card(self.screen, self._font_xs, m["card"], r, hl, en)
            # Position index
            idx_s = self._font_xs.render(f"#{i}", True, TEXT_DIM)
            self.screen.blit(idx_s, (r.x + 3, r.y + 3))
            # Spices on card
            if m["spices"]:
                draw_spice_text(self.screen, self._font_xs, m["spices"], (r.x + 5, r.bottom - 16))
            if self.wizard == WizardState.PICK_MARKET and hl:
                self._clickables.append((r, "market", i))

    # -- scoring row --

    def _draw_scoring(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, SCORING_RECT)
        draw_panel_header(self.screen, self._font_xs, SCORING_RECT, "Scoring Row")
        scoring = data["scoring"]
        n = len(scoring)
        if n == 0:
            return

        legal = set(self._wizard_data.get("legal", [])) if self.wizard == WizardState.PICK_SCORING else set()
        gap = 20
        total_w = n * SCORING_CARD_W + (n - 1) * gap
        sx = (WIDTH - total_w) // 2
        cy = SCORING_RECT.y + PANEL_HDR_H + 4

        for i, sc in enumerate(scoring):
            r = pygame.Rect(sx + i * (SCORING_CARD_W + gap), cy, SCORING_CARD_W, SCORING_CARD_H)
            hl = i in legal
            en = (i in legal) if self.wizard == WizardState.PICK_SCORING else True
            draw_scoring_card(self.screen, self._font_xs, self._font_md, sc["card"], r, hl, en)
            # Gold / silver badge
            if i == 0 and data["gold_remaining"] > 0:
                draw_coin_badge(self.screen, self._font_xs, r, "gold", data["gold_remaining"])
            elif i == 1 and data["silver_remaining"] > 0:
                draw_coin_badge(self.screen, self._font_xs, r, "silver", data["silver_remaining"])
            if self.wizard == WizardState.PICK_SCORING and hl:
                self._clickables.append((r, "scoring", i))

    # -- players --

    def _draw_players(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, PLAYERS_RECT)
        draw_panel_header(self.screen, self._font_xs, PLAYERS_RECT, "Players")
        players = data["players"]
        n = len(players)
        if n == 0:
            return

        slot_w = WIDTH // n
        cy = PLAYERS_RECT.y + PANEL_HDR_H + 4

        for p in players:
            i = p["idx"]
            x = i * slot_w + 10
            is_cur = i == data["current_player"]
            if is_cur:
                bg_r = pygame.Rect(i * slot_w, cy - 2, slot_w, 58)
                pygame.draw.rect(self.screen, CURRENT_PLAYER_BG, bg_r)

            label = "You" if i == 0 else f"P{i}"
            color = HIGHLIGHT_BORDER if is_cur else TEXT_COLOR
            lbl_s = self._font_sm.render(label, True, color)
            self.screen.blit(lbl_s, (x, cy))

            draw_caravan_cubes(self.screen, p["caravan"], (x, cy + 18), cube_size=12, gap=1)

            info = f"H:{p['hand_size']} P:{p['played_size']} S:{p['scored_count']} G:{p['gold']} Sv:{p['silver']}"
            info_s = self._font_xs.render(info, True, TEXT_DIM)
            self.screen.blit(info_s, (x, cy + 36))

    # -- hand cards --

    def _draw_hand(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, HAND_RECT)
        draw_panel_header(self.screen, self._font_xs, HAND_RECT, "Your Hand")
        hand = data["hand"]
        n = len(hand)
        if n == 0:
            return

        legal = set(self._wizard_data.get("legal", [])) if self.wizard == WizardState.PICK_CARD else set()
        gap = 8
        cw = min(HAND_CARD_W, (HAND_RECT.width - 20 - max(n - 1, 0) * gap) // max(n, 1))
        ch = min(HAND_CARD_H, int(cw * 1.33))
        sx = HAND_RECT.x + 10
        cy = HAND_RECT.y + PANEL_HDR_H + 6

        for i, h in enumerate(hand):
            r = pygame.Rect(sx + i * (cw + gap), cy, cw, ch)
            hl = i in legal
            en = (i in legal) if self.wizard == WizardState.PICK_CARD else True
            draw_trader_card(self.screen, self._font_xs, h["card"], r, hl, en)
            if self.wizard == WizardState.PICK_CARD and hl:
                self._clickables.append((r, "hand", i))

    # -- caravan + played info --

    def _draw_caravan_info(self, data: dict) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, CARAVAN_RECT)
        draw_panel_header(self.screen, self._font_xs, CARAVAN_RECT, "Your Caravan & Played")
        x = CARAVAN_RECT.x + 10
        y = CARAVAN_RECT.y + PANEL_HDR_H + 8

        lbl = self._font_sm.render("Caravan:", True, TEXT_COLOR)
        self.screen.blit(lbl, (x, y))
        draw_caravan_cubes(self.screen, data["human_caravan"], (x + lbl.get_width() + 8, y + 2))

        y += 26
        total = sum(data["human_caravan"])
        t_s = self._font_xs.render(f"Total: {total}/10", True, TEXT_DIM)
        self.screen.blit(t_s, (x, y))

        y += 22
        played = data["played"]
        p_s = self._font_sm.render(f"Played pile: {len(played)} card(s)", True, TEXT_COLOR)
        self.screen.blit(p_s, (x, y))

        y += 20
        for j, card_text in enumerate(played[:5]):
            ct = self._font_xs.render(card_text, True, TEXT_DIM)
            self.screen.blit(ct, (x + 8, y + j * 16))
        if len(played) > 5:
            more = self._font_xs.render(f"  ...and {len(played) - 5} more", True, TEXT_DIM)
            self.screen.blit(more, (x + 8, y + 5 * 16))

    # -- action panel --

    def _draw_action_panel(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, ACTION_RECT)
        draw_panel_header(self.screen, self._font_xs, ACTION_RECT, "Actions")
        mouse = pygame.mouse.get_pos()
        x0 = ACTION_RECT.x + 14
        y0 = ACTION_RECT.y + PANEL_HDR_H + 8

        if self.wizard == WizardState.PICK_ACTION_TYPE:
            mask = self._wizard_data.get("mask", [True] * 4)
            labels = ["Play", "Acquire", "Rest", "Score"]
            for i, lbl in enumerate(labels):
                r = pygame.Rect(x0 + (i % 2) * (BTN_W + 10), y0 + (i // 2) * (BTN_H + 8), BTN_W, BTN_H)
                en = bool(mask[i])
                hov = r.collidepoint(mouse) and en
                draw_button(self.screen, self._font_sm, r, lbl, en, hov)
                if en:
                    self._clickables.append((r, "action", i))

        elif self.wizard == WizardState.PICK_CARD:
            s = self._font_sm.render("Click a hand card to play.", True, TEXT_COLOR)
            self.screen.blit(s, (x0, y0))

        elif self.wizard == WizardState.PICK_MARKET:
            s = self._font_sm.render("Click a market card to acquire.", True, TEXT_COLOR)
            self.screen.blit(s, (x0, y0))

        elif self.wizard == WizardState.PICK_SCORING:
            s = self._font_sm.render("Click a scoring card to claim.", True, TEXT_COLOR)
            self.screen.blit(s, (x0, y0))

        elif self.wizard == WizardState.PICK_SPICE:
            legal = self._wizard_data.get("legal", [])
            prompt = self._wizard_data.get("prompt", "Pick a spice:")
            show_done = self._wizard_data.get("show_done", False)
            ps = self._font_sm.render(prompt, True, TEXT_COLOR)
            self.screen.blit(ps, (x0, y0))
            by = y0 + 28
            for si in range(4):
                r = pygame.Rect(x0 + si * (BTN_W + 8), by, BTN_W, BTN_H)
                en = si in legal
                hov = r.collidepoint(mouse) and en
                color = SPICE_COLORS[si]
                lbl = SPICE_NAMES_SHORT[si]
                _draw_spice_button(self.screen, self._font_sm, r, lbl, color, en, hov)
                if en:
                    self._clickables.append((r, "spice", si))
            if show_done:
                dr = pygame.Rect(x0, by + BTN_H + 10, BTN_W, BTN_H)
                hov = dr.collidepoint(mouse)
                draw_button(self.screen, self._font_sm, dr, "Done", True, hov)
                self._clickables.append((dr, "done", 0))

        elif self.wizard == WizardState.PICK_CONTINUE:
            can_again = self._wizard_data.get("can_again", False)
            ps = self._font_sm.render("Exchange again or done?", True, TEXT_COLOR)
            self.screen.blit(ps, (x0, y0))
            by = y0 + 28
            ar = pygame.Rect(x0, by, BTN_W, BTN_H)
            hov_a = ar.collidepoint(mouse) and can_again
            draw_button(self.screen, self._font_sm, ar, "Again", can_again, hov_a)
            if can_again:
                self._clickables.append((ar, "continue", 0))
            dr = pygame.Rect(x0 + BTN_W + 12, by, BTN_W, BTN_H)
            hov_d = dr.collidepoint(mouse)
            draw_button(self.screen, self._font_sm, dr, "Done", True, hov_d)
            self._clickables.append((dr, "continue", 1))

        elif self.wizard == WizardState.AI_THINKING:
            s = self._font_md.render("AI thinking ...", True, HIGHLIGHT_BORDER)
            self.screen.blit(s, (x0, y0 + 20))

        elif self.wizard == WizardState.GAME_OVER:
            scores = self._wizard_data.get("scores", [])
            s = self._font_md.render("Game Over!", True, (244, 67, 54))
            self.screen.blit(s, (x0, y0))
            for i, sc in enumerate(scores):
                label = "You" if i == 0 else f"Player {i}"
                txt = self._font_sm.render(f"{label}: {sc} pts", True, TEXT_COLOR)
                self.screen.blit(txt, (x0, y0 + 28 + i * 20))
            nr = pygame.Rect(x0, y0 + 28 + len(scores) * 20 + 10, BTN_W, BTN_H)
            hov = nr.collidepoint(mouse)
            draw_button(self.screen, self._font_sm, nr, "New Game", True, hov)
            self._clickables.append((nr, "new_game", 0))

    # -- game log --

    def _draw_log(self) -> None:
        pygame.draw.rect(self.screen, PANEL_BG, LOG_RECT)
        draw_panel_header(self.screen, self._font_xs, LOG_RECT, "Game Log (scroll)")
        clip = pygame.Rect(LOG_RECT.x + 4, LOG_RECT.y + PANEL_HDR_H + 2, LOG_RECT.width - 8, LOG_RECT.height - PANEL_HDR_H - 4)
        self.screen.set_clip(clip)
        line_h = 16
        visible = clip.height // line_h
        total = len(self._log_lines)
        # Auto-scroll to bottom unless user scrolled up
        max_scroll = max(0, total - visible)
        if self._log_scroll > max_scroll:
            self._log_scroll = max_scroll
        start = max(0, total - visible - self._log_scroll)
        y = clip.y
        for line in self._log_lines[start : start + visible]:
            s = self._font_xs.render(line, True, TEXT_DIM)
            self.screen.blit(s, (clip.x + 4, y))
            y += line_h
        self.screen.set_clip(None)


# ── Spice-colored button (helper) ──────────────────────────────────────────


def _draw_spice_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    label: str,
    color: tuple[int, int, int],
    enabled: bool,
    hovered: bool,
) -> None:
    if not enabled:
        bg = (70, 70, 70)
    elif hovered:
        bg = tuple(min(c + 30, 255) for c in color)
    else:
        bg = color
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    pygame.draw.rect(surface, (0, 0, 0), rect, width=1, border_radius=4)
    txt_color = (0, 0, 0) if enabled else TEXT_DIM
    s = font.render(label, True, txt_color)
    surface.blit(s, (rect.centerx - s.get_width() // 2, rect.centery - s.get_height() // 2))
