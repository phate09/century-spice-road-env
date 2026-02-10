"""Custom Textual widgets for Century: Spice Road TUI."""

from __future__ import annotations

from rich.text import Text
from textual.widgets import Static

SPICE_STYLE = {
    0: "bold yellow",       # Yellow / Turmeric
    1: "bold red",          # Red / Saffron
    2: "bold green",        # Green / Cardamom
    3: "bold #8B4513",      # Brown / Cinnamon
}
SPICE_CHAR = ["Y", "R", "G", "B"]
SPICE_LABEL = ["Yellow", "Red", "Green", "Brown"]


def colored_spices(caravan: list[int]) -> Text:
    """Return a Rich Text with each spice letter colored."""
    t = Text()
    first = True
    for i, count in enumerate(caravan):
        if count > 0:
            if not first:
                t.append(" ")
            t.append(SPICE_CHAR[i] * count, style=SPICE_STYLE[i])
            first = False
    if first:
        t.append("(empty)", style="dim")
    return t


def colored_spice_char(idx: int) -> Text:
    """Single colored spice letter."""
    return Text(SPICE_CHAR[idx], style=SPICE_STYLE[idx])


class MarketPanel(Static):
    """Displays the market row of trader cards."""

    def update_data(self, market: list[dict], gold_rem: int, silver_rem: int) -> None:
        t = Text()
        t.append("MARKET ROW\n", style="bold underline")
        for m in market:
            t.append(f"  [{m['idx']}] ", style="bold cyan")
            t.append(m["card"])
            if m["spices"]:
                t.append(f"  +{m['spices']}", style="dim")
            t.append("\n")
        self.update(t)


class ScoringPanel(Static):
    """Displays the scoring row."""

    def update_data(self, scoring: list[dict], gold_rem: int, silver_rem: int) -> None:
        t = Text()
        t.append("SCORING ROW\n", style="bold underline")
        for s in scoring:
            t.append(f"  [{s['idx']}] ", style="bold cyan")
            t.append(s["card"])
            if s["idx"] == 0 and gold_rem > 0:
                t.append(f"  [G:{gold_rem}]", style="bold yellow")
            elif s["idx"] == 1 and gold_rem > 0 and silver_rem > 0:
                t.append(f"  [S:{silver_rem}]", style="white")
            t.append("\n")
        self.update(t)


class PlayersPanel(Static):
    """Displays all player summaries."""

    def update_data(self, players: list[dict], current: int) -> None:
        t = Text()
        t.append("PLAYERS\n", style="bold underline")
        for p in players:
            idx = p["idx"]
            marker = "> " if idx == current else "  "
            label = "You" if idx == 0 else f"P{idx}"
            style = "bold" if idx == current else ""
            t.append(f"{marker}{label} ", style=style)
            t.append_text(colored_spices(p["caravan"]))
            t.append(f" ({p['caravan_total']}/10)", style="dim")
            t.append(f"  Sc:{p['scored_count']}", style="magenta")
            if p["gold"]:
                t.append(f" Au:{p['gold']}", style="yellow")
            if p["silver"]:
                t.append(f" Ag:{p['silver']}", style="white")
            t.append(f"  H:{p['hand_size']}", style="dim")
            t.append("\n")
        self.update(t)


class HandPanel(Static):
    """Displays the human player's hand and played pile."""

    def update_data(
        self,
        hand: list[dict],
        played: list[str],
        caravan: list[int],
    ) -> None:
        t = Text()
        t.append("YOUR HAND\n", style="bold underline")
        for h in hand:
            t.append(f"  [{h['idx']}] ", style="bold cyan")
            t.append(h["card"])
            t.append("\n")
        if played:
            t.append("Played: ", style="dim italic")
            t.append(", ".join(played), style="dim")
            t.append("\n")
        t.append("Caravan: ", style="bold")
        t.append_text(colored_spices(caravan))
        total = sum(caravan)
        t.append(f" ({total}/10)", style="dim")
        self.update(t)


class ActionPanel(Static):
    """Displays available actions / wizard prompts."""

    DEFAULT_CSS = "ActionPanel { height: auto; }"

    def show_choose_action(self, mask: list[bool]) -> None:
        t = Text()
        t.append("Choose action:\n", style="bold underline")
        names = [
            ("[P]lay", "Play a card from hand"),
            ("[A]cquire", "Take a market card"),
            ("[R]est", "Return played cards"),
            ("[S]core", "Claim a scoring card"),
        ]
        for i, (key, desc) in enumerate(names):
            if mask[i]:
                t.append(f"  {key}", style="bold green")
                t.append(f"  {desc}\n", style="dim")
            else:
                t.append(f"  {key}", style="dim strike")
                t.append(f"  {desc}\n", style="dim strike")
        t.append("\n  [Q]uit  [N]ew game", style="dim")
        self.update(t)

    def show_pick_card(self, hand: list[dict]) -> None:
        t = Text()
        t.append("Pick card to play:\n", style="bold underline")
        for h in hand:
            t.append(f"  [{h['idx']}] ", style="bold cyan")
            t.append(h["card"])
            t.append("\n")
        t.append("  [Esc] Cancel", style="dim")
        self.update(t)

    def show_pick_market(self, legal: list[int], market: list[dict]) -> None:
        t = Text()
        t.append("Pick market position:\n", style="bold underline")
        for m in market:
            idx = m["idx"]
            if idx in legal:
                cost = f"(cost: {idx} spice{'s' if idx != 1 else ''})" if idx > 0 else "(free)"
                t.append(f"  [{idx}] ", style="bold cyan")
                t.append(f"{m['card']} {cost}")
            else:
                t.append(f"  [{idx}] ", style="dim strike")
                t.append(m["card"], style="dim strike")
            t.append("\n")
        t.append("  [Esc] Cancel", style="dim")
        self.update(t)

    def show_pick_scoring(self, legal: list[int], scoring: list[dict]) -> None:
        t = Text()
        t.append("Pick scoring card:\n", style="bold underline")
        for s in scoring:
            idx = s["idx"]
            if idx in legal:
                t.append(f"  [{idx}] ", style="bold cyan")
                t.append(s["card"])
            else:
                t.append(f"  [{idx}] ", style="dim strike")
                t.append(s["card"], style="dim strike")
            t.append("\n")
        t.append("  [Esc] Cancel", style="dim")
        self.update(t)

    def show_pick_spice(self, legal: list[int], prompt: str) -> None:
        t = Text()
        t.append(f"{prompt}\n", style="bold underline")
        keys = "yrgb"
        for i in range(4):
            if i in legal:
                t.append(f"  [{keys[i].upper()}] ", style="bold cyan")
                t.append_text(colored_spice_char(i))
                t.append(f" {SPICE_LABEL[i]}\n")
            else:
                t.append(f"  [{keys[i].upper()}] {SPICE_LABEL[i]}\n", style="dim strike")
        t.append("  [Esc] Cancel", style="dim")
        self.update(t)

    def show_pick_continue(self, can_again: bool) -> None:
        t = Text()
        t.append("Continue?\n", style="bold underline")
        if can_again:
            t.append("  [A]gain\n", style="bold green")
        else:
            t.append("  [A]gain\n", style="dim strike")
        t.append("  [D]one\n", style="bold green")
        self.update(t)

    def show_waiting(self) -> None:
        t = Text()
        t.append("AI is thinking...", style="bold italic yellow")
        self.update(t)

    def show_game_over(self, scores: list[int]) -> None:
        t = Text()
        t.append("GAME OVER\n\n", style="bold red")
        for i, score in enumerate(scores):
            label = "You" if i == 0 else f"Player {i}"
            t.append(f"  {label}: {score} pts\n", style="bold")
        winner = max(range(len(scores)), key=lambda i: scores[i])
        if winner == 0:
            t.append("\n  YOU WIN!", style="bold green")
        else:
            t.append(f"\n  Player {winner} wins!", style="bold magenta")
        t.append("\n\n  [N]ew game  [Q]uit", style="dim")
        self.update(t)
