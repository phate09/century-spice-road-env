"""Custom Textual widgets for Century: Spice Road TUI."""

from __future__ import annotations

from rich.text import Text
from textual.containers import Container
from textual.message import Message
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


class ItemClicked(Message):
    """Posted when a clickable item is clicked."""

    def __init__(self, panel_type: str, index: int) -> None:
        self.panel_type = panel_type
        self.index = index
        super().__init__()


class ClickableItem(Static):
    """A row that responds to mouse clicks."""

    def __init__(
        self,
        content: Text | str,
        panel_type: str,
        index: int,
        disabled: bool = False,
        **kw,
    ) -> None:
        super().__init__(content, **kw)
        self._panel_type = panel_type
        self._index = index
        if disabled:
            self.add_class("-disabled")

    def on_click(self) -> None:
        if "-disabled" not in self.classes:
            self.post_message(ItemClicked(self._panel_type, self._index))


class MarketPanel(Container):
    """Displays the market row of trader cards."""

    def update_data(self, market: list[dict], gold_rem: int, silver_rem: int) -> None:
        self.remove_children()
        header = Text("MARKET ROW", style="bold underline")
        children: list[Static] = [Static(header)]
        for m in market:
            t = Text()
            t.append(f"  [{m['idx']}] ", style="bold cyan")
            t.append(m["card"])
            if m["spices"]:
                t.append(f"  +{m['spices']}", style="dim")
            children.append(ClickableItem(t, "market", m["idx"]))
        self.mount(*children)


class ScoringPanel(Container):
    """Displays the scoring row."""

    def update_data(self, scoring: list[dict], gold_rem: int, silver_rem: int) -> None:
        self.remove_children()
        header = Text("SCORING ROW", style="bold underline")
        children: list[Static] = [Static(header)]
        for s in scoring:
            t = Text()
            t.append(f"  [{s['idx']}] ", style="bold cyan")
            t.append(s["card"])
            if s["idx"] == 0 and gold_rem > 0:
                t.append(f"  [G:{gold_rem}]", style="bold yellow")
            elif s["idx"] == 1 and gold_rem > 0 and silver_rem > 0:
                t.append(f"  [S:{silver_rem}]", style="white")
            children.append(ClickableItem(t, "scoring", s["idx"]))
        self.mount(*children)


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


class HandPanel(Container):
    """Displays the human player's hand and played pile."""

    def update_data(
        self,
        hand: list[dict],
        played: list[str],
        caravan: list[int],
    ) -> None:
        self.remove_children()
        header = Text("YOUR HAND", style="bold underline")
        children: list[Static] = [Static(header)]
        for h in hand:
            t = Text()
            t.append(f"  [{h['idx']}] ", style="bold cyan")
            t.append(h["card"])
            children.append(ClickableItem(t, "hand", h["idx"]))
        if played:
            t = Text()
            t.append("Played: ", style="dim italic")
            t.append(", ".join(played), style="dim")
            children.append(Static(t))
        t = Text()
        t.append("Caravan: ", style="bold")
        t.append_text(colored_spices(caravan))
        total = sum(caravan)
        t.append(f" ({total}/10)", style="dim")
        children.append(Static(t))
        self.mount(*children)


class ActionPanel(Container):
    """Displays available actions / wizard prompts."""

    def _rebuild(self, *children: Static) -> None:
        self.remove_children()
        if children:
            self.mount(*children)

    def show_choose_action(self, mask: list[bool]) -> None:
        header = Text("Choose action:", style="bold underline")
        names = [
            ("[P]lay", "Play a card from hand"),
            ("[A]cquire", "Take a market card"),
            ("[R]est", "Return played cards"),
            ("[S]core", "Claim a scoring card"),
        ]
        children: list[Static] = [Static(header)]
        for i, (key, desc) in enumerate(names):
            t = Text()
            if mask[i]:
                t.append(f"  {key}", style="bold green")
                t.append(f"  {desc}", style="dim")
            else:
                t.append(f"  {key}", style="dim strike")
                t.append(f"  {desc}", style="dim strike")
            children.append(ClickableItem(t, "action", i, disabled=not mask[i]))
        children.append(Static(Text("  [Q]uit  [N]ew game", style="dim")))
        self._rebuild(*children)

    def show_pick_card(self, hand: list[dict]) -> None:
        header = Text("Pick card to play:", style="bold underline")
        children: list[Static] = [Static(header)]
        for h in hand:
            t = Text()
            t.append(f"  [{h['idx']}] ", style="bold cyan")
            t.append(h["card"])
            children.append(ClickableItem(t, "hand", h["idx"]))
        children.append(Static(Text("  [Esc] Cancel", style="dim")))
        self._rebuild(*children)

    def show_pick_market(self, legal: list[int], market: list[dict]) -> None:
        header = Text("Pick market position:", style="bold underline")
        children: list[Static] = [Static(header)]
        for m in market:
            idx = m["idx"]
            t = Text()
            if idx in legal:
                cost = (
                    f"(cost: {idx} spice{'s' if idx != 1 else ''})"
                    if idx > 0
                    else "(free)"
                )
                t.append(f"  [{idx}] ", style="bold cyan")
                t.append(f"{m['card']} {cost}")
            else:
                t.append(f"  [{idx}] ", style="dim strike")
                t.append(m["card"], style="dim strike")
            children.append(ClickableItem(t, "market", idx, disabled=idx not in legal))
        children.append(Static(Text("  [Esc] Cancel", style="dim")))
        self._rebuild(*children)

    def show_pick_scoring(self, legal: list[int], scoring: list[dict]) -> None:
        header = Text("Pick scoring card:", style="bold underline")
        children: list[Static] = [Static(header)]
        for s in scoring:
            idx = s["idx"]
            t = Text()
            if idx in legal:
                t.append(f"  [{idx}] ", style="bold cyan")
                t.append(s["card"])
            else:
                t.append(f"  [{idx}] ", style="dim strike")
                t.append(s["card"], style="dim strike")
            children.append(
                ClickableItem(t, "scoring", idx, disabled=idx not in legal)
            )
        children.append(Static(Text("  [Esc] Cancel", style="dim")))
        self._rebuild(*children)

    def show_pick_spice(
        self, legal: list[int], prompt: str, show_done: bool = False
    ) -> None:
        header = Text(prompt, style="bold underline")
        keys = "yrgb"
        children: list[Static] = [Static(header)]
        for i in range(4):
            t = Text()
            if i in legal:
                t.append(f"  [{keys[i].upper()}] ", style="bold cyan")
                t.append_text(colored_spice_char(i))
                t.append(f" {SPICE_LABEL[i]}")
            else:
                t.append(
                    f"  [{keys[i].upper()}] {SPICE_LABEL[i]}", style="dim strike"
                )
            children.append(ClickableItem(t, "spice", i, disabled=i not in legal))
        if show_done:
            children.append(
                ClickableItem(Text("  [D]one", style="bold green"), "done", 0)
            )
        children.append(Static(Text("  [Esc] Cancel", style="dim")))
        self._rebuild(*children)

    def show_pick_continue(self, can_again: bool) -> None:
        header = Text("Continue?", style="bold underline")
        children: list[Static] = [Static(header)]
        t = Text()
        if can_again:
            t.append("  [A]gain", style="bold green")
        else:
            t.append("  [A]gain", style="dim strike")
        children.append(ClickableItem(t, "continue", 0, disabled=not can_again))
        children.append(
            ClickableItem(Text("  [D]one", style="bold green"), "continue", 1)
        )
        self._rebuild(*children)

    def show_waiting(self) -> None:
        self._rebuild(Static(Text("AI is thinking...", style="bold italic yellow")))

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
        self._rebuild(Static(t))
