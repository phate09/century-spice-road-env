"""Pure drawing helpers for the Pygame GUI — no game logic here."""

from __future__ import annotations

import pygame

from century_env.gui.constants import (
    CARD_BG,
    CARD_BG_DISABLED,
    TEXT_COLOR,
    TEXT_DIM,
    HIGHLIGHT_BORDER,
    SPICE_COLORS,
    SPICE_LETTERS,
    BTN_COLOR,
    BTN_HOVER,
    BTN_DISABLED,
    BTN_TEXT_COLOR,
    PANEL_HEADER_BG,
    PANEL_HDR_H,
    GOLD_COLOR,
    SILVER_COLOR,
)


# ── Card type detection from display text ───────────────────────────────────

def card_type_from_text(text: str) -> str:
    if text.startswith("Obtain"):
        return "spice"
    if text.startswith("Upgrade"):
        return "conversion"
    if "\u2192" in text or "->" in text:
        return "exchange"
    return "unknown"


# ── Text helpers ────────────────────────────────────────────────────────────

def draw_wrapped_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    rect: pygame.Rect,
    color: tuple[int, int, int],
    padding: int = 6,
    center: bool = True,
) -> None:
    """Word-wrap *text* inside *rect* with optional centering."""
    words = text.split()
    if not words:
        return
    max_w = rect.width - 2 * padding
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}" if current else word
        if font.size(test)[0] <= max_w:
            current = test
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)

    y = rect.y + padding
    for line in lines:
        surf = font.render(line, True, color)
        x = rect.x + (rect.width - surf.get_width()) // 2 if center else rect.x + padding
        surface.blit(surf, (x, y))
        y += surf.get_height() + 2


def draw_spice_text(
    surface: pygame.Surface,
    font: pygame.font.Font,
    text: str,
    pos: tuple[int, int],
) -> int:
    """Render spice characters in their spice color. Returns total width."""
    color_map = {
        "Y": SPICE_COLORS[0],
        "R": SPICE_COLORS[1],
        "G": SPICE_COLORS[2],
        "B": SPICE_COLORS[3],
    }
    x, y = pos
    for ch in text:
        color = color_map.get(ch, TEXT_DIM)
        surf = font.render(ch, True, color)
        surface.blit(surf, (x, y))
        x += surf.get_width()
    return x - pos[0]


# ── Card drawing ────────────────────────────────────────────────────────────

def draw_trader_card(
    surface: pygame.Surface,
    font: pygame.font.Font,
    card_text: str,
    rect: pygame.Rect,
    highlighted: bool = False,
    enabled: bool = True,
) -> None:
    card_text = card_text.replace("\u2192", "->")
    ctype = card_type_from_text(card_text)
    bg = CARD_BG.get(ctype, CARD_BG["unknown"]) if enabled else CARD_BG_DISABLED
    pygame.draw.rect(surface, bg, rect, border_radius=6)
    if highlighted:
        pygame.draw.rect(surface, HIGHLIGHT_BORDER, rect, width=3, border_radius=6)
    else:
        pygame.draw.rect(surface, (0, 0, 0), rect, width=1, border_radius=6)
    color = TEXT_COLOR if enabled else TEXT_DIM
    text_rect = pygame.Rect(rect.x, rect.y + 10, rect.width, rect.height - 20)
    draw_wrapped_text(surface, font, card_text, text_rect, color)


def draw_scoring_card(
    surface: pygame.Surface,
    font: pygame.font.Font,
    font_large: pygame.font.Font,
    card_text: str,
    rect: pygame.Rect,
    highlighted: bool = False,
    enabled: bool = True,
) -> None:
    bg = CARD_BG["scoring"] if enabled else CARD_BG_DISABLED
    pygame.draw.rect(surface, bg, rect, border_radius=6)
    if highlighted:
        pygame.draw.rect(surface, HIGHLIGHT_BORDER, rect, width=3, border_radius=6)
    else:
        pygame.draw.rect(surface, (0, 0, 0), rect, width=1, border_radius=6)

    color = TEXT_COLOR if enabled else TEXT_DIM
    # Parse "YYRR = 8pts" → points + requirement
    pts_str = ""
    req_str = card_text
    if " = " in card_text:
        parts = card_text.split(" = ")
        req_str = parts[0]
        pts_str = parts[1]

    # Points at top
    if pts_str:
        pts_surf = font_large.render(pts_str, True, color)
        surface.blit(pts_surf, (rect.centerx - pts_surf.get_width() // 2, rect.y + 8))

    # Requirement spice text below
    req_y = rect.y + 40
    draw_spice_text(
        surface,
        font,
        req_str,
        (rect.centerx - font.size(req_str)[0] // 2, req_y),
    )


def draw_coin_badge(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    coin_type: str,
    remaining: int,
) -> None:
    """Draw a small gold/silver badge at the top-right of a scoring card rect."""
    color = GOLD_COLOR if coin_type == "gold" else SILVER_COLOR
    badge_r = 10
    cx = rect.right - 14
    cy = rect.y + 14
    pygame.draw.circle(surface, color, (cx, cy), badge_r)
    pygame.draw.circle(surface, (0, 0, 0), (cx, cy), badge_r, 1)
    txt = font.render(str(remaining), True, (0, 0, 0))
    surface.blit(txt, (cx - txt.get_width() // 2, cy - txt.get_height() // 2))


# ── Spice cubes ─────────────────────────────────────────────────────────────

def draw_caravan_cubes(
    surface: pygame.Surface,
    caravan: list[int],
    pos: tuple[int, int],
    cube_size: int = 14,
    gap: int = 2,
) -> int:
    """Draw caravan as colored squares. Returns total width."""
    x, y = pos
    for spice_idx, count in enumerate(caravan):
        color = SPICE_COLORS[spice_idx]
        for _ in range(count):
            pygame.draw.rect(surface, color, (x, y, cube_size, cube_size))
            pygame.draw.rect(surface, (0, 0, 0), (x, y, cube_size, cube_size), 1)
            x += cube_size + gap
    return x - pos[0]


# ── Buttons ─────────────────────────────────────────────────────────────────

def draw_button(
    surface: pygame.Surface,
    font: pygame.font.Font,
    rect: pygame.Rect,
    text: str,
    enabled: bool = True,
    hovered: bool = False,
) -> None:
    if not enabled:
        bg = BTN_DISABLED
    elif hovered:
        bg = BTN_HOVER
    else:
        bg = BTN_COLOR
    pygame.draw.rect(surface, bg, rect, border_radius=4)
    pygame.draw.rect(surface, (0, 0, 0), rect, width=1, border_radius=4)
    color = BTN_TEXT_COLOR if enabled else TEXT_DIM
    surf = font.render(text, True, color)
    surface.blit(surf, (rect.centerx - surf.get_width() // 2, rect.centery - surf.get_height() // 2))


# ── Panel header ────────────────────────────────────────────────────────────

def draw_panel_header(
    surface: pygame.Surface,
    font: pygame.font.Font,
    panel_rect: pygame.Rect,
    title: str,
) -> None:
    hdr = pygame.Rect(panel_rect.x, panel_rect.y, panel_rect.width, PANEL_HDR_H)
    pygame.draw.rect(surface, PANEL_HEADER_BG, hdr)
    surf = font.render(title, True, TEXT_COLOR)
    surface.blit(surf, (hdr.x + 10, hdr.y + 3))
