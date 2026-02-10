"""Layout, color, and size constants for the Pygame GUI (1280x800)."""

import pygame

# ── Window ──────────────────────────────────────────────────────────────────
WIDTH = 1280
HEIGHT = 800
FPS = 30

# ── Base colors ─────────────────────────────────────────────────────────────
BG_COLOR = (30, 30, 30)
PANEL_BG = (45, 45, 45)
PANEL_HEADER_BG = (60, 60, 60)
TEXT_COLOR = (220, 220, 220)
TEXT_DIM = (140, 140, 140)
HIGHLIGHT_BORDER = (255, 215, 0)  # Gold highlight for legal/selected items
CURRENT_PLAYER_BG = (60, 70, 60)

# ── Spice colors ────────────────────────────────────────────────────────────
SPICE_COLORS = {
    0: (255, 235, 59),   # Yellow (Turmeric)
    1: (244, 67, 54),    # Red (Saffron)
    2: (76, 175, 80),    # Green (Cardamom)
    3: (121, 85, 72),    # Brown (Cinnamon)
}
SPICE_LETTERS = ["Y", "R", "G", "B"]
SPICE_NAMES_SHORT = ["Yellow", "Red", "Green", "Brown"]

# ── Card background colors by type ──────────────────────────────────────────
CARD_BG = {
    "spice": (46, 125, 50),       # Green-ish
    "conversion": (25, 118, 210),  # Blue-ish
    "exchange": (230, 115, 0),     # Orange-ish
    "scoring": (123, 31, 162),     # Purple-ish
    "unknown": (80, 80, 80),
}
CARD_BG_DISABLED = (60, 60, 60)

# ── Card sizes ──────────────────────────────────────────────────────────────
TRADER_CARD_W = 100
TRADER_CARD_H = 140
SCORING_CARD_W = 100
SCORING_CARD_H = 120
HAND_CARD_W = 90
HAND_CARD_H = 120

# ── Buttons ─────────────────────────────────────────────────────────────────
BTN_W = 110
BTN_H = 34
BTN_COLOR = (55, 120, 170)
BTN_HOVER = (75, 145, 200)
BTN_DISABLED = (70, 70, 70)
BTN_TEXT_COLOR = (255, 255, 255)

# ── Gold / Silver coin badge ────────────────────────────────────────────────
GOLD_COLOR = (255, 215, 0)
SILVER_COLOR = (192, 192, 192)

# ── Layout regions (x, y, w, h) ────────────────────────────────────────────
HEADER_RECT = pygame.Rect(0, 0, WIDTH, 45)
MARKET_RECT = pygame.Rect(0, 45, WIDTH, 170)
SCORING_RECT = pygame.Rect(0, 215, WIDTH, 150)
PLAYERS_RECT = pygame.Rect(0, 365, WIDTH, 85)
HAND_RECT = pygame.Rect(0, 450, 800, 160)
CARAVAN_RECT = pygame.Rect(800, 450, 480, 160)
ACTION_RECT = pygame.Rect(0, 610, 500, 190)
LOG_RECT = pygame.Rect(500, 610, 780, 190)

# ── Panel header height ────────────────────────────────────────────────────
PANEL_HDR_H = 22

# ── Pygame custom events ───────────────────────────────────────────────────
AI_STEP_EVENT = pygame.USEREVENT + 1
AI_DONE_EVENT = pygame.USEREVENT + 2
