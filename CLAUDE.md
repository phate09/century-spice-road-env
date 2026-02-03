# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Century: Spice Road RL environment for the [Jumanji](https://github.com/instadeepai/jumanji) framework. A JAX-based, fully JIT-compilable implementation of the board game supporting 2-5 players.

## Commands

```bash
# Install (editable with dev dependencies)
pip install -e ".[dev]"

# Run all tests
pytest century_env/tests/ -v

# Run single test file
pytest century_env/tests/test_env.py -v

# Run single test
pytest century_env/tests/test_env.py::TestReset::test_reset_initializes_valid_state -v

# Run tests in parallel
pytest century_env/tests/ -n auto
```

## Architecture

### Core Flow

The environment follows Jumanji's `Environment` interface with `reset(key)` and `step(state, action)` methods. All game logic is JIT-compilable via JAX.

### Module Responsibilities

- **types.py**: State/Observation dataclasses using `@chex.dataclass`. Defines `Phase` and `ActionType` enums for the turn state machine.
- **cards.py**: Static card data arrays (`ALL_TRADER_CARDS`, `SCORING_CARDS`) and card query functions.
- **mechanics.py**: Pure functions for card execution (`apply_spice_card`, `apply_conversion`, `apply_exchange`).
- **masks.py**: Legal action mask computation via `get_action_mask(state)` returning a 6-tuple of boolean arrays.
- **transitions.py**: State mutation functions (`transition_choose_action`, `transition_execute_card`, etc.) using `jax.lax.switch` for phase dispatch.
- **env.py**: `CenturySpiceRoad` environment class orchestrating reset/step/observation logic.

### Turn State Machine

```
CHOOSE_ACTION -> Play    -> EXECUTE_CARD (loop) -> [DISCARD_OVERFLOW?] -> done
             -> Acquire -> PLACE_SPICE (loop)  -> [DISCARD_OVERFLOW?] -> done
             -> Rest    -> done
             -> Score   -> done
```

### Action Format

6-element multi-discrete array: `[action_type, card_idx, market_pos, scoring_idx, spice_type, continue_flag]`. Only relevant fields are used per phase.

### Observations

Ego-centric: current player always sees themselves as "player 0". Opponents are rotated so `opp[0]` is the next player clockwise. Includes action mask tuple for legal actions.

### Testing Patterns

Tests use session-scoped fixtures with pre-warmed JIT cache (see `conftest.py`). Helper functions like `make_action()`, `play_action()`, `acquire_action()` simplify action construction in tests.
