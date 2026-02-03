# Century: Spice Road - Jumanji RL Environment

A JAX-based reinforcement learning environment for the Century: Spice Road board game, compatible with the [Jumanji](https://github.com/instadeepai/jumanji) framework.

## Features

- **Full game rules**: Complete implementation of Century: Spice Road with no simplifications
- **2-5 players**: Variable player count with proper starting conditions
- **JAX-native**: All game logic is JIT-compilable for fast training
- **Ego-centric observations**: Current player always sees themselves as "player 0"
- **Multi-discrete action space**: 6-head action space for phase-based turn structure
- **Vectorizable**: Supports `jax.vmap` for parallel environment execution

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
import jax
from century_env import CenturySpiceRoad

# Create environment
env = CenturySpiceRoad(num_players=4)

# Reset
key = jax.random.PRNGKey(42)
state, timestep = env.reset(key)

# Take an action
action = jnp.array([0, 0, 0, 0, 0, 1], dtype=jnp.int32)  # Play card 0
state, timestep = env.step(state, action)
```

## Action Space

The action is a 6-element multi-discrete array:

| Head | Size | Description |
|------|------|-------------|
| action_type | 4 | Play / Acquire / Rest / Score |
| card_idx | 25 | Hand card selection |
| market_pos | 6 | Market position selection |
| scoring_idx | 5 | Scoring card selection |
| spice_type | 4 | Spice type selection |
| continue_flag | 2 | AGAIN (0) / DONE (1) |

## Game Phases

```
CHOOSE_ACTION -> [Play|Acquire|Rest|Score]
    Play -> EXECUTE_CARD (loop for conversion/exchange) -> done
    Acquire -> PLACE_SPICE (loop) -> done
    Rest -> done
    Score -> done
```

## Running Tests

```bash
pytest century_env/tests/ -v
```

## Project Structure

```
century_env/
├── __init__.py          # Public exports
├── constants.py         # Game constants
├── types.py             # State, Observation, Phase dataclasses
├── cards.py             # Card data and utilities
├── mechanics.py         # Pure functions for card execution
├── masks.py             # Legal action computation
├── transitions.py       # State update logic
├── env.py               # CenturySpiceRoad environment class
├── specs.py             # Observation/action specifications
├── rewards.py           # Reward computation
├── termination.py       # Game end detection
├── render.py            # Text-based visualization
├── registration.py      # Jumanji registration
├── agents/
│   └── random_agent.py  # Baseline random policy
└── tests/               # Test suite
```

## License

MIT
