"""Tests for exchange card execution (YYYY->GB, G->RR, etc).

Exchange semantics: playing the card auto-applies the first exchange.
AGAIN applies it again, DONE exits without applying.
"""

import jax
import jax.numpy as jnp
import pytest

from century_env import CenturySpiceRoad, Phase
from century_env.cards import ALL_TRADER_CARDS
from century_env.constants import CARD_TYPE_EXCHANGE


# ---------------------------------------------------------------------------
# Card data (from cards.py)
# ---------------------------------------------------------------------------
# Card 33: YYYY -> GB  [2, 0, 4, 0, 0, 0, 0, 0, 1, 1]
# Card 22: G -> RR     [2, 0, 0, 0, 1, 0, 0, 2, 0, 0]

CARD_YYYY_GB = ALL_TRADER_CARDS[33]
CARD_G_RR = ALL_TRADER_CARDS[22]


def _make_action(action_type=0, card_idx=0, market_pos=0,
                 scoring_idx=0, spice_type=0, continue_flag=1):
    return jnp.array([action_type, card_idx, market_pos,
                      scoring_idx, spice_type, continue_flag], dtype=jnp.int32)


def _setup_exchange_state(env, card, caravan):
    """Create a state where player 0 has the given exchange card at index 0
    and the specified caravan, ready to play."""
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)

    # Put exchange card in player 0's hand at index 0
    new_hands = state.hands.at[0, 0].set(card)
    # Set player 0's caravan
    new_caravans = state.caravans.at[0].set(jnp.array(caravan, dtype=jnp.int32))

    state = state.replace(
        hands=new_hands,
        caravans=new_caravans,
        phase=jnp.int32(Phase.CHOOSE_ACTION),
        current_player=jnp.int32(0),
    )
    return state


@pytest.fixture(scope="module")
def env():
    return CenturySpiceRoad(num_players=4)


class TestYYYYtoGB:
    """Tests for the YYYY -> GB exchange card."""

    def test_single_use_done(self, env):
        """Play YYYY->GB once with DONE: 4Y -> 0Y 1G 1B."""
        state = _setup_exchange_state(env, CARD_YYYY_GB, [4, 0, 0, 0])

        # Play card 0 — auto-applies first exchange
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        assert int(state.phase) == Phase.EXECUTE_CARD
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 1, 1], f"After play: expected [0,0,1,1], got {caravan}"

        # DONE (just exit)
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 1, 1], f"After DONE: expected [0,0,1,1], got {caravan}"
        assert int(state.phase) == Phase.CHOOSE_ACTION

    def test_again_then_done(self, env):
        """Play YYYY->GB with AGAIN then DONE: 8Y -> 4Y,1G,1B -> 0Y,2G,2B."""
        state = _setup_exchange_state(env, CARD_YYYY_GB, [8, 0, 0, 0])

        # Play card 0 — auto-applies first exchange
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        assert int(state.phase) == Phase.EXECUTE_CARD
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [4, 0, 1, 1], f"After play: expected [4,0,1,1], got {caravan}"

        # AGAIN (second exchange: 4Y,1G,1B -> 0Y,2G,2B)
        state, ts = env.step(state, _make_action(continue_flag=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 2, 2], f"After AGAIN: expected [0,0,2,2], got {caravan}"
        assert int(state.phase) == Phase.EXECUTE_CARD

        # DONE (just exit)
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 2, 2], f"After DONE: expected [0,0,2,2], got {caravan}"

    def test_cannot_afford_second_use(self, env):
        """Play YYYY->GB with exactly 4Y: auto-applies once, then DONE exits."""
        state = _setup_exchange_state(env, CARD_YYYY_GB, [4, 0, 0, 0])

        # Play card 0 — auto-applies: 4Y -> 0Y 1G 1B
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 1, 1], f"After play: expected [0,0,1,1], got {caravan}"

        # DONE — can't afford another, just exit
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 0, 1, 1], (
            f"DONE should not apply exchange. "
            f"Expected [0,0,1,1], got {caravan}"
        )
        assert all(c >= 0 for c in caravan), f"Negative spices! {caravan}"


class TestGtoRR:
    """Tests for the G -> RR exchange card."""

    def test_single_use_done(self, env):
        """Play G->RR once with DONE: 1G -> 2R."""
        state = _setup_exchange_state(env, CARD_G_RR, [0, 0, 1, 0])

        # Play card 0 — auto-applies first exchange
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        assert int(state.phase) == Phase.EXECUTE_CARD
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 2, 0, 0], f"After play: expected [0,2,0,0], got {caravan}"

        # DONE (just exit)
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 2, 0, 0], f"After DONE: expected [0,2,0,0], got {caravan}"

    def test_triple_use(self, env):
        """Play G->RR three times: 3G -> 2G,2R -> 1G,4R -> 6R."""
        state = _setup_exchange_state(env, CARD_G_RR, [0, 0, 3, 0])

        # Play card 0 — auto-applies first exchange
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 2, 2, 0], f"After play: expected [0,2,2,0], got {caravan}"

        # AGAIN (2nd use)
        state, ts = env.step(state, _make_action(continue_flag=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 4, 1, 0], f"After 2nd: expected [0,4,1,0], got {caravan}"

        # AGAIN (3rd use)
        state, ts = env.step(state, _make_action(continue_flag=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 6, 0, 0], f"After 3rd: expected [0,6,0,0], got {caravan}"

        # DONE (just exit)
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 6, 0, 0], f"After DONE: expected [0,6,0,0], got {caravan}"

    def test_cannot_afford_after_play(self, env):
        """Play G->RR with 1G: auto-applies, then DONE (can't afford another)."""
        state = _setup_exchange_state(env, CARD_G_RR, [0, 0, 1, 0])

        # Play card 0 — auto-applies: 1G -> 2R
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 2, 0, 0], f"After play: expected [0,2,0,0], got {caravan}"

        # DONE — must NOT apply again (no G left)
        state, ts = env.step(state, _make_action(continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [0, 2, 0, 0], (
            f"DONE should not apply exchange when unaffordable. "
            f"Expected [0,2,0,0], got {caravan}"
        )
        assert all(c >= 0 for c in caravan), f"Negative spices! {caravan}"


class TestConversionExtraUpgrade:
    """Test that conversion cards don't apply extra upgrades on DONE."""

    def test_upgrade_x2_no_extra(self, env):
        """Upgrade x2 with AGAIN+AGAIN should apply exactly 2, DONE should not add a 3rd."""
        # Card 1: Upgrade x2 [1, 2, 0, 0, 0, 0, 0, 0, 0, 0]
        upgrade_card = ALL_TRADER_CARDS[1]
        state = _setup_exchange_state(env, upgrade_card, [3, 0, 0, 0])

        # Play card 0
        state, ts = env.step(state, _make_action(action_type=0, card_idx=0))
        assert int(state.phase) == Phase.EXECUTE_CARD

        # AGAIN: upgrade Y->R (3Y -> 2Y 1R), remaining=1
        state, ts = env.step(state, _make_action(spice_type=0, continue_flag=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [2, 1, 0, 0], f"After 1st upgrade: {caravan}"

        # AGAIN: upgrade Y->R (2Y 1R -> 1Y 2R), remaining=0
        state, ts = env.step(state, _make_action(spice_type=0, continue_flag=0))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [1, 2, 0, 0], f"After 2nd upgrade: {caravan}"

        # DONE — should NOT apply a 3rd upgrade
        state, ts = env.step(state, _make_action(spice_type=0, continue_flag=1))
        caravan = [int(state.caravans[0][i]) for i in range(4)]
        assert caravan == [1, 2, 0, 0], (
            f"DONE should not apply extra upgrade when remaining=0. "
            f"Expected [1,2,0,0], got {caravan}"
        )
