"""
Century: Spice Road - Observation and Action Specifications

Defines the specs for the Jumanji environment interface.
"""

import jax.numpy as jnp
from jumanji import specs
from jumanji.types import StepType

from century_env.constants import (
    MAX_PLAYER_CARDS,
    MAX_SCORED_CARDS,
    NUM_MARKET_SLOTS,
    NUM_SCORING_SLOTS,
    NUM_SPICE_TYPES,
    NUM_DECK_TRADER_CARDS,
    NUM_SCORING_CARDS,
    MAX_PLAYERS,
    NUM_ACTION_TYPES,
    NUM_CONTINUE_FLAGS,
)
from century_env.types import Observation


def observation_spec() -> specs.Spec:
    """Return the observation specification.

    Returns nested spec matching Observation dataclass structure.
    """
    return specs.Spec(
        Observation,
        "ObservationSpec",
        # My state
        my_hand=specs.BoundedArray((MAX_PLAYER_CARDS, 10), jnp.int32, 0, 10,
                                    "my_hand"),
        my_hand_size=specs.BoundedArray((), jnp.int32, 0, MAX_PLAYER_CARDS,
                                         "my_hand_size"),
        my_played=specs.BoundedArray((MAX_PLAYER_CARDS, 10), jnp.int32, 0, 10,
                                      "my_played"),
        my_played_size=specs.BoundedArray((), jnp.int32, 0, MAX_PLAYER_CARDS,
                                           "my_played_size"),
        my_caravan=specs.BoundedArray((NUM_SPICE_TYPES,), jnp.int32, 0, 20,
                                       "my_caravan"),
        my_scored=specs.BoundedArray((MAX_SCORED_CARDS, 5), jnp.int32, 0, 20,
                                      "my_scored"),
        my_scored_count=specs.BoundedArray((), jnp.int32, 0, MAX_SCORED_CARDS,
                                            "my_scored_count"),
        my_gold=specs.BoundedArray((), jnp.int32, 0, 20, "my_gold"),
        my_silver=specs.BoundedArray((), jnp.int32, 0, 20, "my_silver"),

        # Opponent state (4 opponents max)
        opp_active=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.bool_, False, True,
                                       "opp_active"),
        opp_hand_sizes=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.int32, 0,
                                           MAX_PLAYER_CARDS, "opp_hand_sizes"),
        opp_played=specs.BoundedArray((MAX_PLAYERS - 1, MAX_PLAYER_CARDS, 10),
                                       jnp.int32, 0, 10, "opp_played"),
        opp_played_sizes=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.int32, 0,
                                             MAX_PLAYER_CARDS, "opp_played_sizes"),
        opp_caravans=specs.BoundedArray((MAX_PLAYERS - 1, NUM_SPICE_TYPES),
                                         jnp.int32, 0, 20, "opp_caravans"),
        opp_scored=specs.BoundedArray((MAX_PLAYERS - 1, MAX_SCORED_CARDS, 5),
                                       jnp.int32, 0, 20, "opp_scored"),
        opp_scored_counts=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.int32, 0,
                                              MAX_SCORED_CARDS, "opp_scored_counts"),
        opp_gold=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.int32, 0, 20,
                                     "opp_gold"),
        opp_silver=specs.BoundedArray((MAX_PLAYERS - 1,), jnp.int32, 0, 20,
                                       "opp_silver"),

        # Market state
        market_cards=specs.BoundedArray((NUM_MARKET_SLOTS, 10), jnp.int32, 0, 10,
                                         "market_cards"),
        market_spices=specs.BoundedArray((NUM_MARKET_SLOTS, NUM_SPICE_TYPES),
                                          jnp.int32, 0, 20, "market_spices"),
        market_size=specs.BoundedArray((), jnp.int32, 0, NUM_MARKET_SLOTS,
                                        "market_size"),
        scoring_row=specs.BoundedArray((NUM_SCORING_SLOTS, 5), jnp.int32, 0, 20,
                                        "scoring_row"),
        scoring_row_size=specs.BoundedArray((), jnp.int32, 0, NUM_SCORING_SLOTS,
                                             "scoring_row_size"),

        # Supply info
        gold_remaining=specs.BoundedArray((), jnp.int32, 0, 20, "gold_remaining"),
        silver_remaining=specs.BoundedArray((), jnp.int32, 0, 20, "silver_remaining"),
        trader_deck_size=specs.BoundedArray((), jnp.int32, 0, NUM_DECK_TRADER_CARDS,
                                             "trader_deck_size"),
        scoring_deck_size=specs.BoundedArray((), jnp.int32, 0, NUM_SCORING_CARDS,
                                              "scoring_deck_size"),

        # Game info
        num_players=specs.BoundedArray((), jnp.int32, 2, MAX_PLAYERS, "num_players"),
        current_phase=specs.BoundedArray((), jnp.int32, 0, 6, "current_phase"),

        # Action state
        selected_card=specs.BoundedArray((10,), jnp.int32, 0, 10, "selected_card"),
        remaining_upgrades=specs.BoundedArray((), jnp.int32, 0, 10,
                                               "remaining_upgrades"),
        acquire_target_position=specs.BoundedArray((), jnp.int32, 0, NUM_MARKET_SLOTS,
                                                    "acquire_target_position"),
        spices_placed_count=specs.BoundedArray((), jnp.int32, 0, NUM_MARKET_SLOTS,
                                                "spices_placed_count"),

        # Action mask (tuple of 6 arrays)
        action_mask=specs.Spec(
            tuple,
            "ActionMaskSpec",
            specs.BoundedArray((NUM_ACTION_TYPES,), jnp.bool_, False, True,
                               "action_type_mask"),
            specs.BoundedArray((MAX_PLAYER_CARDS,), jnp.bool_, False, True,
                               "card_idx_mask"),
            specs.BoundedArray((NUM_MARKET_SLOTS,), jnp.bool_, False, True,
                               "market_pos_mask"),
            specs.BoundedArray((NUM_SCORING_SLOTS,), jnp.bool_, False, True,
                               "scoring_idx_mask"),
            specs.BoundedArray((NUM_SPICE_TYPES,), jnp.bool_, False, True,
                               "spice_type_mask"),
            specs.BoundedArray((NUM_CONTINUE_FLAGS,), jnp.bool_, False, True,
                               "continue_mask"),
        ),
    )


def action_spec() -> specs.MultiDiscreteArray:
    """Return the multi-discrete action specification.

    Action is a 6-element array:
        [action_type, card_idx, market_pos, scoring_idx, spice_type, continue_flag]

    Sizes:
        - action_type: 4 (Play/Acquire/Rest/Score)
        - card_idx: 25 (hand card selection)
        - market_pos: 6 (market position selection)
        - scoring_idx: 5 (scoring card selection)
        - spice_type: 4 (spice selection)
        - continue_flag: 2 (AGAIN=0, DONE=1)
    """
    num_values = jnp.array([
        NUM_ACTION_TYPES,   # 4
        MAX_PLAYER_CARDS,   # 25
        NUM_MARKET_SLOTS,   # 6
        NUM_SCORING_SLOTS,  # 5
        NUM_SPICE_TYPES,    # 4
        NUM_CONTINUE_FLAGS, # 2
    ], dtype=jnp.int32)

    return specs.MultiDiscreteArray(
        num_values=num_values,
        name="action",
        dtype=jnp.int32
    )


def discount_spec() -> specs.BoundedArray:
    """Return the discount specification."""
    return specs.BoundedArray(
        shape=(),
        dtype=jnp.float32,
        minimum=0.0,
        maximum=1.0,
        name="discount"
    )


def reward_spec() -> specs.Array:
    """Return the reward specification."""
    return specs.Array(
        shape=(),
        dtype=jnp.float32,
        name="reward"
    )
