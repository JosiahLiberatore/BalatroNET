"""Masked-lane primitives. No project imports — sits at the bottom of the graph."""

import jax.numpy as jnp


def first_n(mask, n):
    """Keep only the first n True lanes, in lane order. Caps on RANK, not index."""
    return mask & (jnp.cumsum(mask) - 1 < n)


def compact(x, keep):
    """Move the keep lanes to the front, order preserved. Stable — nothing dropped."""
    return x[jnp.argsort(~keep)]


def selected_cards(hand, hand_valid, select, k=5):
    """Agent's multi-hot -> (cards[k], valid[k], effective sel in LANE order)."""
    sel = first_n(select & hand_valid, k)
    return compact(hand, sel)[:k], compact(sel, sel)[:k], sel
