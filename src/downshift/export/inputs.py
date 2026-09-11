"""Dummy input synthesis (IMPLEMENTATION_PLAN.md §5.2).

Only Tier 1 (user-supplied) and Tier 5 (fail loudly) exist today:

- Tier 1 — user-supplied example inputs. Always honored, always wins.
- Tier 2 (adapter-derived) and Tier 3 (signature introspection) need model-family-specific
  knowledge — HF's OnnxConfig, PyG's in_channels, forward() annotations — that doesn't exist
  until the hf/pyg adapters land. Faking a heuristic now would produce confidently wrong
  shapes for models we haven't actually built support for; better to fail loudly than guess.
- Tier 4 (interactive wizard) is a CLI concern, deferred to v0.3.
"""

from typing import Any


def synthesize(user_inputs: tuple[Any, ...] | None) -> tuple[Any, ...]:
    if user_inputs is not None:
        return user_inputs
    raise NotImplementedError(
        "No example_inputs given, and automatic input synthesis (adapter-derived or "
        "signature-introspection) isn't implemented yet. Pass example_inputs explicitly."
    )
