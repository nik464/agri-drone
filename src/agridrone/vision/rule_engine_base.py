"""Rule engine protocol and registry (Step 7 of research-upgrade).

Design constraints
------------------
* **Do not rename or change the signature** of the existing rule engine in
  ``rule_engine.py``. It remains the default and keeps its public API.
* Add a ``Protocol`` that both the existing handcrafted engine and the new
  learned / LLM engines satisfy structurally, so the matrix runner can swap
  them with one string.

The existing ``rule_engine.evaluate`` function is already structurally
compatible; we wrap it here rather than touching it.
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class RuleEngine(Protocol):
    """Common contract for {handcrafted, learned_tree, llm_generated}.

    An engine is any callable that accepts:

    * ``features``        — the 20+ metric dict from ``feature_extractor``
    * ``classifier_result`` — top-k dict or ``None``
    * ``crop_type``       — ``"wheat"`` or ``"rice"`` (string)

    and returns a **result object** that exposes at minimum:

    * ``top_disease: str``
    * ``top_confidence: float``
    * ``candidates: list``  (opaque; must be JSON-serialisable via the
      engine's own ``result_to_dict`` helper)

    The handcrafted engine already satisfies this; see ``rule_engine.py``.
    """

    def __call__(
        self,
        features: Any,
        classifier_result: dict | None,
        crop_type: str = "wheat",
        **kwargs: Any,
    ) -> Any: ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_REGISTRY: dict[str, Callable[..., Any]] = {}


def register(name: str, fn: Callable[..., Any]) -> None:
    if name in _REGISTRY:
        raise ValueError(f"rule engine already registered: {name}")
    _REGISTRY[name] = fn


def get(name: str) -> Callable[..., Any]:
    if name == "none":
        return _no_op
    if name not in _REGISTRY:
        raise KeyError(
            f"unknown rule engine: {name!r} "
            f"(known: {sorted(_REGISTRY)} + 'none')"
        )
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(["none", *_REGISTRY.keys()])


def _no_op(features, classifier_result, crop_type="wheat", **_):
    """The ``rule_engine == 'none'`` baseline used in Config A.

    Returns ``None`` so callers can short-circuit without needing to build a
    synthetic result. The matrix runner treats this as "use the classifier
    top-1 directly".
    """
    return None


# ---------------------------------------------------------------------------
# Default registrations (lazy — avoids heavy imports at module load time)
# ---------------------------------------------------------------------------

def _register_defaults() -> None:
    try:
        from agridrone.vision.rule_engine import evaluate as _handcrafted

        if "handcrafted" not in _REGISTRY:
            register("handcrafted", _handcrafted)
    except Exception:
        # Keep registration best-effort; tests + dry-runs must not crash
        # just because torch is missing.
        pass

    try:
        from agridrone.vision.rules_learned import evaluate_learned

        if "learned_tree" not in _REGISTRY:
            register("learned_tree", evaluate_learned)
    except Exception:
        pass

    try:
        from agridrone.vision.rules_llm import evaluate_llm

        if "llm_generated" not in _REGISTRY:
            register("llm_generated", evaluate_llm)
    except Exception:
        pass


_register_defaults()
