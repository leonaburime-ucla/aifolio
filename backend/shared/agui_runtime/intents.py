from __future__ import annotations

"""Intent parsing helpers for AG-UI navigation and tab switching.

These utilities are pure functions used by AG-UI orchestration logic to map
natural language phrases into deterministic UI navigation targets.
"""

import re

NAV_ROUTE_ALIASES: dict[str, str] = {
    "/": "/",
    "home": "/",
    "ai chat": "/",
    "chat": "/",
    "ag-ui": "/ag-ui",
    "agui": "/ag-ui",
    "agentic research": "/agentic-research",
    "agentic-research": "/agentic-research",
    "pytorch": "/ml/pytorch",
    "tensorflow": "/ml/tensorflow",
    "knowledge distillation": "/ml/knowledge-distillation",
    "knowledge-distillation": "/ml/knowledge-distillation",
}

AG_UI_TAB_ALIASES: dict[str, str] = {
    "charts": "charts",
    "chart": "charts",
    "home": "charts",
    "base": "charts",
    "agentic research": "agentic-research",
    "agentic-research": "agentic-research",
    "research": "agentic-research",
    "ar": "agentic-research",
    "pytorch": "pytorch",
    "torch": "pytorch",
    "tensorflow": "tensorflow",
    "tf": "tensorflow",
}


def resolve_navigation_target(text: str) -> str | None:
    """Resolve a free-form navigation phrase to a known app route.

    Args:
        text: User-provided navigation text.

    Returns:
        Canonical route string (e.g. `/ml/pytorch`) or `None` when unresolved.
    """
    normalized = text.strip().lower()
    if not normalized:
        return None

    direct = NAV_ROUTE_ALIASES.get(normalized)
    if direct:
        return direct

    patterns = [
        r"(?:navigate|go|open|take me|switch)\s+(?:to\s+)?(.+)$",
        r"(?:can you|please)\s+(?:navigate|go|open)\s+(?:to\s+)?(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        target = match.group(1).strip(" .!?")
        resolved = NAV_ROUTE_ALIASES.get(target)
        if resolved:
            return resolved
        if target.startswith("/") and target in set(NAV_ROUTE_ALIASES.values()):
            return target
    return None


def resolve_ag_ui_tab_target(text: str) -> str | None:
    """Resolve a free-form tab-switch phrase to a known AG-UI tab identifier.

    Args:
        text: User-provided tab-switch text.

    Returns:
        Canonical tab key (e.g. `pytorch`, `agentic-research`) or `None`.
    """
    normalized = text.strip().lower()
    if not normalized:
        return None

    direct = AG_UI_TAB_ALIASES.get(normalized)
    if direct:
        return direct

    patterns = [
        r"(?:switch|change|open|go)\s+(?:to\s+)?(.+?)(?:\s+tab)?$",
        r"(?:can you|please)\s+(?:switch|change|open|go)\s+(?:to\s+)?(.+?)(?:\s+tab)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        target = match.group(1).strip(" .!?")
        target = re.split(r"\b(?:and|then|also|after)\b", target, maxsplit=1)[0].strip()
        target = re.sub(r"\b(tab|page|screen|workspace)\b$", "", target).strip()
        resolved = AG_UI_TAB_ALIASES.get(target)
        if resolved:
            return resolved
    return None


def is_pure_tab_switch_intent(text: str) -> bool:
    """Check whether text expresses only a tab-switch intent.

    Args:
        text: User-provided prompt text.

    Returns:
        `True` when the prompt is a tab switch without chained actions.
    """
    normalized = text.strip().lower()
    if not normalized:
        return False
    if resolve_ag_ui_tab_target(normalized) is None:
        return False
    return re.search(r"\b(and|then|also|after)\b", normalized) is None
