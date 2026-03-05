import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)

from shared.agui_runtime import intents


def test_resolve_navigation_target_supports_aliases_and_routes():
    assert intents.resolve_navigation_target("chat") == "/"
    assert intents.resolve_navigation_target("go to /ml/pytorch") == "/ml/pytorch"


def test_resolve_navigation_target_handles_polite_prompt_and_empty_text():
    assert intents.resolve_navigation_target("please navigate to tensorflow") == "/ml/tensorflow"
    assert intents.resolve_navigation_target("   ") is None


def test_resolve_ag_ui_tab_target_trims_follow_on_phrase():
    assert intents.resolve_ag_ui_tab_target("please switch to pytorch tab and then train") == "pytorch"


def test_resolve_ag_ui_tab_target_handles_aliases_and_empty_text():
    assert intents.resolve_ag_ui_tab_target("tf") == "tensorflow"
    assert intents.resolve_ag_ui_tab_target("   ") is None
    assert intents.resolve_ag_ui_tab_target("unknown tab") is None


def test_is_pure_tab_switch_intent_rejects_chained_actions():
    assert intents.is_pure_tab_switch_intent("switch to tensorflow") is True
    assert intents.is_pure_tab_switch_intent("switch to tensorflow and train") is False
    assert intents.is_pure_tab_switch_intent("   ") is False
