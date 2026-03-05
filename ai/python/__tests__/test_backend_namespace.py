import sys
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = APP_ROOT.parent

for path in (str(APP_ROOT), str(PROJECT_ROOT)):
    if path not in sys.path:
        sys.path.append(path)


def test_backend_namespace_exports_core_aliases():
    import backend.agents as agents
    import backend.ml as ml
    import backend.server as server_ns

    assert callable(agents.coordinator_agent)
    assert callable(ml.load_ml_dataset)
    assert server_ns.app is not None


def test_backend_module_aliases_import_without_errors():
    import backend.agents.coordinator  # noqa: F401
    import backend.agents.data_scientist  # noqa: F401
    import backend.agents.langsmith  # noqa: F401
    import backend.ml.data  # noqa: F401
    import backend.server.http  # noqa: F401
    import backend.server.ml  # noqa: F401
