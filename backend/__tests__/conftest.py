import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
BACKEND_ROOT = TESTS_ROOT.parent
REPO_ROOT = BACKEND_ROOT.parent
SHARED_ROOT = BACKEND_ROOT / "shared"

for path in (str(SHARED_ROOT), str(BACKEND_ROOT), str(REPO_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
