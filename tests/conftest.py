import sys
import os
import importlib.util

_src_path = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, _src_path)

# Load src/__main__.py into sys.modules so `import __main__` in tests picks it up.
# We load it under a temporary name first to avoid triggering the
# `if __name__ == "__main__": main()` guard, then alias it into sys.modules.
_spec = importlib.util.spec_from_file_location(
    "_grid_prophet_main", os.path.join(_src_path, "__main__.py")
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
sys.modules["__main__"] = _mod
