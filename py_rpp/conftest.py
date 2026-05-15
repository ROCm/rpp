import sys
import pathlib

# Ensure the CMake build directory is first in sys.path so that
# `import py_rpp` resolves to build/py_rpp/ (with the compiled .so)
# rather than the bare source py_rpp/ directory.
BUILD_DIR = pathlib.Path(__file__).parent / "build"
if BUILD_DIR.exists():
    sys.path.insert(0, str(BUILD_DIR))
