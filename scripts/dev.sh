#!/usr/bin/env sh
set -eu

# Compatibility wrapper for macOS/Linux. The Python task owns the shared
# cross-platform path and process handling used on Windows as well.
exec "${PYTHON:-python3}" scripts/beatforge.py dev
