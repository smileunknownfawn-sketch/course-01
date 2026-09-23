"""Streamlit Cloud entry point.

Streamlit reruns this file after every widget interaction.  A plain import would
execute src.dashboard.app only on the first run because Python caches imported
modules, leaving later reruns blank.  Reload the dashboard module when it is
already present so the UI is rendered on every interaction.
"""

from __future__ import annotations

import importlib
import sys

MODULE_NAME = "src.dashboard.app"

if MODULE_NAME in sys.modules:
    importlib.reload(sys.modules[MODULE_NAME])
else:
    importlib.import_module(MODULE_NAME)
