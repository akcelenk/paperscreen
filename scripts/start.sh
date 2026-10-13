#!/usr/bin/env bash
# Double-click launcher (macOS: rename to start.command). Activates the venv and opens the app.
cd "$(dirname "$0")/.."
source .venv/bin/activate
paperscreen app
