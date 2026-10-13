#!/usr/bin/env bash
# Fetch new papers and write today's digest, then check system health. Run once a day from the repo root.
# Keep this single-process: the arXiv rate limit applies to all your requests combined.
# `paperscreen monitor` exits with 1 when it raises an alert, so cron mails you only when something is off.
set -euo pipefail
paperscreen refresh
paperscreen monitor --out reports/monitor-latest.md > /dev/null || {
  echo "paperscreen: monitoring raised alerts, see reports/monitor-latest.md" >&2
  exit 1
}
