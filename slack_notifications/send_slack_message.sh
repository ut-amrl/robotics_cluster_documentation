#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: SLACK_HOOK_URI=<url> ${0##*/} <message...>" >&2
  echo "  or: export SLACK_HOOK_URI='https://hooks.slack.com/services/…'" >&2
  echo 'Example: '"${0##*/}"' "Hello, world!"' >&2
  exit 1
}

if [[ -z "${SLACK_HOOK_URI:-}" ]]; then
  echo "Error: SLACK_HOOK_URI is not set." >&2
  echo "Set it to your Slack incoming webhook URL before running this script, for example:" >&2
  echo "  export SLACK_HOOK_URI='https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXX'" >&2
  echo "Or prefix the command once:" >&2
  echo "  SLACK_HOOK_URI='https://hooks.slack.com/services/…' ${0##*/} \"Your message\"" >&2
  exit 1
fi

if [[ $# -lt 1 ]]; then
  usage
fi

message=$*

if [[ -z "$message" ]]; then
  echo "Error: message is empty." >&2
  exit 1
fi

if [[ "$SLACK_HOOK_URI" != https://hooks.slack.com/services/* ]]; then
  echo "Warning: SLACK_HOOK_URI does not look like a Slack incoming webhook (expected https://hooks.slack.com/services/...)" >&2
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Error: python3 is required to build the JSON payload." >&2
  exit 1
fi

# Pass message via env so quotes, newlines, and Unicode do not break the shell.
payload=$(MESS="$message" python3 -c 'import json, os; print(json.dumps({"text": os.environ["MESS"]}))')

if ! curl -sS -f -X POST \
  -H 'Content-Type: application/json; charset=utf-8' \
  -d "$payload" \
  "$SLACK_HOOK_URI" >/dev/null; then
  echo "Error: failed to post to Slack (check URL and network)." >&2
  exit 1
fi
