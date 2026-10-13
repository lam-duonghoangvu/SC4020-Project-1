#!/bin/sh
# PostToolUse hook: lint and format Python files Claude just changed.
f=$(jq -r '.tool_input.file_path // .tool_input.notebook_path // empty')
case "$f" in
  *.py|*.pyi|*.ipynb) ;;
  *) exit 0 ;;
esac

ruff check --fix --quiet "$f" >&2
status=$?
ruff format --quiet "$f" || true
# Exit 2 sends remaining lint errors back to Claude so it can fix them.
[ "$status" -eq 0 ] || exit 2
