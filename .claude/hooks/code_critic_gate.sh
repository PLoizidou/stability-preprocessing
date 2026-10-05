#!/usr/bin/env bash
# Stop hook: gates on whether "major" .py/.ipynb changes were produced since the
# last time this fired, across stability-preprocessing and stability-analysis.
# If so, blocks the stop and asks Claude to run the code-critic subagent first.
set -euo pipefail

REPOS=(
  "/home/toor/Desktop/stability-preprocessing"
  "/home/toor/Desktop/stability-analysis"
)
MIN_CHANGED_LINES=5
STATE_DIR="$HOME/.claude/state/code-critic-gate"
mkdir -p "$STATE_DIR"

changed_repos=()

for repo in "${REPOS[@]}"; do
  [ -d "$repo/.git" ] || continue

  diff_content="$(git -C "$repo" diff -- '*.py' '*.ipynb'; git -C "$repo" diff --cached -- '*.py' '*.ipynb')"
  untracked="$(git -C "$repo" status --porcelain -- '*.py' '*.ipynb' | grep '^??' || true)"
  changed_lines="$(printf '%s\n' "$diff_content" | grep -c '^[+-][^+-]' || true)"

  state_file="$STATE_DIR/$(basename "$repo").hash"
  current_hash="$( { printf '%s' "$diff_content"; printf '%s' "$untracked"; } | sha256sum | cut -d' ' -f1)"
  prev_hash=""
  [ -f "$state_file" ] && prev_hash="$(cat "$state_file")"

  if [ "$current_hash" != "$prev_hash" ] && { [ "${changed_lines:-0}" -ge "$MIN_CHANGED_LINES" ] || [ -n "$untracked" ]; }; then
    changed_repos+=("$repo")
  fi
  printf '%s' "$current_hash" > "$state_file"
done

if [ "${#changed_repos[@]}" -gt 0 ]; then
  repo_list="$(printf '%s, ' "${changed_repos[@]}")"
  repo_list="${repo_list%, }"
  reason="Meaningful .py/.ipynb changes were just produced in: ${repo_list}. Before finishing this turn, invoke the 'code-critic' subagent (Agent tool, subagent_type: code-critic) to review these changes for workflow violations, coding mistakes, and scientific rigor issues per CLAUDE.md. Then act on or report any warranted findings."
  python3 -c "import json,sys; print(json.dumps({'decision': 'block', 'reason': sys.argv[1]}))" "$reason"
else
  echo '{}'
fi
