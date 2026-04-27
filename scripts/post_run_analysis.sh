#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/post_run_analysis.sh
#   bash scripts/post_run_analysis.sh /absolute/path/to/batch_dir
#
# If no batch dir is supplied, this script picks the latest folder under:
#   results/qwen3_full_factorial/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ $# -ge 1 ]]; then
  BATCH_DIR="$1"
else
  LATEST="$(ls -td results/qwen3_full_factorial/*/ 2>/dev/null | head -n 1 || true)"
  if [[ -z "${LATEST}" ]]; then
    echo "No batch folder found under results/qwen3_full_factorial/"
    exit 1
  fi
  BATCH_DIR="${LATEST%/}"
fi

if [[ ! -d "$BATCH_DIR" ]]; then
  echo "Batch directory not found: $BATCH_DIR"
  exit 1
fi

if [[ ! -f "$BATCH_DIR/progress.json" ]]; then
  echo "Missing progress.json in $BATCH_DIR"
  exit 1
fi

echo "Using batch dir: $BATCH_DIR"

# 1) Aggregate factorial-cell metrics and write report files.
python -m gshp.cli analyze-batch "$BATCH_DIR"

# 1.5) Paper-focused contrasts and interpretation report.
python scripts/paper_tables.py "$BATCH_DIR"

# 2) Print compact completion stats from progress.json.
python - "$BATCH_DIR" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]) / "progress.json"
data = json.loads(p.read_text())
total = int(data.get("total_planned", 0))
completed = len(data.get("completed", []))
failed = len(data.get("failed", []))
skipped = len(data.get("skipped", []))
done = completed + failed + skipped
print(f"Progress: {done}/{total} (completed={completed}, failed={failed}, skipped={skipped})")
if done < total:
    print("Warning: batch appears incomplete. Aggregates reflect current partial results.")
PY

echo ""
echo "Wrote:"
echo "  - $BATCH_DIR/batch_analysis.json"
echo "  - $BATCH_DIR/condition_summary.csv"
echo "  - $BATCH_DIR/report.md"
echo "  - $BATCH_DIR/paper_tables.json"
echo "  - $BATCH_DIR/paper_report.md"
echo ""
echo "Next:"
echo "  - Open report: $BATCH_DIR/report.md"
echo "  - Share condition_summary.csv for quick table/plotting"
