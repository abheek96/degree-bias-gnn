#!/usr/bin/env bash
# cleanup_results.sh — Remove entries in results/ older than N days.
#
# Usage:
#   ./cleanup_results.sh              # dry run, 7 days, ./results
#   ./cleanup_results.sh --delete
#   ./cleanup_results.sh --days 14 --delete
#   ./cleanup_results.sh --results-dir /path/to/results --days 3 --delete

set -euo pipefail

RESULTS_DIR="./results"
DAYS=7
DELETE=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --results-dir) RESULTS_DIR="$2"; shift 2 ;;
        --days)        DAYS="$2";        shift 2 ;;
        --delete)      DELETE=true;      shift   ;;
        *) echo "Unknown argument: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d "$RESULTS_DIR" ]]; then
    echo "Directory not found: $RESULTS_DIR" >&2
    exit 1
fi

MODE=$( $DELETE && echo "DELETE" || echo "DRY RUN" )
echo "[$MODE] Scanning $RESULTS_DIR — entries older than $DAYS day(s)"
echo ""

COUNT=0
while IFS= read -r -d '' entry; do
    COUNT=$((COUNT + 1))
    if $DELETE; then
        rm -rf "$entry"
        echo "  deleted: $entry"
    else
        echo "  would delete: $entry"
    fi
done < <(find "$RESULTS_DIR" -maxdepth 1 -mindepth 1 -mtime "+$((DAYS - 1))" -print0)

echo ""
if [[ $COUNT -eq 0 ]]; then
    echo "Nothing to remove."
elif $DELETE; then
    echo "Deleted $COUNT item(s)."
else
    echo "Would delete $COUNT item(s). Re-run with --delete to remove them."
fi
