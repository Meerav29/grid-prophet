#!/usr/bin/env bash
# Race-weekend forecast: refresh data, fit, post-quali forecast, publish CSV.
# Usage: scripts/weekend_forecast.sh <season> <round>   (run after Saturday quali)
# Needs data/driver_rounds.csv (cloud: `git checkout origin/forecast-data -- data/driver_rounds.csv data/driver_meta.csv`).
set -euo pipefail
SEASON=${1:?season} ROUND=${2:?round}
PY=${PYTHON:-python3.13}
export PYTHONPATH=src
PREV=$((ROUND - 1))

$PY -m collect_v2 --start-year "$SEASON" --end-year "$SEASON"   # resumable; fetches new rounds incl. this quali
$PY -m cli fit --through "$SEASON:$PREV"
$PY -m cli race --season "$SEASON" --round "$ROUND" --mode post-quali --grid-from-quali

OUT=$(cat outputs/latest.txt)
cp "$OUT/race_forecast_${SEASON}_${ROUND}.csv" "forecasts/"
echo "Forecast copied to forecasts/race_forecast_${SEASON}_${ROUND}.csv; commit and push it."
