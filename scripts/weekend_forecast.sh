#!/usr/bin/env bash
# Race-weekend forecast: refresh data, fit, post-quali forecast, publish CSV.
#
# Usage:
#   scripts/weekend_forecast.sh <season> <round>              # Saturday, after quali: grid = quali order
#   scripts/weekend_forecast.sh <season> <round> <grid.csv>   # Sunday, after the FIA posts the grid
#
# The grid file has columns abbreviation,grid and must list every entrant once
# with slots 1..n (penalties and pit-lane starts already applied). Don't build
# it from race rows: a pit-lane start read back as P22 made R19 worse.
#
# Needs data/driver_rounds.csv (cloud: `git checkout origin/forecast-data -- data/driver_rounds.csv data/driver_meta.csv`).
set -euo pipefail
SEASON=${1:?season} ROUND=${2:?round} GRID_FILE=${3:-}
PY=${PYTHON:-python3.13}
export PYTHONPATH=src
PREV=$((ROUND - 1))

$PY -m collect_v2 --start-year "$SEASON" --end-year "$SEASON"   # resumable; fetches new rounds incl. this quali
$PY -m cli fit --through "$SEASON:$PREV"

if [[ -n "$GRID_FILE" ]]; then
  $PY -m cli race --season "$SEASON" --round "$ROUND" --mode post-quali --grid-file "$GRID_FILE"
  SUFFIX=_grid
else
  $PY -m cli race --season "$SEASON" --round "$ROUND" --mode post-quali --grid-from-quali
  SUFFIX=
fi

OUT=$(cat outputs/latest.txt)
mkdir -p forecasts
DEST="forecasts/race_forecast_${SEASON}_${ROUND}${SUFFIX}.csv"
cp "$OUT/race_forecast_${SEASON}_${ROUND}.csv" "$DEST"
echo "Forecast copied to $DEST; commit and push it."
