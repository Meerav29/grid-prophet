# driver_rounds.csv validation report

Total rows: 8255
Seasons covered: [np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024), np.int64(2025), np.int64(2026)]
Session type counts:
session_type
Q    3833
R    3810
S     612

## Rounds collected per season

season
2018    21
2019    21
2020    17
2021    22
2022    22
2023    22
2024    24
2025    24
2026    17

## DNF status mapping

Unclassified rows with no dnf_cause (should be 0 or explainable): 0
dnf_cause distribution:
dnf_cause
NaN           3766
other          328
incident       166
mechanical     162

Rows whose status is flagged needs_review in dnf_status_map.csv: 295 (6.7% of race/sprint rows)
status
Retired       261
Suspension     10
Wheel           7
Puncture        6
Mechanical      3
Undertray       3
Vibrations      2
Tyre            1
Rear wing       1
Front wing      1

## Weather coverage (spec sec 5)

Race/sprint rows: wet 1090, dry 3210, not recorded 122 (2.8% unmeasured)
Rounds with at least one wet race/sprint session: 49
2018:1, 2018:3, 2018:5, 2018:6, 2018:8, 2018:11, 2018:14, 2018:16, 2018:21, 2019:6, 2019:11, 2019:16, 2020:2, 2020:3, 2020:11, 2020:14, 2021:2, 2021:9, 2021:11, 2021:12, 2021:15, 2021:20, 2022:5, 2022:7, 2022:10, 2022:13, 2022:17, 2022:18, 2023:4, 2023:6, 2023:12, 2023:13, 2023:20, 2024:9, 2024:10, 2024:11, 2024:12, 2024:21, 2025:1, 2025:6, 2025:10, 2025:12, 2025:13, 2025:18, 2025:21, 2026:4, 2026:12, 2026:16, 2026:17
Unrecorded weather is treated as dry by the pace likelihood, never as wet -- see model.weather.

## Clean-air lap filter validation

### fastest clean-air lap (primary) -- all sessions (n=3565)
Pearson r = 0.177 (p=2.25e-26), Spearman r = 0.529 (p=2.79e-256)
Dry + classified only (n=2702): Pearson r = 0.165, Spearman r = 0.546
Target from spec: ~0.8+ -- BELOW TARGET

### median clean-air lap (diagnostic) -- all sessions (n=3565)
Pearson r = 0.184 (p=1.34e-28), Spearman r = 0.589 (p=0.00e+00)
Dry + classified only (n=2702): Pearson r = 0.185, Spearman r = 0.630
Target from spec: ~0.8+ -- BELOW TARGET

Top-3 clean-air pace vs actual podium overlap: 313/561 driver-slots (55.8%)

## Circuit type coverage

circuit_type
high_downforce     2
high_speed        11
mixed             17
street             7
