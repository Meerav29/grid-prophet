# driver_rounds.csv validation report

Total rows: 8013
Seasons covered: [np.int64(2018), np.int64(2019), np.int64(2020), np.int64(2021), np.int64(2022), np.int64(2023), np.int64(2024), np.int64(2025), np.int64(2026)]
Session type counts:
session_type
R    3744
Q    3701
S     568

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
2026    13

## DNF status mapping

Unclassified rows with no dnf_cause (should be 0 or explainable): 0
dnf_cause distribution:
dnf_cause
NaN           3663
other          321
incident       166
mechanical     162

Rows whose status is flagged needs_review in dnf_status_map.csv: 266 (6.2% of race/sprint rows)
status
Retired       232
Suspension     10
Wheel           7
Puncture        6
Mechanical      3
Undertray       3
Vibrations      2
Tyre            1
Rear wing       1
Front wing      1

## Clean-air lap filter validation

### fastest clean-air lap (primary) -- all sessions (n=3443)
Pearson r = 0.175 (p=5.98e-25), Spearman r = 0.518 (p=1.77e-235)
Dry + classified only (n=2663): Pearson r = 0.161, Spearman r = 0.544
Target from spec: ~0.8+ -- BELOW TARGET

### median clean-air lap (diagnostic) -- all sessions (n=3443)
Pearson r = 0.181 (p=9.23e-27), Spearman r = 0.583 (p=0.00e+00)
Dry + classified only (n=2663): Pearson r = 0.178, Spearman r = 0.629
Target from spec: ~0.8+ -- BELOW TARGET

Top-3 clean-air pace vs actual podium overlap: 305/549 driver-slots (55.6%)

## Circuit type coverage

circuit_type
high_downforce     2
high_speed        11
mixed             17
street             7
