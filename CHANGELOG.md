# Changelog

## v0 — Original obstacle detection (`versions/v0_original_pre_imu.py`)

Baseline script: YOLOv4-tiny + stereo depth fusion, dual-system fallback
(named object vs. generic depth-based obstacle), persistence filtering,
speak cooldown, temperature safety cutoff, auto-reconnect loop.

## v1 — First IMU integration (`versions/v1_first_imu_integration.py`)

Added the OAK-D Lite's onboard IMU to compensate for head tilt when
classifying an object's height, so a wearer looking down at the floor
doesn't cause a head-height object to be misread as floor-level (and vice
versa).

- Added an IMU pipeline node (accelerometer + gyroscope)
- `compute_pitch_from_accel()` — pitch from raw accelerometer via `atan2`
- Exponential smoothing of pitch (accelerometer-only at this stage)
- Startup calibration step: captures a neutral pitch baseline while the
  wearer holds still and looks straight ahead
- `classify_height_band()` — rotates a detection's camera-frame vertical
  offset (`spatialCoordinates.y/z`, from named/YOLO detections only at this
  stage) into a gravity-aligned frame using the current pitch, then buckets
  it into floor / torso / head
- Messages updated to append a height phrase (e.g. "near the floor", "at
  head height") when available
- Added `DEBUG_IMU_AXES` flag to print raw accel values, used to verify
  which physical axis corresponds to pitch on the actual mount

**Testing found**: raw accelerometer nod/shake data confirmed the axis
choice (ay/az for pitch, ax excluded as the roll/side axis) was correct.

## v2 — Pitch sign verification

Added `DEBUG_PITCH` to print smoothed pitch once a second. Testing on
hardware confirmed the sign convention (positive = looking down, negative =
looking up) was correct as implemented — no change needed to the
accelerometer-based pitch calculation itself.

## v3 — Gyro complementary filter

Accelerometer-only pitch picks up motion acceleration on top of gravity
during fast head movement, causing inaccurate spikes. Added:

- Gyroscope reading (`compute_gyro_pitch_rate()`) fused with the
  accelerometer estimate via a complementary filter (98% gyro / 2% accel
  weighting)
- `DEBUG_HEIGHT_BANDS` flag: prints label, classified band, pitch, and raw
  spatial coordinates for every close detection — used for all height-band
  validation from this point on

**Testing found**: the gyro sign was opposite the initial right-hand-rule
assumption — pitch flipped direction entirely once the gyro was added.
Fixed by negating the gyro rate term.

**Testing found (after the sign fix)**: pitch was jumping erratically
between consecutive readings even beyond what fast head motion should
cause. Root cause: `dt` for the gyro integration was computed from
wall-clock processing time, which is inaccurate when IMU packets arrive
batched — several packets can be processed in a near-instant loop even
though the underlying samples span real time. Fixed by using each packet's
own device timestamp for `dt` instead.

## v4 — Floor-level obstacle detection for unclassified objects

Most real trip hazards (dropped items, cords, curbs, clutter) aren't in
COCO's 80 classes, so YOLO alone misses them. The original raw-depth
fallback also only scanned the middle third of the frame vertically
(torso-height range), meaning it never looked at anything low in frame.

- Generalized `get_zone_distance_depth()` to accept a configurable vertical
  scan band instead of a hardcoded one
- Added a second scan band (`FLOOR_SCAN_ROW_START/END`, bottom quarter of
  frame) alongside the original (`GENERAL_SCAN_ROW_START/END`, middle
  third); each zone now checks both and keeps whichever is closer/more
  urgent
- Added `estimate_band_from_pixel_row()` — since raw-depth detections don't
  have `spatialCoordinates`, this derives an approximate camera-frame Y
  offset from the scanned region's position in-frame (using a
  `VERTICAL_FOV_DEG` estimate) and reuses the same verified rotation math as
  `classify_height_band()`
- Unified message construction (`build_message()`) so unclassified floor
  hazards get distinct phrasing ("Low obstacle, very close, watch your
  step") instead of the generic caution message
- Fixed a real bug found during this testing pass: detections with
  `spatialCoordinates.z == 0` (depthai's "no valid depth" signal) were being
  treated as extremely close obstacles instead of being filtered out —
  added `has_valid_spatial_coords()` to reject them

## v5 — Gyro bias correction (current)

Extended testing (several minutes of continuous use) revealed the pitch
estimate would drift steadily upward over time and fail to recover even
while the wearer was stationary — climbing past 100° and plateauing rather
than returning to baseline. Root cause: every real gyroscope has a small
non-zero reading at rest (zero-rate bias); left unsubtracted, the
complementary filter's heavy gyro weighting (98%) integrates that bias
continuously, and the 2% accelerometer correction isn't enough to counter
it over minutes of use.

- `calibrate()` now measures the gyro's at-rest bias (`gyro_bias_x`) during
  the same stillness window used for the pitch baseline
- `compute_gyro_pitch_rate()` subtracts this bias before use

**Known limitation**: gyro bias isn't perfectly constant — it drifts slowly
with temperature and time. This fix addresses the acute runaway-drift
failure mode observed in testing, but very long sessions may still benefit
from periodic recalibration. Not yet implemented.
