# NavAid

A wearable navigation aid for elderly users, built on a Raspberry Pi 5 and a
Luxonis OAK-D Lite depth camera. It detects obstacles in the wearer's path —
including obstacles YOLO can't classify, and objects on the floor regardless
of the wearer's head angle — and announces spoken warnings through a
head-worn speaker (e.g. *"chair right in front of you"*, *"low obstacle, very
close, watch your step"*).

## How it works

- **Object detection + depth fusion**: YOLOv4-tiny (COCO, 80 classes) fused
  with stereo depth via depthai's `YoloSpatialDetectionNetwork`, giving each
  detection a real-world distance and position.
- **Dual-path obstacle detection**: named objects (chair, person, etc.) go
  through the YOLO path; anything YOLO can't classify — clutter, curbs,
  dropped items — is still caught by a raw-depth fallback that scans two
  vertical bands of the frame (general/torso-height and floor-height).
- **IMU-based height classification**: the OAK-D Lite's onboard BMI270 IMU is
  used to track head pitch, so an object's height (floor / torso / head) is
  computed relative to the *ground*, not relative to wherever the wearer's
  head happens to be pointing. Looking down at your feet doesn't make a
  shelf at head height misread as a floor hazard, and vice versa.
- **Sensor fusion for pitch**: accelerometer + gyroscope combined via a
  complementary filter, with a calibration step at startup that measures both
  a neutral pitch baseline and the gyro's at-rest bias (uncorrected gyro bias
  causes slow runaway drift over long sessions — see CHANGELOG).
- **Noise reduction**: persistence filtering (an obstacle must be seen for
  several consecutive frames before it's announced), a cooldown between
  spoken alerts, and filtering of invalid zero-depth detections.
- **Reliability**: runs as an auto-restarting systemd service, with an outer
  reconnect loop and a temperature safety cutoff.

## Hardware

- Raspberry Pi 5
- Luxonis OAK-D Lite depth camera (onboard BMI270 6-axis IMU), USB-C to the
  Pi's USB 3.0 port
- USB speaker (audio via ALSA/`aplay`)
- Planned: DRV2605L haptic motor + ERM vibration motor; bone conduction
  headset via a USB-to-3.5mm adapter
- Mount: GoPro-style head strap + camera clamp (in progress)

## Software

- Raspberry Pi OS (Debian-based), Python 3.13, venv-managed
- Key packages: `depthai==2.24.0.0`, `depthai-sdk`, `numpy`, `blobconverter`,
  `opencv-contrib-python`
- Runs as a systemd service for auto-start/auto-restart on boot/crash

## Repo layout

- `navaid_main.py` — current version of the script
- `versions/` — earlier snapshots kept for history (see below)
- `CHANGELOG.md` — what changed at each stage and why, including the testing
  that drove each fix
- `KNOWN_ISSUES.md` — open items and things still being tuned

## A note on version history

This project was developed iteratively, testing directly on the Raspberry
Pi hardware after each change. Two full historical snapshots are kept
verbatim in `versions/` — the original pre-IMU script, and the first working
IMU integration. Later iterations were developed as incremental edits rather
than saved as separate files, so their exact intermediate states aren't
reproduced file-for-file; `CHANGELOG.md` documents what changed and why at
each of those stages instead. From this point forward, real git commits
should be used to preserve every version going forward — see "Setting up
git history" below.

## Debug flags

`navaid_main.py` has several debug flags near the top, meant to be enabled
one at a time when testing on hardware:

- `DEBUG_IMU_AXES` — prints raw accelerometer + gyroscope values; used to
  confirm which physical axis corresponds to pitch on your specific mount
- `DEBUG_PITCH` — prints the current smoothed pitch once a second; used to
  verify pitch sign (looking down should read positive)
- `DEBUG_HEIGHT_BANDS` — prints label/band/pitch/coordinates for every close
  detection, for both the YOLO and depth-fallback paths; used to validate
  floor/torso/head classification

## Setting up git history

```bash
cd navaid-repo
git init
git add .
git commit -m "Initial commit: NavAid project history through IMU tilt compensation and floor detection"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

Going forward, commit after each meaningful change on hardware (one commit
per fix/feature) so `git log` becomes the real version history instead of
files named `_v1`, `_v2`, etc.
