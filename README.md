# NavAid

A wearable navigation aid for people with low vision, built on a Raspberry Pi 5
and a Luxonis OAK-D Lite depth camera. It detects obstacles in the wearer's
path — including obstacles YOLO can't classify, and objects on the floor
regardless of the wearer's head angle — and announces spoken warnings through a
head-worn speaker (e.g. *"chair right in front of you"*, *"low obstacle, very
close, watch your step"*).

I started NavAid for my grandmother, who lost most of her vision after an
illness. The goal is a device reliable enough for her to test at home.

<!-- TODO: add a photo of the full wearable here, e.g.
![NavAid prototype](docs/images/navaid-worn.jpg) -->

## Status

**Working prototype, tested on the bench and indoors.** Obstacle detection,
depth fusion, IMU head-tilt compensation and spoken alerts all run on the Pi.
Haptic feedback, the head mount and battery testing are still in progress — see
[Roadmap](#roadmap) and [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

<!-- TODO: add a short GIF of the depth view / a demo video link here -->

## How it works

- **Object detection + depth fusion**: YOLOv4-tiny (COCO, 80 classes) fused
  with stereo depth via depthai's `YoloSpatialDetectionNetwork`, giving each
  detection a real-world distance and position.
- **Dual-path obstacle detection**: named objects (chair, person, etc.) go
  through the YOLO path; anything YOLO can't classify — clutter, curbs,
  dropped items — is still caught by a raw-depth fallback that scans two
  vertical bands of the frame (general/torso-height and floor-height).
- **IMU-based height classification**: the OAK-D Lite's onboard BMI270 IMU
  tracks head pitch, so an object's height (floor / torso / head) is computed
  relative to the *ground*, not relative to wherever the wearer's head happens
  to be pointing. Looking down at your feet doesn't make a shelf at head
  height read as a floor hazard, and vice versa.
- **Sensor fusion for pitch**: accelerometer + gyroscope combined via a
  complementary filter, with a calibration step at startup that measures both
  a neutral pitch baseline and the gyro's at-rest bias (uncorrected gyro bias
  causes slow runaway drift over long sessions — see CHANGELOG).
- **Noise reduction**: persistence filtering (an obstacle must be seen for 4
  consecutive frames before it's announced), an 8-second cooldown between
  spoken alerts, and filtering of invalid zero-depth detections.
- **Reliability**: runs as an auto-restarting systemd service, with an outer
  reconnect loop and a temperature safety cutoff (60 °C).

### Alert behavior

| Setting | Value | Where |
|---|---|---|
| Alert distance | under 2.5 ft | `danger_threshold_ft` |
| "right in front of you" | under 1.5 ft | `describe_distance()` |
| "very close" | 1.5–3 ft | `describe_distance()` |
| Frames before alerting | 4 in a row | `required_frames` |
| Cooldown between alerts | 8 s | `speak_cooldown` |
| Height bands | floor < 1.5 ft, torso < 4 ft, head above | `FLOOR_BAND_MAX_FT`, `TORSO_BAND_MAX_FT` |

## Hardware

- Raspberry Pi 5
- Luxonis OAK-D Lite depth camera (onboard BMI270 6-axis IMU), USB-C to the
  Pi's USB 3.0 port
- USB speaker (audio via ALSA / `aplay`)
- Planned: DRV2605L haptic motor driver + ERM vibration motor; bone conduction
  headset via a USB-to-3.5mm adapter
- Mount: GoPro-style head strap + camera clamp (in progress)

<!-- TODO: add a close-up photo of the camera + Pi wiring, e.g.
![Camera and Pi](docs/images/navaid-hardware.jpg) -->

## Setup

Tested on Raspberry Pi OS (Debian-based) with Python 3.13.

### 1. System packages

```bash
sudo apt update
sudo apt install -y espeak alsa-utils
# TODO: confirm any other apt packages you installed
```

### 2. OAK-D USB permissions

The OAK-D needs a udev rule so it can be used without `sudo`:

```bash
echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

<!-- TODO: confirm this matches what's on your Pi (ls /etc/udev/rules.d/) -->

### 3. Python environment

```bash
git clone https://github.com/Adithyaa2/NavAid.git ~/navaid
cd ~/navaid
python3 -m venv venv
source venv/bin/activate
pip install depthai==2.24.0.0 depthai-sdk numpy blobconverter opencv-contrib-python
```

On first run, `blobconverter` downloads the YOLOv4-tiny model, so the Pi needs
internet access once.

### 4. Audio output

`speak()` sends audio to ALSA card 2 (`plughw:2,0`). Find your speaker's card
number with:

```bash
aplay -l
```

and update the `plughw:X,0` value in `speak()` if it's different.

## Running

### Manually (for testing)

```bash
sudo systemctl stop navaid      # free the camera if the service is running
source ~/navaid/venv/bin/activate
python3 ~/navaid/navaid_main.py
```

Hold still and look straight ahead for the first ~2 seconds while it
calibrates the IMU.

### As a service (auto-start on boot)

Example unit file at `/etc/systemd/system/navaid.service`:

```ini
# TODO: replace with the actual file from the Pi
# (cat /etc/systemd/system/navaid.service)
[Unit]
Description=NavAid obstacle detection
After=multi-user.target

[Service]
User=<your-username>
WorkingDirectory=/home/<your-username>/navaid
ExecStart=/home/<your-username>/navaid/venv/bin/python3 navaid_main.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

Useful commands:

```bash
sudo systemctl enable --now navaid   # start now and on every boot
sudo systemctl status navaid         # is it running?
journalctl -u navaid -f              # live log output
sudo systemctl stop navaid           # stop it (needed before a manual run)
```

## Debug flags

`navaid_main.py` has several debug flags near the top, meant to be enabled one
at a time when testing on hardware:

- `DEBUG_IMU_AXES` — prints raw accelerometer + gyroscope values; used to
  confirm which physical axis corresponds to pitch on your specific mount
- `DEBUG_PITCH` — prints the current smoothed pitch once a second; used to
  verify pitch sign (looking down should read positive)
- `DEBUG_HEIGHT_BANDS` — prints label/band/pitch/coordinates for every close
  detection, for both the YOLO and depth-fallback paths; used to validate
  floor/torso/head classification

## Roadmap

- [ ] Haptic feedback via the DRV2605L (hardware purchased, not wired in)
- [ ] Bone conduction headset
- [ ] Physical head-mount assembly
- [ ] Fix overlapping audio (queue speech instead of launching in parallel)
- [ ] Battery runtime testing under continuous real-world use
- [ ] Periodic gyro bias recalibration during long sessions
- [ ] Verify `VERTICAL_FOV_DEG` and measure real `camera_height_ft` when worn
- [ ] Home testing with my grandmother

Details on each are in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

## Repo layout

- `navaid_main.py` — current version of the script
- `versions/` — earlier snapshots kept for history
- `CHANGELOG.md` — what changed at each stage and why, including the testing
  that drove each fix
- `KNOWN_ISSUES.md` — open items, things still being tuned, and environment
  gotchas

## Version history

NavAid was developed iteratively, testing on the Raspberry Pi hardware after
each change. Two full historical snapshots are kept verbatim in `versions/` —
the original pre-IMU script and the first working IMU integration. Later
iterations (v2–v5) were made as incremental edits, so `CHANGELOG.md` documents
what changed and why at each of those stages. From here on, every change is
tracked in git commits.
