# Known Issues / Open Items

## Unverified / needs on-hardware confirmation

- **`VERTICAL_FOV_DEG` (currently 55.0)** — used to estimate height bands for
  unclassified (depth-fallback) detections. This is a starting guess, not a
  confirmed spec for the OAK-D Lite's depth frame after alignment to the
  color camera. Verify using `DEBUG_HEIGHT_BANDS`: place a known object on
  the floor vs. at head height (isolated from other objects, so it hits the
  fallback path) and check whether the reported band matches reality.
- **`camera_height_ft` (currently 5.5)** — should be updated to the wearer's
  actual measured camera height when worn, rather than left as a guess.

## Audio overlap

`speak()` launches `espeak | aplay` in the background (`&`) without waiting
for the previous call to finish. When a new warning fires while the speaker
is still playing the last one, this produces:

```
aplay: main:850: audio open error: Device or resource busy
```

Not currently fixed — a queued/serialized speech approach (or dropping
overlapping calls) would resolve it.

## Not yet implemented

- Haptic feedback via the DRV2605L motor driver (hardware purchased, not
  wired in)
- Bone conduction headset configuration (hardware purchased; needs its ALSA
  card number identified via `aplay -l` once connected, and the `speak()`
  function's `plughw:X,0` updated accordingly)
- Physical head-mount assembly
- Battery runtime testing under continuous real-world use
- Periodic gyro bias/pitch recalibration during a long session (see
  CHANGELOG v5 known limitation)

## Environment gotchas (not code bugs, but easy to trip on)

- `X_LINK_DEVICE_ALREADY_IN_USE` — means something else (usually the
  `navaid.service` systemd service) already has the camera open. Run
  `sudo systemctl stop navaid` before any manual `python3` run.
- `ModuleNotFoundError: No module named 'depthai'` — the venv isn't
  activated in the current shell. Run `source ~/navaid/venv/bin/activate`.
- Windows `scp`/`ssh` to a `.local` mDNS hostname (e.g. `daddy.local`) fails
  with "Could not resolve hostname" because Windows doesn't support mDNS
  out of the box. Use the Pi's static IP instead, or install Bonjour.
