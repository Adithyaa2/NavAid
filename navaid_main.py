import depthai as dai
import numpy as np
import blobconverter
import time
import os
import subprocess
import math

def create_pipeline():
    pipeline = dai.Pipeline()
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    mono_left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
    mono_right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)

    stereo = pipeline.create(dai.node.StereoDepth)
    stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(416, 416)
    cam_rgb.setInterleaved(False)
    cam_rgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam_rgb.setFps(10)

    spatial_nn = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
    spatial_nn.setBlobPath(blobconverter.from_zoo(name="yolov4_tiny_coco_416x416", zoo_type="depthai", shaves=6))
    spatial_nn.setConfidenceThreshold(0.2)
    spatial_nn.setBoundingBoxScaleFactor(0.5)
    spatial_nn.setDepthLowerThreshold(100)
    spatial_nn.setDepthUpperThreshold(5000)
    spatial_nn.setNumClasses(80)
    spatial_nn.setCoordinateSize(4)
    spatial_nn.setAnchors([10,14, 23,27, 37,58, 81,82, 135,169, 344,319])
    spatial_nn.setAnchorMasks({"side26": [1,2,3], "side13": [3,4,5]})
    spatial_nn.setIouThreshold(0.3)

    cam_rgb.preview.link(spatial_nn.input)
    stereo.depth.link(spatial_nn.inputDepth)

    xout_nn = pipeline.create(dai.node.XLinkOut)
    xout_nn.setStreamName("detections")
    spatial_nn.out.link(xout_nn.input)

    xout_depth = pipeline.create(dai.node.XLinkOut)
    xout_depth.setStreamName("depth")
    stereo.depth.link(xout_depth.input)

    # --- NEW: IMU node ---
    imu = pipeline.create(dai.node.IMU)
    imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, 100)
    imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, 100)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)

    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)
    # --- END NEW ---

    return pipeline

labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
          "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
          "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
          "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa",
          "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
          "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
          "toothbrush"]

def speak(text):
    os.system(f'espeak "{text}" --stdout | aplay -D plughw:2,0 &')

def get_temp():
    result = subprocess.run(['vcgencmd', 'measure_temp'], capture_output=True, text=True)
    temp_str = result.stdout.strip()
    return float(temp_str.replace("temp=", "").replace("'C", ""))

danger_threshold_ft = 2.5          # tightened from 3.0
speak_cooldown = 8                 # increased from 5
last_spoken_time = 0
temp_check_interval = 10
last_temp_check = 0
MAX_SAFE_TEMP = 60.0

required_frames = 4                # obstacle must be seen this many times in a row before alerting
consecutive_close_frames = {"left": 0, "center": 0, "right": 0}

# --- NEW: IMU / tilt compensation config ---
# Set DEBUG_IMU_AXES = True first and run for ~10s while nodding your head up/down
# and shaking it side to side, watching the printed accel values, to confirm which
# axis responds to nose-up/nose-down pitch. Depthai's axis convention can vary by
# how the module is physically mounted, so don't trust this blindly on hardware.
DEBUG_IMU_AXES = False

# When True, prints "<label>: <band> | pitch=X.Xdeg | y=Ymm z=Zmm" for every close
# detection, every loop iteration. Use this for the height-band validation tests
# (step 4 in the testing protocol) instead of hand-editing the script each time.
DEBUG_HEIGHT_BANDS = False

# When True, prints the current relative pitch in degrees once a second. Use this
# to confirm the SIGN of pitch: looking down should read positive, looking up
# should read negative (per the convention in compute_pitch_from_accel's docstring).
DEBUG_PITCH = False
last_pitch_debug_print = 0

# Approximate height of the camera above the ground when worn (feet). Adjust to
# how you actually wear it (forehead vs chest-mounted, etc). This doesn't need to
# be exact -- it just sets where the floor/torso/head band boundaries fall.
camera_height_ft = 5.5

# Band boundaries (feet above ground)
FLOOR_BAND_MAX_FT = 1.5
TORSO_BAND_MAX_FT = 4.0

pitch_smoothed = 0.0
PITCH_SMOOTHING_ALPHA = 0.15   # lower = smoother/slower to react, higher = twitchier
pitch_baseline = 0.0
calibrated = False

# --- NEW: gyro complementary filter (reduces the large pitch spikes seen during
# fast head movement, since accelerometer-only pitch picks up motion acceleration
# on top of gravity during quick nods) ---
USE_COMPLEMENTARY_FILTER = True
GYRO_TRUST = 0.98   # weight on gyro-integrated estimate vs accel (0-1); higher = smoother but more drift-prone
last_imu_sample_time = None
gyro_bias_x = 0.0   # measured during calibrate(); every real gyro has a small
                     # nonzero reading at rest, and integrating that unsubtracted
                     # for minutes at GYRO_TRUST=0.98 causes runaway drift
# --- END NEW ---

def compute_pitch_from_accel(ax, ay, az):
    """
    Returns pitch in radians. Positive = camera tilted downward (looking toward floor).
    NOTE: axis mapping (which of ax/ay/az is 'forward' vs 'vertical') depends on
    physical IMU orientation on your specific mount. Verify with DEBUG_IMU_AXES
    before trusting this. Default assumption: ay = vertical (down positive),
    az = forward (out of camera lens).
    """
    return math.atan2(az, ay)

def compute_gyro_pitch_rate(gx):
    """
    Angular rate (rad/s) contributing to pitch. gx is rotation about the same
    axis that stayed stable in the accel debug log while nodding (the axis
    ax/az swapped around) -- physically, that's the pitch hinge axis, so gx is
    the right gyro component to use.

    CONFIRMED (via testing): this chip's gyro sign is opposite the right-hand-
    rule assumption -- positive gx alone would push pitch in the WRONG
    direction (looking-up instead of looking-down), so it's negated here to
    match the accel-based convention that was already verified correct
    (positive = looking down).

    gyro_bias_x is subtracted first -- this chip's gyro reads a small nonzero
    rate even at rest, and leaving that unsubtracted causes slow runaway drift
    over minutes of use (confirmed via testing: pitch climbed to 100+ degrees
    and plateaued while the wearer was stationary).
    """
    return -(gx - gyro_bias_x)

def update_pitch(ax, ay, az, gx=None, dt=None):
    global pitch_smoothed, last_imu_sample_time
    accel_pitch = compute_pitch_from_accel(ax, ay, az)

    use_gyro = (
        USE_COMPLEMENTARY_FILTER
        and gx is not None
        and dt is not None
        and 0 < dt < 0.5  # guard against huge gaps (e.g. after a reconnect)
    )

    if use_gyro:
        gyro_rate = compute_gyro_pitch_rate(gx)
        gyro_estimate = pitch_smoothed + (gyro_rate * dt)
        pitch_smoothed = (GYRO_TRUST * gyro_estimate) + ((1 - GYRO_TRUST) * accel_pitch)
    else:
        # Fallback: accel-only exponential smoothing (used when gyro/dt aren't
        # available, e.g. during calibrate() which only reads accel)
        pitch_smoothed = (PITCH_SMOOTHING_ALPHA * accel_pitch) + ((1 - PITCH_SMOOTHING_ALPHA) * pitch_smoothed)

    return pitch_smoothed

def calibrate(device, q_imu, duration_sec=2.0):
    """Capture a neutral pitch baseline AND the gyro's at-rest bias. Have the
    wearer look straight ahead and hold still for both to be measured correctly
    -- any motion during this window corrupts the bias estimate specifically,
    since it assumes the true rotation rate is zero throughout."""
    global pitch_baseline, calibrated, gyro_bias_x
    speak("Calibrating, please look straight ahead and hold still")
    pitch_samples = []
    gyro_samples = []
    start = time.time()
    while time.time() - start < duration_sec:
        imu_data = q_imu.tryGet()
        if imu_data is not None:
            for packet in imu_data.packets:
                a = packet.acceleroMeter
                g = packet.gyroscope
                pitch_samples.append(compute_pitch_from_accel(a.x, a.y, a.z))
                gyro_samples.append(g.x)
        time.sleep(0.02)
    if pitch_samples:
        pitch_baseline = sum(pitch_samples) / len(pitch_samples)
    if gyro_samples:
        gyro_bias_x = sum(gyro_samples) / len(gyro_samples)
    calibrated = True
    print(f"Calibration complete. Baseline pitch: {math.degrees(pitch_baseline):.1f} deg | "
          f"Gyro bias: {gyro_bias_x:.4f} rad/s")

def classify_height_band(spatial_y_mm, spatial_z_mm, pitch_rad):
    """
    spatial_y_mm: object's vertical offset from camera in camera frame (depthai
        convention: typically down is positive -- VERIFY this on your hardware by
        holding an object above vs below the camera and checking the sign).
    spatial_z_mm: object's forward distance from camera in camera frame.
    pitch_rad: current smoothed pitch relative to calibrated baseline (positive =
        looking down).
    Returns one of "floor", "torso", "head".
    """
    rel_pitch = pitch_rad - pitch_baseline

    y_m = spatial_y_mm / 1000.0
    z_m = spatial_z_mm / 1000.0

    # Rotate the camera-frame vertical offset into a gravity-aligned frame.
    # world_down_m > 0 means the object sits below camera level.
    world_down_m = (y_m * math.cos(rel_pitch)) + (z_m * math.sin(rel_pitch))

    height_above_ground_ft = camera_height_ft - (world_down_m * 3.28084)

    if height_above_ground_ft < FLOOR_BAND_MAX_FT:
        return "floor"
    elif height_above_ground_ft < TORSO_BAND_MAX_FT:
        return "torso"
    else:
        return "head"

HEIGHT_BAND_PHRASE = {
    "floor": "near the floor",
    "torso": None,  # no extra phrase for the common case, keeps default messages unchanged
    "head": "at head height",
}
# --- END NEW ---

def describe_distance(feet):
    if feet < 1.5:
        return "right in front of you"
    elif feet < 3:
        return "very close"
    else:
        return "nearby"

def get_zone(det, frame_width=416):
    center_x = (det.xmin + det.xmax) / 2 * frame_width
    third = frame_width / 3
    if center_x < third:
        return "left"
    elif center_x < 2 * third:
        return "center"
    else:
        return "right"

# --- NEW: separate vertical scan bands for the raw-depth fallback ---
# GENERAL band = original torso-height obstacle range. FLOOR band = new, added to
# catch low/unrecognized obstacles (clutter, curbs, dropped items) that YOLO can't
# name and that the original middle-third-only scan never looked at.
GENERAL_SCAN_ROW_START, GENERAL_SCAN_ROW_END = 1/3, 2/3
FLOOR_SCAN_ROW_START, FLOOR_SCAN_ROW_END = 3/4, 1.0

# Approximate vertical field of view of the depth frame, in degrees. Used to
# convert a scan region's row position into an angle for height estimation.
# STARTING GUESS, NOT VERIFIED -- OAK-D Lite's exact effective vertical FOV after
# depth alignment isn't something I can confirm without the hardware in front of
# me. Verify/tune using DEBUG_HEIGHT_BANDS: place a known object (e.g. on the
# floor) in view, check whether it's classified "floor" as expected, and adjust
# this value if it's consistently off.
VERTICAL_FOV_DEG = 55.0
# --- END NEW ---

def get_zone_distance_depth(frame, zone, row_start_frac=GENERAL_SCAN_ROW_START, row_end_frac=GENERAL_SCAN_ROW_END):
    h, w = frame.shape
    third = w // 3
    r0, r1 = int(h * row_start_frac), int(h * row_end_frac)
    if zone == "left":
        region = frame[r0:r1, 0:third]
    elif zone == "center":
        region = frame[r0:r1, third:2*third]
    elif zone == "right":
        region = frame[r0:r1, 2*third:w]

    valid = region[region > 0]
    if len(valid) > 20:
        closest_mm = np.percentile(valid, 10)
        dist_ft = (closest_mm / 1000) * 3.28084
        row_center_frac = (row_start_frac + row_end_frac) / 2
        return dist_ft, row_center_frac
    return None, None

# --- NEW: estimate height band for raw-depth (unclassified) detections ---
def estimate_band_from_pixel_row(row_center_frac, distance_ft, pitch_rad):
    """
    Coarser sibling of classify_height_band, for detections that don't have
    depthai's spatialCoordinates (i.e. the raw-depth fallback, not a YOLO
    detection). Converts the scanned region's vertical position in-frame into an
    approximate angle using VERTICAL_FOV_DEG, derives a pseudo camera-frame Y
    offset from that angle + distance, then reuses the same verified rotation
    math as classify_height_band.
    """
    angle_from_center_deg = (row_center_frac - 0.5) * VERTICAL_FOV_DEG
    distance_mm = (distance_ft / 3.28084) * 1000
    y_approx_mm = distance_mm * math.tan(math.radians(angle_from_center_deg))
    return classify_height_band(y_approx_mm, distance_mm, pitch_rad)
# --- END NEW ---

def update_zone_counter(zone, is_close):
    if is_close:
        consecutive_close_frames[zone] += 1
    else:
        consecutive_close_frames[zone] = 0

# --- NEW: filter invalid spatial detections ---
def has_valid_spatial_coords(det):
    """
    depthai reports z=0 (and often x=0, y=0 too) when it couldn't compute valid
    depth for a bounding box -- e.g. object near the edge of the stereo overlap,
    or outside setDepthLowerThreshold/setDepthUpperThreshold. z=0 does NOT mean
    "zero distance" -- it means "no reading" -- so these must be filtered out,
    not treated as extremely close obstacles.
    """
    return det.spatialCoordinates.z > 50  # small margin above 0, in mm
# --- END NEW ---

def run_detection_loop():
    global last_spoken_time, last_temp_check, calibrated, last_pitch_debug_print, last_imu_sample_time
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_detections = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        q_imu = device.getOutputQueue(name="imu", maxSize=50, blocking=False)
        print("Connected! Running detection loop...")

        # --- NEW: run calibration once per pipeline (re)start ---
        if not calibrated:
            calibrate(device, q_imu)
        # --- END NEW ---

        # --- NEW: optional axis debug mode ---
        if DEBUG_IMU_AXES:
            print("DEBUG_IMU_AXES enabled. Nod head up/down, shake side to side. Ctrl+C to stop.")
            while True:
                imu_data = q_imu.tryGet()
                if imu_data is not None:
                    for packet in imu_data.packets:
                        a = packet.acceleroMeter
                        g = packet.gyroscope
                        print(f"ax={a.x:.2f} ay={a.y:.2f} az={a.z:.2f}  |  gx={g.x:.2f} gy={g.y:.2f} gz={g.z:.2f}")
                time.sleep(0.1)
        # --- END NEW ---

        while True:
            current_time = time.time()

            if (current_time - last_temp_check) > temp_check_interval:
                temp = get_temp()
                last_temp_check = current_time
                if temp > MAX_SAFE_TEMP:
                    speak("Warning, device is overheating. Pausing briefly to cool down.")
                    time.sleep(60)
                    continue

            in_det = q_detections.tryGet()
            in_depth = q_depth.tryGet()

            # --- NEW: drain IMU queue and update smoothed pitch every loop ---
            imu_data = q_imu.tryGet()
            if imu_data is not None:
                for packet in imu_data.packets:
                    a = packet.acceleroMeter
                    g = packet.gyroscope
                    # Use the sensor's own device timestamp, not wall-clock processing
                    # time -- when packets arrive batched, processing them in a tight
                    # loop makes wall-clock dt near-zero even though the underlying
                    # samples span real time, which was corrupting the gyro integration.
                    sample_time = a.timestamp.get().total_seconds()
                    dt = (sample_time - last_imu_sample_time) if last_imu_sample_time is not None else None
                    update_pitch(a.x, a.y, a.z, g.x, dt)
                    last_imu_sample_time = sample_time

            if DEBUG_PITCH:
                if (current_time - last_pitch_debug_print) > 1.0:
                    rel_pitch_deg = math.degrees(pitch_smoothed - pitch_baseline)
                    print(f"pitch = {rel_pitch_deg:.1f} deg (positive should mean looking down)")
                    last_pitch_debug_print = current_time
            # --- END NEW ---

            if in_det is None or in_depth is None:
                time.sleep(0.1)
                continue

            detections = in_det.detections
            depth_frame = in_depth.getFrame()

            zone_info = {"left": None, "center": None, "right": None}

            close_objects = []
            for det in detections:
                if not has_valid_spatial_coords(det):
                    continue
                distance_ft = (det.spatialCoordinates.z / 1000) * 3.28084
                if distance_ft < danger_threshold_ft:
                    close_objects.append((det, distance_ft))

            if close_objects:
                for det, dist in close_objects:
                    zone = get_zone(det)
                    label = labels[det.label]
                    # --- NEW: classify height band for named detections ---
                    band = classify_height_band(
                        det.spatialCoordinates.y,
                        det.spatialCoordinates.z,
                        pitch_smoothed,
                    )
                    if DEBUG_HEIGHT_BANDS:
                        rel_pitch_deg = math.degrees(pitch_smoothed - pitch_baseline)
                        print(f"{label}: {band} | pitch={rel_pitch_deg:.1f}deg | "
                              f"y={det.spatialCoordinates.y:.0f}mm z={det.spatialCoordinates.z:.0f}mm")
                    # --- END NEW ---
                    if zone_info[zone] is None or dist < zone_info[zone][1]:
                        zone_info[zone] = (label, dist, band)
            else:
                for zone in ["left", "center", "right"]:
                    # --- NEW: scan both general (torso-height) and floor bands ---
                    general_dist, general_row = get_zone_distance_depth(
                        depth_frame, zone, GENERAL_SCAN_ROW_START, GENERAL_SCAN_ROW_END
                    )
                    floor_dist, floor_row = get_zone_distance_depth(
                        depth_frame, zone, FLOOR_SCAN_ROW_START, FLOOR_SCAN_ROW_END
                    )
                    candidates = []
                    if general_dist is not None and general_dist < danger_threshold_ft:
                        candidates.append((general_dist, general_row))
                    if floor_dist is not None and floor_dist < danger_threshold_ft:
                        candidates.append((floor_dist, floor_row))

                    if candidates:
                        dist, row = min(candidates, key=lambda c: c[0])  # most urgent (closest)
                        band = estimate_band_from_pixel_row(row, dist, pitch_smoothed)
                        if DEBUG_HEIGHT_BANDS:
                            rel_pitch_deg = math.degrees(pitch_smoothed - pitch_baseline)
                            print(f"[depth-fallback] {zone}: {band} | pitch={rel_pitch_deg:.1f}deg | "
                                  f"dist={dist:.2f}ft row_frac={row:.2f}")
                        zone_info[zone] = (None, dist, band)
                    # --- END NEW ---

            for zone in ["left", "center", "right"]:
                update_zone_counter(zone, zone_info[zone] is not None)

            if (current_time - last_spoken_time) > speak_cooldown:
                message = None

                center_confirmed = consecutive_close_frames["center"] >= required_frames
                left_confirmed = consecutive_close_frames["left"] >= required_frames
                right_confirmed = consecutive_close_frames["right"] >= required_frames

                # --- NEW: unified message builder, phrases floor hazards naturally ---
                def build_message(label, desc, band, side_phrase=None):
                    side = f" on your {side_phrase}" if side_phrase else ""
                    if label:
                        # Named (YOLO) detection
                        base = f"{label}{side}, {desc}"
                        extra = HEIGHT_BAND_PHRASE.get(band) if band else None
                        return f"{base}, {extra}" if extra else base
                    elif band == "floor":
                        # Unclassified low obstacle -- phrase distinctly since this
                        # is specifically a trip/step hazard, not just "something's there"
                        return f"Low obstacle{side}, {desc}, watch your step"
                    else:
                        return f"Obstacle{side}, {desc}, proceed with caution"
                # --- END NEW ---

                if center_confirmed:
                    label, dist, band = zone_info["center"]
                    desc = describe_distance(dist)
                    message = build_message(label, desc, band)
                elif left_confirmed and right_confirmed:
                    message = "Careful, obstacles on both sides"
                elif left_confirmed:
                    label, dist, band = zone_info["left"]
                    desc = describe_distance(dist)
                    message = build_message(label, desc, band, side_phrase="left")
                elif right_confirmed:
                    label, dist, band = zone_info["right"]
                    desc = describe_distance(dist)
                    message = build_message(label, desc, band, side_phrase="right")

                if message:
                    print(f"SPEAKING: {message}")
                    speak(message)
                    last_spoken_time = current_time

            time.sleep(0.1)

while True:
    try:
        run_detection_loop()
    except Exception as e:
        print(f"Pipeline error: {e}. Reconnecting in 5 seconds...")
        time.sleep(5)
