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

def compute_pitch_from_accel(ax, ay, az):
    """
    Returns pitch in radians. Positive = camera tilted downward (looking toward floor).
    NOTE: axis mapping (which of ax/ay/az is 'forward' vs 'vertical') depends on
    physical IMU orientation on your specific mount. Verify with DEBUG_IMU_AXES
    before trusting this. Default assumption: ay = vertical (down positive),
    az = forward (out of camera lens).
    """
    return math.atan2(az, ay)

def update_pitch(ax, ay, az):
    global pitch_smoothed
    raw_pitch = compute_pitch_from_accel(ax, ay, az)
    pitch_smoothed = (PITCH_SMOOTHING_ALPHA * raw_pitch) + ((1 - PITCH_SMOOTHING_ALPHA) * pitch_smoothed)
    return pitch_smoothed

def calibrate(device, q_imu, duration_sec=2.0):
    """Capture a neutral pitch baseline. Have the wearer look straight ahead and hold still."""
    global pitch_baseline, calibrated
    speak("Calibrating, please look straight ahead and hold still")
    samples = []
    start = time.time()
    while time.time() - start < duration_sec:
        imu_data = q_imu.tryGet()
        if imu_data is not None:
            for packet in imu_data.packets:
                a = packet.acceleroMeter
                samples.append(compute_pitch_from_accel(a.x, a.y, a.z))
        time.sleep(0.02)
    if samples:
        pitch_baseline = sum(samples) / len(samples)
    calibrated = True
    print(f"Calibration complete. Baseline pitch: {math.degrees(pitch_baseline):.1f} deg")

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

def get_zone_distance_depth(frame, zone):
    h, w = frame.shape
    third = w // 3
    if zone == "left":
        region = frame[h//3 : 2*h//3, 0:third]
    elif zone == "center":
        region = frame[h//3 : 2*h//3, third:2*third]
    elif zone == "right":
        region = frame[h//3 : 2*h//3, 2*third:w]

    valid = region[region > 0]
    if len(valid) > 20:
        closest_mm = np.percentile(valid, 10)
        return (closest_mm / 1000) * 3.28084
    return None

def update_zone_counter(zone, is_close):
    if is_close:
        consecutive_close_frames[zone] += 1
    else:
        consecutive_close_frames[zone] = 0

def run_detection_loop():
    global last_spoken_time, last_temp_check, calibrated
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
                        print(f"ax={a.x:.2f} ay={a.y:.2f} az={a.z:.2f}")
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
                    update_pitch(a.x, a.y, a.z)
            # --- END NEW ---

            if in_det is None or in_depth is None:
                time.sleep(0.1)
                continue

            detections = in_det.detections
            depth_frame = in_depth.getFrame()

            zone_info = {"left": None, "center": None, "right": None}

            close_objects = []
            for det in detections:
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
                    # --- END NEW ---
                    if zone_info[zone] is None or dist < zone_info[zone][1]:
                        zone_info[zone] = (label, dist, band)
            else:
                for zone in ["left", "center", "right"]:
                    dist = get_zone_distance_depth(depth_frame, zone)
                    if dist and dist < danger_threshold_ft:
                        zone_info[zone] = (None, dist, None)  # no height band for depth fallback yet

            for zone in ["left", "center", "right"]:
                update_zone_counter(zone, zone_info[zone] is not None)

            if (current_time - last_spoken_time) > speak_cooldown:
                message = None

                center_confirmed = consecutive_close_frames["center"] >= required_frames
                left_confirmed = consecutive_close_frames["left"] >= required_frames
                right_confirmed = consecutive_close_frames["right"] >= required_frames

                # --- NEW: helper to append height phrase when available ---
                def with_height(base_message, band):
                    phrase = HEIGHT_BAND_PHRASE.get(band) if band else None
                    return f"{base_message}, {phrase}" if phrase else base_message
                # --- END NEW ---

                if center_confirmed:
                    label, dist, band = zone_info["center"]
                    desc = describe_distance(dist)
                    base = f"{label} {desc}" if label else f"Obstacle {desc}, proceed with caution"
                    message = with_height(base, band)
                elif left_confirmed and right_confirmed:
                    message = "Careful, obstacles on both sides"
                elif left_confirmed:
                    label, dist, band = zone_info["left"]
                    desc = describe_distance(dist)
                    base = f"{label} on your left, {desc}" if label else f"Obstacle on your left, {desc}"
                    message = with_height(base, band)
                elif right_confirmed:
                    label, dist, band = zone_info["right"]
                    desc = describe_distance(dist)
                    base = f"{label} on your right, {desc}" if label else f"Obstacle on your right, {desc}"
                    message = with_height(base, band)

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
