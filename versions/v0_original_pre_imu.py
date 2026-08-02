import depthai as dai
import numpy as np
import blobconverter
import time
import os
import subprocess

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
    global last_spoken_time, last_temp_check
    pipeline = create_pipeline()

    with dai.Device(pipeline) as device:
        q_detections = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
        q_depth = device.getOutputQueue(name="depth", maxSize=4, blocking=False)
        print("Connected! Running detection loop...")

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
                    if zone_info[zone] is None or dist < zone_info[zone][1]:
                        zone_info[zone] = (label, dist)
            else:
                for zone in ["left", "center", "right"]:
                    dist = get_zone_distance_depth(depth_frame, zone)
                    if dist and dist < danger_threshold_ft:
                        zone_info[zone] = (None, dist)

            for zone in ["left", "center", "right"]:
                update_zone_counter(zone, zone_info[zone] is not None)

            if (current_time - last_spoken_time) > speak_cooldown:
                message = None

                center_confirmed = consecutive_close_frames["center"] >= required_frames
                left_confirmed = consecutive_close_frames["left"] >= required_frames
                right_confirmed = consecutive_close_frames["right"] >= required_frames

                if center_confirmed:
                    label, dist = zone_info["center"]
                    desc = describe_distance(dist)
                    message = f"{label} {desc}" if label else f"Obstacle {desc}, proceed with caution"
                elif left_confirmed and right_confirmed:
                    message = "Careful, obstacles on both sides"
                elif left_confirmed:
                    label, dist = zone_info["left"]
                    desc = describe_distance(dist)
                    message = f"{label} on your left, {desc}" if label else f"Obstacle on your left, {desc}"
                elif right_confirmed:
                    label, dist = zone_info["right"]
                    desc = describe_distance(dist)
                    message = f"{label} on your right, {desc}" if label else f"Obstacle on your right, {desc}"

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
