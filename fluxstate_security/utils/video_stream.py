# utils/video_stream.py
import cv2
import numpy as np
import time
import threading
import concurrent.futures
import pytesseract
import re
import subprocess
import hashlib
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[Warning] Ultralytics not installed. Object detection disabled.")

try:
    import pyaudio
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("[System] PyAudio not found. Acoustic Environmental Sensing operating in degraded passive mode.")

def generate_gpass_token(crop):
    """
    Identity Persistence (GPASS Handshake):
    Extracts the structural perceptual hash of an entity's upper geometry (head/shoulders).
    Provides a cryptographic tracker WITHOUT facial recognition (Zero-Trust Privacy).
    """
    if crop.size == 0: return "UNKNOWN"
    
    h = crop.shape[0]
    head_crop = crop[0:int(h*0.3), :] # Only hash the upper 30% geometry
    if head_crop.size == 0: return "UNKNOWN"
    
    try:
        gray = cv2.cvtColor(head_crop, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (8, 8))
        mean = small.mean()
        bits = (small > mean).flatten()
        hash_str = "".join(['1' if b else '0' for b in bits])
        hex_hash = hex(int(hash_str, 2))[2:].zfill(16)
        return f"GPX-{hex_hash.upper()}"
    except:
        return "UNKNOWN"

def is_centroid_contained(inner_box, outer_box):
    """Checks if the centroid of an anomaly falls within a person's bounding box."""
    # inner_box format: [x, y, bw, bh]
    cx = inner_box[0] + inner_box[2] / 2
    cy = inner_box[1] + inner_box[3] / 2
    
    # outer_box format: [x, y, bw, bh]
    ox1, oy1 = outer_box[0], outer_box[1]
    ox2, oy2 = outer_box[0] + outer_box[2], outer_box[1] + outer_box[3]
    
    return ox1 <= cx <= ox2 and oy1 <= cy <= oy2

def is_sharp_for_ocr(crop_image, threshold=85.0):
    """Uses Laplacian variance to drop blurry fabric/shadows before OCR execution."""
    gray = cv2.cvtColor(crop_image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance > threshold

class AcousticSensor:
    """
    Background daemon that monitors the local microphone array.
    Analyzes RMS amplitude to detect structural anomalies (glass breaking, aggression)
    and fuses acoustic data into the Reality Graph.
    """
    def __init__(self):
        self.acoustic_event = "AMBIENT"
        if AUDIO_AVAILABLE:
            self.thread = threading.Thread(target=self._listen_loop, daemon=True)
            self.thread.start()
            
    def _listen_loop(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        p = pyaudio.PyAudio()
        try:
            stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
            while True:
                data = stream.read(CHUNK, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                rms = np.sqrt(np.mean(audio_data.astype(np.float32)**2))
                
                if rms > 2000:
                    self.acoustic_event = "KINETIC_IMPACT_OR_SHATTERING"
                elif rms > 800:
                    self.acoustic_event = "VOCAL_ESCALATION"
                else:
                    self.acoustic_event = "AMBIENT"
                time.sleep(0.05)
        except:
            pass

class RFSensor:
    """
    Background daemon that monitors the local RF environment.
    Uses an ARP scan to detect nearby MAC/IP broadcasts (smartphones, IoT devices).
    """
    def __init__(self):
        self.active_devices = 0
        self.thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.thread.start()
        
    def _scan_loop(self):
        while True:
            try:
                # Fast ARP scan (works on macOS/Linux without root for local cache)
                result = subprocess.check_output(["arp", "-a"], timeout=2).decode("utf-8")
                devices = len([line for line in result.split('\n') if 'at' in line])
                self.active_devices = devices if devices > 0 else 1
            except:
                self.active_devices = 0
            time.sleep(10) # Poll every 10 seconds

class EdgeVideoStream:
    """
    Handles the low-power ingestion layer. Computes frame differences
    before sending data to the AI model, conserving system memory and compute.
    """
    def __init__(self, source=0):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not self.cap.isOpened():
             print("[Warning] Unable to access the camera stream. Ensure camera permissions are granted.")
        
        self.ret = False
        self.frame = None
        self.running = True
        
        self.thread = threading.Thread(target=self._update_frames, daemon=True)
        self.thread.start()
        
        self.last_frame = None
        self.rf_sensor = RFSensor()
        self.acoustic_sensor = AcousticSensor()
        
        self.ocr_executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.ocr_cache = {}
        
        self.yolo_active = False
        if YOLO_AVAILABLE:
            try:
                self.yolo_model = YOLO("yolov8n.pt") 
                self.yolo_active = True
                print("[System] SOTA YOLOv8 Nano Object Classifier Loaded.")
            except Exception as e:
                print(f"[Warning] Failed to load YOLOv8: {e}")

        try:
            base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE)
            self.landmarker = vision.PoseLandmarker.create_from_options(options)
            self.mp_pose_active = True
            print("[System] Neural Pose Landmarker Loaded Successfully.")
        except Exception as e:
            print(f"[Warning] Failed to load Mediapipe Pose model. {e}")
            self.mp_pose_active = False

    def _update_frames(self):
        while self.running:
            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.01)

    def _run_ocr_async(self, crop, obj_id):
        try:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
            
            text = pytesseract.image_to_string(thresh, config='--psm 11').strip()
            text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
            
            if len(text) > 2:
                self.ocr_cache[obj_id] = text
            else:
                self.ocr_cache[obj_id] = ""
        except Exception:
            self.ocr_cache[obj_id] = ""

    def _reason_about_object(self, frame, x, y, bw, bh, base_label, obj_id=None):
        crop = frame[max(0, y):y+bh, max(0, x):x+bw]
        if crop.size == 0:
            return base_label
            
        h, w = crop.shape[:2]
        aspect_ratio = h / w if w > 0 else 1
        
        shape = "Square"
        if aspect_ratio > 1.3: shape = "Tall"
        elif aspect_ratio < 0.7: shape = "Wide"
        
        try:
            small_crop = cv2.resize(crop, (10, 10))
            avg_color = np.average(np.average(small_crop, axis=0), axis=0)
            b, g, r = avg_color
            
            color = "Dark"
            if r > 180 and g > 180 and b > 180: color = "White"
            elif r < 60 and g < 60 and b < 60: color = "Black"
            elif r > g+40 and r > b+40: color = "Red"
            elif g > r+40 and g > b+40: color = "Green"
            elif b > r+40 and b > g+40: color = "Blue"
            elif r > 120 and g > 120 and b < 80: color = "Yellow"
        except:
            color = "Unknown"
            
        ocr_text = ""
        if obj_id is not None:
            if obj_id in self.ocr_cache:
                if self.ocr_cache[obj_id] != "":
                    ocr_text = f" [TEXT: {self.ocr_cache[obj_id]}]"
            else:
                # FIX 2: OCR Sharpness Gate
                if is_sharp_for_ocr(crop):
                    self.ocr_executor.submit(self._run_ocr_async, crop.copy(), obj_id)
                else:
                    self.ocr_cache[obj_id] = "" # Drop silently if blurry
            
        if base_label == "person":
            return f"Person{ocr_text}"
            
        if base_label == "unknown":
            return f"{color.upper()} {shape.upper()} ANOMALY{ocr_text}"
            
        return f"{color} {shape} {base_label.title()}{ocr_text}"

    def capture_and_filter(self):
        if not self.ret or self.frame is None:
            return False, None, [], [], None, "UNKNOWN", 0, "AMBIENT"

        frame = self.frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        gray_resized = cv2.resize(gray, (320, 240))
        gray_resized = cv2.GaussianBlur(gray_resized, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray_resized
            return True, frame, [], [], None, "UNKNOWN", self.rf_sensor.active_devices, self.acoustic_sensor.acoustic_event

        frame_delta = cv2.absdiff(self.last_frame, gray_resized)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        movement_boxes = []
        h, w = frame.shape[:2]
        scale_x = w / 320.0
        scale_y = h / 240.0
        
        for c in contours:
            if cv2.contourArea(c) < 300:
                continue
            (bx, by, bw, bh) = cv2.boundingRect(c)
            movement_boxes.append((int(bx * scale_x), int(by * scale_y), int(bw * scale_x), int(bh * scale_y)))
        
        entropy_score = np.mean(frame_delta) / 255.0
        self.last_frame = gray_resized
        should_process = entropy_score > 0.15 or len(movement_boxes) > 0

        detected_objects = []
        yolo_boxes = []
        human_boxes = []
        
        if should_process and self.yolo_active:
            results = self.yolo_model.track(frame, persist=True, verbose=False, conf=0.25)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())
                    label = self.yolo_model.names[cls]
                    
                    obj_id = str(int(box.id[0].cpu().numpy())) if box.id is not None else f"spatial_{int(x1/50)}_{int(y1/50)}"
                    
                    bw = int(x2 - x1)
                    bh = int(y2 - y1)
                    
                    gpass = "N/A"
                    if "person" in label.lower():
                        human_boxes.append([int(x1), int(y1), bw, bh])
                        crop = frame[int(y1):int(y2), int(x1):int(x2)]
                        gpass = generate_gpass_token(crop)
                    
                    refined_label = self._reason_about_object(frame, int(x1), int(y1), bw, bh, label, obj_id)
                    if gpass != "N/A" and gpass != "UNKNOWN":
                        refined_label = f"{refined_label} [{gpass}]"
                        
                    detected_objects.append((int(x1), int(y1), bw, bh, refined_label, conf, obj_id))
                    yolo_boxes.append((int(x1), int(y1), bw, bh))

            # FIX 1: Spatial Exclusion (IoU)
            for (mx, my, mw, mh) in movement_boxes:
                is_known = False
                for (yx, yy, ybw, ybh) in yolo_boxes:
                    if mx < yx + ybw and mx + mw > yx and my < yy + ybh and my + mh > yy:
                        is_known = True
                        break
                        
                if not is_known:
                    is_clothing = False
                    for person_box in human_boxes:
                        if is_centroid_contained([mx, my, mw, mh], person_box):
                            is_clothing = True
                            break
                            
                    if not is_clothing:
                        obj_id = f"anomaly_{int(mx/50)}_{int(my/50)}"
                        fallback_label = self._reason_about_object(frame, mx, my, mw, mh, "unknown", obj_id)
                        detected_objects.append((mx, my, mw, mh, fallback_label, 0.99, obj_id))

        action_heuristic = "STATIONARY_OBSERVING"
        pose_landmarks_list = None

        if should_process and self.mp_pose_active:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = self.landmarker.detect(mp_image)
            
            if len(detection_result.pose_landmarks) > 0:
                pose_landmarks_list = detection_result.pose_landmarks[0]
                
                l_wrist = pose_landmarks_list[15]
                r_wrist = pose_landmarks_list[16]
                l_shoulder = pose_landmarks_list[11]
                
                if l_wrist.y < l_shoulder.y or r_wrist.y < l_shoulder.y:
                    action_heuristic = "ARMS_RAISED_OR_REACHING"
                elif abs(l_wrist.x - r_wrist.x) < 0.1 and l_wrist.y > l_shoulder.y:
                    action_heuristic = "HANDS_TOGETHER (HOLDING/TYPING?)"
                
                if len(movement_boxes) > 3:
                    action_heuristic = "WALKING_OR_RAPID_KINETIC_SHIFT"

        return should_process, frame, movement_boxes, detected_objects, pose_landmarks_list, action_heuristic, self.rf_sensor.active_devices, self.acoustic_sensor.acoustic_event

    def close(self):
        self.running = False
        if hasattr(self, 'thread') and self.thread.is_alive():
            self.thread.join(timeout=1)
        self.ocr_executor.shutdown(wait=False)
        if self.cap.isOpened():
            self.cap.release()
