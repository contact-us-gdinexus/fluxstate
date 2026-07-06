# app.py
import time
import sys
import cv2
import os
import json
import requests
import ctypes
import numpy as np
import datetime
from collections import defaultdict, deque
from .utils.video_stream import EdgeVideoStream
from .core.state_manager import RealityGraph, IntelligenceState
from .core.engine import FluxInferenceEngine
from .core.forensics import ForensicDatabase
from .core.agent import SemanticAgent

class PhysicsTracker:
    def __init__(self, history_frames=10):
        # Maps a ByteTrack ID to a rolling queue of their last 10 coordinates
        self.history = defaultdict(lambda: deque(maxlen=history_frames))
        
    def get_smoothed_vector(self, track_id, current_centroid):
        self.history[track_id].append(current_centroid)
        pts = list(self.history[track_id])
        
        # Wait for enough frames to calculate stable momentum
        if len(pts) < 3:
            return (0, 0) 
            
        # Calculate rolling average of velocity (dx, dy)
        dx = np.mean([pts[i][0] - pts[i-1][0] for i in range(1, len(pts))])
        dy = np.mean([pts[i][1] - pts[i-1][1] for i in range(1, len(pts))])
        
        # Multiply by a scalar to predict the future position robustly
        return (dx * 10, dy * 10)

class FluxStateNode:
    """
    The core FluxState Edge Library. 
    Can be deployed headlessly on existing NVR/CCTV networks to emit JSON telemetry,
    or run locally with the Holographic UI for debugging.
    """
    def __init__(self, stream_source=None):
        import json
        try:
            import os
            policy_path = os.path.join(os.path.dirname(__file__), "intelligence_policy.json")
            with open(policy_path, "r") as f:
                self.policy = json.load(f)
        except:
            self.policy = {}
            
        if stream_source is None:
            # Fallback to policy source (RTSP IP Camera string or local USB int 0)
            ingestion = self.policy.get("INGESTION", {})
            stream_source = ingestion.get("rtsp_stream_url") if ingestion.get("rtsp_stream_url") else ingestion.get("source", 0)
            # If string is empty, default to 0
            if stream_source == "": stream_source = 0
            
        self.stream = EdgeVideoStream(stream_source)
        self.graph = RealityGraph()
        self.physics_tracker = PhysicsTracker()
        self.forensics = ForensicDatabase()
        
        # Inject the same DB instance to avoid double loading and stale caches
        self.ai_engine = FluxInferenceEngine(db=self.forensics)
        
        self.adaptive_mode = os.environ.get("ADAPTIVE_INTELLIGENCE", "False").lower() in ("true", "1")
        if self.adaptive_mode:
            from .core.agent import MLXModelPool
            from .core.state_manager import EpisodicMemoryBuffer
            self.agent = MLXModelPool(enable_adaptive=True)
            self.episodic_memory = EpisodicMemoryBuffer()
        else:
            from .core.agent import SemanticAgent
            self.agent = SemanticAgent()
            self.episodic_memory = None
            
        # Seamless SDK Integration Hooks
        self.on_threat_detected = None
        self.on_telemetry_update = None
        self.is_running = True
        
    def _push_to_integration_bus(self, telemetry):
        """
        Commercial VMS Integration: Pushes JSON telemetry to central dashboards via Webhooks.
        """
        import urllib.request
        import json
        import threading
        
        bus_config = self.policy.get("INTEGRATION_BUS", {})
        webhook_url = bus_config.get("webhook_url")
        if not webhook_url: return
            
        push_on_threat = bus_config.get("push_on_threat_only", True)
        if push_on_threat:
            context = telemetry.get("context_log", "")
            if "THREAT VECTOR" not in context:
                return
                
        def _post():
            try:
                req = urllib.request.Request(webhook_url, data=json.dumps(telemetry).encode('utf-8'), headers={'Content-Type': 'application/json'})
                urllib.request.urlopen(req, timeout=2)
            except Exception:
                pass # Fail silently to prevent edge node crash if central server goes down
                
        threading.Thread(target=_post, daemon=True).start()

    def poll_telemetry(self):
        """
        Headless API: Runs one tick of the edge inference pipeline and returns the semantic JSON telemetry.
        This is the actual production hook used to send data to the Web Command Center.
        """
        should_infer, frame, movement_boxes, detected_objects, pose_landmarks, action, rf_count, acoustic = self.stream.capture_and_filter()
        
        telemetry = {
            "timestamp": time.time(),
            "should_infer": should_infer,
            "rf_devices_nearby": rf_count,
            "acoustic_event": acoustic,
            "entities": [],
            "action_heuristic": action
        }
        
        for (x, y, bw, bh, label, conf, obj_id) in detected_objects:
            cx, cy = x + bw // 2, y + bh // 2
            dx, dy = self.physics_tracker.get_smoothed_vector(obj_id, (cx, cy))
            telemetry["entities"].append({
                "id": obj_id,
                "label": label,
                "confidence": conf,
                "position_2d": {"x": cx, "y": cy},
                "velocity_vector": {"dx": dx, "dy": dy}
            })
            
        if should_infer:
            context, vlm_reasoning, adaptive_status = self.ai_engine.generate_context_reasoning(
                detected_objects, movement_boxes, action, rf_count, acoustic, frame=frame
            )
            self.graph.log_event(context)
            telemetry["context_log"] = context
            if vlm_reasoning:
                telemetry["vlm_reasoning"] = vlm_reasoning
            
            # --- VLM AGENT INTERCEPTION ---
            # If a threat was confirmed, OR if it's legacy mode and we see THREAT VECTOR
            if adaptive_status in ("CONFIRMED", "SUPPRESSED") or "THREAT VECTOR" in context or "ANOMALY" in context.upper():
                
                # In legacy mode, we run the SemanticAgent here manually.
                if not self.adaptive_mode:
                    vlm_frame = frame.copy()
                    for (x, y, bw, bh, label, conf, obj_id) in detected_objects:
                        if "person" not in label.lower():
                            cv2.rectangle(vlm_frame, (x, y), (x + bw, y + bh), (0, 0, 255), 3)
                            cv2.putText(vlm_frame, "TARGET ANOMALY", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    prompt = (
                        "Initiate Priority Threat Assessment. FOCUS YOUR ATTENTION EXCLUSIVELY ON THE OBJECT(S) OUTLINED IN RED BOUNDING BOXES. "
                        "Perform a high-fidelity visual inspection of the pixels inside the red boundaries. "
                        "Determine if the bounded object represents a tactical threat, such as an improvised weapon, firearm, explosive, or hostile instrument. "
                        "Rule out harmless civilian electronics or everyday objects based on visual evidence. "
                        "Provide a definitive classification of the anomaly and assess the immediate tactical risk."
                    )
                    vlm_reasoning = self.agent.investigate_scene(context, vlm_frame, prompt=prompt)
                    telemetry["vlm_reasoning"] = vlm_reasoning
                    print(f"[VLM Reasoner] {vlm_reasoning}")
                    
                # Store episode in Episodic Memory for ALL detected anomalies (even if suppressed, for false negative feedback)
                if self.episodic_memory:
                    rule_triggered = "UNKNOWN_THREAT"
                    if "THREAT VECTOR:" in context:
                        rule_triggered = context.split("THREAT VECTOR:")[1].split("]")[0].strip()
                    elif adaptive_status == "SUPPRESSED":
                        rule_triggered = "ADAPTIVE_SUPPRESSED"
                        
                    event_id = self.episodic_memory.add_episode(
                        rule_triggered=rule_triggered,
                        scene_description=context,
                        telemetry=telemetry,
                        frame=frame
                    )
                    telemetry["event_id"] = event_id
                    telemetry["feedback_url"] = f"/api/v2/feedback"
                    print(f"[EpisodicMemory] Cached event {event_id} (Status: {adaptive_status}) for feedback.")
                
            self._push_to_integration_bus(telemetry)
            self.forensics.log_event(telemetry)
            
            # Fire SDK Hooks
            if self.on_threat_detected and "THREAT VECTOR" in context:
                self.on_threat_detected(telemetry)
                
        if self.on_telemetry_update:
            self.on_telemetry_update(telemetry)
            
        # --- TRUE ZERO-TRACE PRIVACY ENGINE (C-Level Wipe) ---
        # Honest Engineering: Python GC is unsafe for privacy.
        # We extract the underlying C memory pointer of the numpy array
        # and forcefully memset it to 0 before the frame is destroyed.
        if frame is not None:
            ctypes.memset(frame.ctypes.data, 0, frame.nbytes)
            
        return telemetry
        
    def start_headless_daemon(self):
        """
        Production SDK Entrypoint: Spawns the inference engine in a non-blocking daemon thread.
        Fires user-defined callbacks silently in the background.
        """
        import threading
        print("[SDK] Initializing Zero-Trace Headless Daemon...")
        
        def _loop():
            while self.is_running:
                try:
                    self.poll_telemetry()
                    time.sleep(0.01)
                except RuntimeError as e:
                    if "shutdown" in str(e).lower():
                        break # Prevent ThreadPoolExecutor crash on Ctrl+C exit
                    raise e
                except Exception:
                    pass
                
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        """
        Cleanly terminates the SDK and releases hardware resources.
        """
        self.is_running = False
        try:
            self.stream.close()
        except:
            pass

    def run_debug_ui(self):
        """
        Runs the full 3D Holographic MVP UI on the local machine for demonstration purposes.
        """
        print(f"==================================================")
        print(f" Starting FluxState Edge Node (Debug UI) ")
        print(f"==================================================")
        try:
            while True:
                should_infer, frame, movement_boxes, detected_objects, pose_landmarks, action_heuristic, rf_count, acoustic = self.stream.capture_and_filter()
                if frame is None:
                   time.sleep(5)
                   continue

                ui_frame = frame.copy()
                h, w = ui_frame.shape[:2]
                
                overlay = ui_frame.copy()
                cv2.rectangle(overlay, (0, 0), (350, h), (10, 15, 20), -1) 
                cv2.rectangle(overlay, (w - 300, 0), (w, h), (10, 15, 20), -1) 
                cv2.addWeighted(overlay, 0.7, ui_frame, 0.3, 0, ui_frame)

                cv2.putText(ui_frame, "FLUXSTATE REALITY GRAPH", (15, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(ui_frame, f"TIME: {datetime.datetime.now().strftime('%H:%M:%S.%f')[:-3]}", (15, 55), cv2.FONT_HERSHEY_PLAIN, 1.0, (200, 200, 200), 1)
                
                cv2.putText(ui_frame, "SEMANTIC TELEMETRY:", (15, 90), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 200, 0), 1)
                
                persons = [obj for obj in detected_objects if "person" in obj[4].lower()]
                cv2.putText(ui_frame, f"Human Entities: {len(persons)}", (15, 110), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 0, 255), 1)
                cv2.putText(ui_frame, f"Object Entities: {len(detected_objects) - len(persons)}", (15, 130), cv2.FONT_HERSHEY_PLAIN, 0.9, (0, 255, 255), 1)
                cv2.putText(ui_frame, f"Kinetic Deltas: {len(movement_boxes)}", (15, 150), cv2.FONT_HERSHEY_PLAIN, 0.9, (255, 100, 0), 1)
                cv2.putText(ui_frame, f"Primary Action: {action_heuristic}", (15, 170), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 255, 0), 1)
                cv2.putText(ui_frame, f"RF MAC/IP Devices: {rf_count}", (15, 190), cv2.FONT_HERSHEY_PLAIN, 0.8, (255, 255, 0), 1)
                cv2.putText(ui_frame, f"Acoustic State: {acoustic}", (15, 210), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 100, 255), 1)

                # --- 3D ISOMETRIC DIGITAL TWIN INITIALIZATION ---
                minimap_size = 250
                flat_map = np.zeros((minimap_size, minimap_size, 3), dtype=np.uint8)
                
                # Draw Hologram Grid (Glowing Cyan/Teal) on 2D floor
                grid_color = (25, 45, 25)
                for i in range(0, minimap_size, 25):
                    cv2.line(flat_map, (i, 0), (i, minimap_size), grid_color, 1)
                    cv2.line(flat_map, (0, i), (minimap_size, i), grid_color, 1)
                    
                # Draw a simulated restricted Geofenced "Red Zone"
                cv2.rectangle(flat_map, (150, 150), (230, 230), (0, 0, 50), -1)
                cv2.rectangle(flat_map, (150, 150), (230, 230), (0, 0, 255), 1)
                
                entities_to_draw = []

                # --- CENTRAL CAMERA VIEW & TRAJECTORIES ---
                for (x, y, bw, bh) in movement_boxes:
                    cv2.rectangle(ui_frame, (x, y), (x + bw, y + bh), (255, 100, 0), 1)
                    cv2.drawMarker(ui_frame, (x + bw//2, y + bh//2), (255, 100, 0), cv2.MARKER_CROSS, 10, 1)

                for (x, y, bw, bh, label, conf, obj_id) in detected_objects:
                    box_color = (255, 0, 255) if "person" in label.lower() else (0, 255, 255)
                    cv2.rectangle(ui_frame, (x, y), (x + bw, y + bh), box_color, 2)
                    cv2.putText(ui_frame, f"[{label.upper()}] {conf:.2f}", (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 0.8, box_color, 1)

                    # 2. Physics & Trajectory Engine (Anchored to Feet)
                    foot_x = x + bw // 2
                    foot_y = y + bh 
                    
                    dx, dy = self.physics_tracker.get_smoothed_vector(obj_id, (foot_x, foot_y))
                    
                    if abs(dx) > 2 or abs(dy) > 2:
                        pred_x = int(foot_x + dx) 
                        pred_y = int(foot_y + dy)
                        cv2.arrowedLine(ui_frame, (foot_x, foot_y), (pred_x, pred_y), (0, 0, 255), 2, tipLength=0.2)
                        
                    # 3. Queue entities for 3D mapping
                    map_x = int((foot_x / w) * minimap_size)
                    map_y = int((foot_y / h) * minimap_size)
                    
                    pred_map_x = int(map_x + (dx / w) * minimap_size)
                    pred_map_y = int(map_y + (dy / h) * minimap_size)
                    
                    foot_color = (0, 100, 50) if "person" in label.lower() else (0, 70, 120)
                    cv2.circle(flat_map, (map_x, map_y), 6, foot_color, -1)
                    if abs(dx) > 2 or abs(dy) > 2:
                        cv2.line(flat_map, (map_x, map_y), (pred_map_x, pred_map_y), (0, 0, 150), 1)
                        
                    physical_height = int((bh / h) * minimap_size)
                    entities_to_draw.append((map_x, map_y, label, physical_height, pred_map_x, pred_map_y))

                # --- WARP 2D FLOORPLAN INTO 3D ISOMETRIC PERSPECTIVE ---
                src_pts = np.float32([[0, 0], [minimap_size, 0], [0, minimap_size], [minimap_size, minimap_size]])
                trap_top_width = int(minimap_size * 0.4)
                trap_offset = (minimap_size - trap_top_width) // 2
                
                dst_pts = np.float32([
                    [trap_offset, 60], 
                    [trap_offset + trap_top_width, 60], 
                    [10, minimap_size - 10], 
                    [minimap_size - 10, minimap_size - 10]
                ])
                
                matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
                iso_map = cv2.warpPerspective(flat_map, matrix, (minimap_size, minimap_size))
                
                # --- PROJECT ENTITIES VERTICALLY INTO 3D SPACE ---
                if len(entities_to_draw) > 0:
                    pts_2d = np.float32([[[e[0], e[1]]] for e in entities_to_draw])
                    pts_3d_floor = cv2.perspectiveTransform(pts_2d, matrix)
                    
                    for i, (map_x, map_y, label, e_h, pred_x, pred_y) in enumerate(entities_to_draw):
                        floor_px = int(pts_3d_floor[i][0][0])
                        floor_py = int(pts_3d_floor[i][0][1])
                        
                        if "person" in label.lower():
                            map_color = (0, 255, 150)
                            marker = cv2.MARKER_SQUARE
                            txt = "HUMAN"
                        else:
                            map_color = (0, 150, 255)
                            marker = cv2.MARKER_DIAMOND
                            txt = label.split()[0].upper()[:8]
                        
                        top_px = floor_px
                        top_py = floor_py - int(e_h * 0.6) 
                        
                        cv2.line(iso_map, (floor_px, floor_py), (top_px, top_py), map_color, 2)
                        cv2.drawMarker(iso_map, (top_px, top_py), map_color, marker, 8, 2)
                        cv2.putText(iso_map, txt, (top_px + 10, top_py + 4), cv2.FONT_HERSHEY_PLAIN, 0.7, map_color, 1)
                        
                        if 150 < pred_x < 230 and 150 < pred_y < 230:
                            cv2.putText(ui_frame, "PRE-BREACH INTENT DETECTED", (350, minimap_size + 30), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 1)

                cv2.putText(iso_map, "3D ARCHITECTURAL TWIN", (5, 15), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 200), 1)

                roi = ui_frame[10:10+minimap_size, 370:370+minimap_size]
                blended = cv2.addWeighted(roi, 0.2, iso_map, 0.9, 0)
                ui_frame[10:10+minimap_size, 370:370+minimap_size] = blended

                # --- 3D SKELETAL RENDERING ---
                if pose_landmarks:
                    key_points = {
                        0: "NOSE",
                        11: "L_SHOULDER", 12: "R_SHOULDER",
                        13: "L_ELBOW", 14: "R_ELBOW",
                        15: "L_WRIST", 16: "R_WRIST",
                        23: "L_HIP", 24: "R_HIP"
                    }
                    
                    connections = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
                    for start, end in connections:
                        if start < len(pose_landmarks) and end < len(pose_landmarks):
                            pt1 = pose_landmarks[start]
                            pt2 = pose_landmarks[end]
                            if pt1.visibility > 0.3 and pt2.visibility > 0.3:
                                x1, y1 = int(pt1.x * w), int(pt1.y * h)
                                x2, y2 = int(pt2.x * w), int(pt2.y * h)
                                cv2.line(ui_frame, (x1, y1), (x2, y2), (245, 117, 66), 2)

                    for idx, label in key_points.items():
                        if idx < len(pose_landmarks):
                            lm = pose_landmarks[idx]
                            if lm.visibility > 0.3:
                                cx, cy = int(lm.x * w), int(lm.y * h)
                                cv2.circle(ui_frame, (cx, cy), 4, (245, 66, 230), -1)
                                cv2.putText(ui_frame, f"{label} ({lm.visibility:.2f})", (cx + 5, cy - 5), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), 1)

                # --- DYNAMIC REASONING ENGINE ---
                cv2.putText(ui_frame, "CONTEXTUAL EVENT LOG", (w - 285, 30), cv2.FONT_HERSHEY_TRIPLEX, 0.5, (0, 255, 255), 1)
                
                context = ""
                if not should_infer:
                    self.graph.set_state(IntelligenceState.IDLE)
                    if time.time() - self.graph.last_event_time > 10:
                        self.graph.log_event("Environment stable. No kinetic anomalies.")
                else:
                    self.graph.set_state(IntelligenceState.REASONING)
                    context, vlm_reasoning, adaptive_status = self.ai_engine.generate_context_reasoning(
                        detected_objects, movement_boxes, action_heuristic, rf_count, acoustic, frame=frame
                    )
                    self.graph.log_event(context)
                    
                # Cache telemetry for the API Server
                self.latest_telemetry = {
                    "timestamp": time.time(),
                    "should_infer": should_infer,
                    "rf_devices_nearby": rf_count,
                    "acoustic_event": acoustic,
                    "entities": [{"id": e[6], "label": e[4], "confidence": e[5]} for e in detected_objects],
                    "action_heuristic": action_heuristic,
                    "context_log": context
                }
                
                if should_infer and vlm_reasoning:
                    self.latest_telemetry["vlm_reasoning"] = vlm_reasoning
                    
                # Push active threats to Webhook VMS Integration
                if should_infer:
                    self._push_to_integration_bus(self.latest_telemetry)

                # Draw Event Log
                y_offset = 70
                for i, event in enumerate(reversed(self.graph.get_recent_events(6))):
                    text = event['summary']
                    words = text.split(' ')
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line) + len(word) < 30:
                            current_line += word + " "
                        else:
                            lines.append(current_line)
                            current_line = word + " "
                    lines.append(current_line)
                    
                    cv2.putText(ui_frame, f"> {time.strftime('%H:%M:%S', time.localtime(event['timestamp']))}", (w - 285, y_offset), cv2.FONT_HERSHEY_PLAIN, 0.8, (0, 200, 0), 1)
                    for line in lines:
                        y_offset += 15
                        cv2.putText(ui_frame, line.strip(), (w - 275, y_offset), cv2.FONT_HERSHEY_PLAIN, 0.7, (200, 200, 200), 1)
                    y_offset += 20

                # Bottom Status Bar
                cv2.rectangle(ui_frame, (0, h - 30), (w, h), (30, 30, 30), -1)
                cv2.putText(ui_frame, f"SYSTEM STATE: {self.graph.current_state.value}", (15, h - 10), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow("FluxState Edge Debug Node", ui_frame)
                if cv2.waitKey(30) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass
        finally:
            self.stream.close()
            cv2.destroyAllWindows()
            sys.exit(0)

    def start_api_server(self, port=8000):
        """
        Starts a background HTTP server on port 8000 to broadcast the Reality Graph JSON telemetry.
        This allows any Command Center Web Dashboard to hook into this Edge Node.
        """
        import json
        import jwt
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        
        node = self
        JWT_SECRET = os.environ.get("FLUX_JWT_SECRET", "flux_edge_secret_key_992")
        
        class TelemetryHandler(BaseHTTPRequestHandler):
            def _send_cors_headers(self):
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
                self.send_header('Access-Control-Allow-Headers', 'Authorization, Content-Type')

            def do_OPTIONS(self):
                self.send_response(200)
                self._send_cors_headers()
                self.end_headers()

            def _validate_jwt(self):
                auth_header = self.headers.get('Authorization')
                if not auth_header or not auth_header.startswith("Bearer "):
                    return None, "Missing or invalid Authorization header"
                token = auth_header.split(" ")[1]
                try:
                    payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
                    if payload.get("role") not in ["admin", "security_operator"]:
                        return None, "Insufficient role permissions"
                    return payload, None
                except jwt.ExpiredSignatureError:
                    return None, "Token expired"
                except jwt.InvalidTokenError:
                    return None, "Invalid token"

            def do_GET(self):
                if self.path == '/telemetry':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._send_cors_headers()
                    self.end_headers()
                    
                    data = getattr(node, 'latest_telemetry', node.poll_telemetry())
                    self.wfile.write(json.dumps(data).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def do_POST(self):
                if self.path == '/api/v2/feedback':
                    payload, error = self._validate_jwt()
                    if error:
                        self.send_response(401)
                        self.send_header('Content-type', 'application/json')
                        self._send_cors_headers()
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": error}).encode('utf-8'))
                        return

                    content_length = int(self.headers.get('Content-Length', 0))
                    try:
                        post_data = json.loads(self.rfile.read(content_length))
                    except json.JSONDecodeError:
                        self.send_response(400)
                        self.end_headers()
                        return

                    event_id = post_data.get("event_id")
                    human_label = post_data.get("human_label")
                    operator_id = post_data.get("operator_id", payload.get("sub", "unknown"))
                    operator_role = post_data.get("operator_role", payload.get("role", "operator"))
                    
                    if not event_id or not human_label:
                        self.send_response(400)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Missing event_id or human_label"}).encode('utf-8'))
                        return

                    if not node.episodic_memory:
                        self.send_response(503)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Adaptive Intelligence offline"}).encode('utf-8'))
                        return

                    episode = node.episodic_memory.get_episode(event_id)
                    if not episode:
                        self.send_response(404)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": "Event expired or not found"}).encode('utf-8'))
                        return
                        
                    # Generate embedding
                    try:
                        emb = node.agent.generate_embedding(episode["scene_description"])
                    except Exception as e:
                        self.send_response(500)
                        self.end_headers()
                        self.wfile.write(json.dumps({"error": f"Embedding failed: {str(e)}"}).encode('utf-8'))
                        return
                        
                    # Store in ContextLedger
                    node.forensics.log_feedback_event(
                        rule_triggered=episode["rule_triggered"],
                        scene_description=episode["scene_description"],
                        embedding=emb,
                        human_label=human_label,
                        operator_id=operator_id,
                        operator_role=operator_role,
                        event_id=event_id
                    )
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self._send_cors_headers()
                    self.end_headers()
                    self.wfile.write(json.dumps({"status": "success", "event_id": event_id}).encode('utf-8'))
                else:
                    self.send_response(404)
                    self.end_headers()
                    
            def log_message(self, format, *args):
                pass 
                
        from http.server import ThreadingHTTPServer
        server = ThreadingHTTPServer(('0.0.0.0', port), TelemetryHandler)
        print(f"[Network] FluxState Edge JSON API broadcasting on http://localhost:{port}/telemetry")
        threading.Thread(target=server.serve_forever, daemon=True).start()

if __name__ == "__main__":
    node = FluxStateNode()
    node.start_api_server(port=8000)
    node.run_debug_ui()
