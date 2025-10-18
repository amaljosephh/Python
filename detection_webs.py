
import cv2
import numpy as np
import time
import base64
from collections import deque
import mediapipe as mp

# ========================================
# ENHANCED 3D FACE DETECTION & TRACKING
# ========================================
# This system uses MediaPipe Face Mesh for 3D facial landmark detection
# Features:
# - 3D Movement Tracking (x, y, z coordinates)
# - 3D Stability Detection (monitors depth changes)
# - 3D Face Straightness (uses depth symmetry)
# - 3D Distance Estimation (combines 2D area + depth)
# - Advanced Gaze Detection with 3D iris tracking
# - Blink Detection & Liveness Detection
# ========================================

# === Constants ===
EAR_THRESHOLD = 0.20  # STRICT: Eyes must close significantly for blink detection (lower = stricter)
MOVEMENT_THRESHOLD_XY = 10  # Easier 2D movement threshold (pixels)
MOVEMENT_THRESHOLD_Z = 0.01  # Easier 3D depth movement threshold (normalized)
STABILITY_THRESHOLD_XY = 8  # More lenient 2D stability threshold (pixels)
STABILITY_THRESHOLD_Z = 0.01  # More lenient 3D depth stability threshold (normalized)
CAPTURE_SIZE = (413, 531)
SHARPNESS_THRESHOLD = 100
REQUIRED_BLINKS = 2  # STRICT: Must blink TWICE for anti-spoofing
BLINK_WINDOW_SECONDS = 8  # STRICT: Must complete 2 blinks within 8 seconds
BLINK_DEBOUNCE_TIME = 0.7  # STRICT: Minimum time between blinks (prevents false positives)
NOSE_INDEX = 1

# === Eye Gaze Constants ===
# Gaze thresholds (lower = more strict) - VERY LENIENT for easy use
GAZE_HORIZONTAL_THRESHOLD = 0.40  # Maximum horizontal iris offset ratio (very lenient)
GAZE_VERTICAL_THRESHOLD = 0.45    # Maximum vertical iris offset ratio (very lenient)
GAZE_MAX_INDIVIDUAL_EYE = 0.45    # Maximum offset for any single eye (very lenient)
GAZE_SYMMETRY_THRESHOLD = 0.35    # Maximum difference between left and right eye gaze (lenient)
GAZE_CONSISTENCY_FRAMES = 1       # Only 1 frame needed (very easy)

# === 3D Face Scan Verification Constants ===
CIRCULAR_MOVEMENT_THRESHOLD = 5    # Very small angle variation needed (5 degrees - very easy)
MOVEMENT_FRAMES_REQUIRED = 8       # Only 8 frames (~1.6 seconds - very quick)
SCAN_TIMEOUT_SECONDS = 45          # More time allowed
MIN_3D_DEPTH_VARIATION = 0.01      # Minimal depth variation required (very lenient)

# === Mediapipe Setup ===
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=2,
    refine_landmarks=True,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.6
)

face_detection = mp_face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.8
)

LEFT_EYE = list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE = list(mp_face_mesh.FACEMESH_RIGHT_EYE)
MOUTH = list(mp_face_mesh.FACEMESH_LIPS)
NOSE = list(mp_face_mesh.FACEMESH_NOSE)

# Extract iris landmarks dynamically (available when refine_landmarks=True)
LEFT_IRIS = list(mp_face_mesh.FACEMESH_LEFT_IRIS)
RIGHT_IRIS = list(mp_face_mesh.FACEMESH_RIGHT_IRIS)

# Extract all unique iris landmark indices
LEFT_IRIS_INDICES = sorted(set(i for conn in LEFT_IRIS for i in conn))
RIGHT_IRIS_INDICES = sorted(set(i for conn in RIGHT_IRIS for i in conn))

# Iris center is the first point in the sorted indices (center point of iris)
LEFT_IRIS_CENTER = LEFT_IRIS_INDICES[0]
RIGHT_IRIS_CENTER = RIGHT_IRIS_INDICES[0]

# Extract eye corner indices from eye landmark connections
LEFT_EYE_INDICES = sorted(set(i for conn in LEFT_EYE for i in conn))
RIGHT_EYE_INDICES = sorted(set(i for conn in RIGHT_EYE for i in conn))

# === Dynamic Facial Landmark Indices ===
# These are MediaPipe Face Mesh standard landmark indices
# Reference: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png

# Eye landmarks for face orientation
LEFT_EYE_OUTER = 33   # Left eye outer corner
LEFT_EYE_INNER = 133  # Left eye inner corner
RIGHT_EYE_OUTER = 263 # Right eye outer corner
RIGHT_EYE_INNER = 362 # Right eye inner corner

# Face contour landmarks
LEFT_FACE_CONTOUR = 234   # Left face edge/contour
RIGHT_FACE_CONTOUR = 454  # Right face edge/contour
LEFT_CHEEK = 205          # Left cheek
RIGHT_CHEEK = 425         # Right cheek

# Mouth landmarks
LEFT_MOUTH_CORNER = 61    # Left corner of mouth
RIGHT_MOUTH_CORNER = 291  # Right corner of mouth

# Central facial landmarks
FOREHEAD_CENTER = 10      # Center of forehead
CHIN_CENTER = 152         # Center of chin

# Key facial points for 3D profiling
KEY_FACIAL_LANDMARKS = [
    NOSE_INDEX,           # 1: Nose tip
    LEFT_EYE_OUTER,       # 33: Left eye
    RIGHT_EYE_OUTER,      # 263: Right eye
    LEFT_MOUTH_CORNER,    # 61: Left mouth
    RIGHT_MOUTH_CORNER,   # 291: Right mouth
    LEFT_FACE_CONTOUR,    # 234: Left face edge
    RIGHT_FACE_CONTOUR,   # 454: Right face edge
    FOREHEAD_CENTER,      # 10: Forehead
    CHIN_CENTER           # 152: Chin
]

# Landmarks for distance estimation
DISTANCE_LANDMARKS = [
    NOSE_INDEX,           # Nose
    LEFT_EYE_OUTER,       # Left eye
    RIGHT_EYE_OUTER,      # Right eye
    LEFT_MOUTH_CORNER,    # Left mouth corner
    RIGHT_MOUTH_CORNER    # Right mouth corner
]

# === State ===
blink_times = deque()
prev_nose = None  # Stores 3D coordinates (x, y, z) for movement tracking
has_moved = False
captured = False
countdown_started = False
countdown_start_time = None

# === Stabilizer Buffer ===
stabilizer_buffer = deque(maxlen=5)  # Keep last 5 frames for stabilization

# === Gaze Tracking ===
gaze_history = deque(maxlen=GAZE_CONSISTENCY_FRAMES)  # Track gaze over multiple frames

# === 3D Face Scan Verification State ===
scan_mode_active = False
scan_start_time = None
scan_completed_steps = {
    'left_turn_verified': False,
    'right_turn_verified': False,
    'blink_verified': False
}
current_scan_step = 'initialize'  # Steps: initialize -> warmup -> verify_left -> verify_right -> center_blink -> finalize
movement_frame_counter = 0
frame_count = 0
face_3d_profile = []  # Store 3D landmark snapshots during scan
head_angles_history = deque(maxlen=50)  # Track head angles for circular movement

# === Active Turn Verification ===
left_turn_frames = 0    # Counter for consecutive left turn detections
right_turn_frames = 0   # Counter for consecutive right turn detections
REQUIRED_TURN_FRAMES = 4  # Need 4 consecutive frames (~0.8 seconds) - Fast verification
MIN_TURN_ANGLE = 5  # Minimum angle for turn detection (degrees) - Balanced for security & usability
MIN_TURN_CONFIDENCE = 0.5  # Minimum confidence for turn detection - STRICT
REQUIRED_MIN_VOTES = 2  # Need at least 2 voting methods to agree

# === Final Capture State ===
capture_ready_frames = 0  # Count frames where all conditions are met
REQUIRED_STABLE_FRAMES = 7  # Need 7 stable frames (~1.4 seconds) - Conservative reduction


def compute_ear(landmarks, eye_conns, w, h):
    pts = list(set(i for c in eye_conns for i in c))
    coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in pts]
    ys = [pt[1] for pt in coords]
    xs = [pt[0] for pt in coords]
    vertical = max(ys) - min(ys)
    horizontal = max(xs) - min(xs)
    return vertical / horizontal if horizontal else 0

def detect_blinks(ear, threshold):
    """
    STRICT blink detection for anti-spoofing.
    Requires REQUIRED_BLINKS blinks within BLINK_WINDOW_SECONDS.
    Each blink must be separated by at least BLINK_DEBOUNCE_TIME to prevent false positives.
    """
    now = time.time()
    if ear < threshold:
        # Only register a new blink if enough time has passed since last blink
        if not blink_times or now - blink_times[-1] > BLINK_DEBOUNCE_TIME:
            blink_times.append(now)
            print(f"👁️ BLINK detected! Count: {len(blink_times)}/{REQUIRED_BLINKS}, EAR: {ear:.3f}")
    
    # Remove old blinks outside the time window
    while blink_times and now - blink_times[0] > BLINK_WINDOW_SECONDS:
        blink_times.popleft()
    
    return len(blink_times) >= REQUIRED_BLINKS

# def is_face_straight(landmarks, w, h):
#     left_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in [33, 133]], axis=0)
#     right_eye = np.mean([(landmarks[i].x * w, landmarks[i].y * h) for i in [362, 263]], axis=0)
#     eye_diff_y = abs(left_eye[1] - right_eye[1])
#     eye_diff_x = abs(left_eye[0] - right_eye[0])
#     angle = np.arctan2(eye_diff_y, eye_diff_x) * (180 / np.pi)
#     return angle < 10

def is_face_straight(landmarks, w, h):
    """Enhanced 3D face straightness detection using depth information"""
    # Get 3D coordinates (x, y, z) for key facial landmarks - using dynamic indices
    left_eye_3d = np.mean([
        (landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z) 
        for i in [LEFT_EYE_OUTER, LEFT_EYE_INNER]
    ], axis=0)
    right_eye_3d = np.mean([
        (landmarks[i].x * w, landmarks[i].y * h, landmarks[i].z) 
        for i in [RIGHT_EYE_INNER, RIGHT_EYE_OUTER]
    ], axis=0)
    nose_3d = (landmarks[NOSE_INDEX].x * w, landmarks[NOSE_INDEX].y * h, landmarks[NOSE_INDEX].z)

    eye_center_x = (left_eye_3d[0] + right_eye_3d[0]) / 2
    nose_x = nose_3d[0]

    # Horizontal offset between nose and center of eyes (2D)
    offset = abs(nose_x - eye_center_x)

    # Angle between eyes (roll) - 2D
    eye_diff_y = abs(left_eye_3d[1] - right_eye_3d[1])
    eye_diff_x = abs(left_eye_3d[0] - right_eye_3d[0])
    angle = np.arctan2(eye_diff_y, eye_diff_x) * (180 / np.pi)

    # 3D depth symmetry check - eyes should be at similar depth when face is straight
    eye_depth_diff = abs(left_eye_3d[2] - right_eye_3d[2])
    depth_symmetry_ok = eye_depth_diff < 0.04  # More lenient depth difference

    # Enhanced 3D straightness: nose centered, eyes level, and eyes at similar depth (all very lenient)
    return offset < 40 and angle < 12 and depth_symmetry_ok


def is_looking_at_camera(landmarks, w, h):
    """
    Intelligent gaze detection with multiple validation methods.
    Checks:
    1. Horizontal iris position (left/right)
    2. Vertical iris position (up/down)
    3. 3D depth analysis (z-coordinate)
    4. Symmetry between both eyes
    5. Consistency over multiple frames
    
    Returns:
        tuple: (is_looking_straight, debug_info)
    """
    global gaze_history
    
    # Check if iris landmarks are available (requires refine_landmarks=True)
    if len(landmarks) <= max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
        return False, {"error": "Iris landmarks not available"}
    
    # Get iris centers (dynamically extracted from FACEMESH_LEFT_IRIS and FACEMESH_RIGHT_IRIS)
    left_iris = landmarks[LEFT_IRIS_CENTER]
    right_iris = landmarks[RIGHT_IRIS_CENTER]
    
    # === HORIZONTAL GAZE DETECTION (Left/Right) ===
    # Dynamically find eye corners from the eye landmark indices
    left_eye_points_x = [(landmarks[i].x * w, landmarks[i].y * h, i) for i in LEFT_EYE_INDICES]
    left_outer_x = min(left_eye_points_x, key=lambda p: p[0])[0]  # Leftmost = outer corner
    left_inner_x = max(left_eye_points_x, key=lambda p: p[0])[0]  # Rightmost = inner corner
    
    right_eye_points_x = [(landmarks[i].x * w, landmarks[i].y * h, i) for i in RIGHT_EYE_INDICES]
    right_outer_x = max(right_eye_points_x, key=lambda p: p[0])[0]  # Rightmost = outer corner
    right_inner_x = min(right_eye_points_x, key=lambda p: p[0])[0]  # Leftmost = inner corner
    
    # Get iris positions (2D)
    left_iris_x = left_iris.x * w
    left_iris_y = left_iris.y * h
    right_iris_x = right_iris.x * w
    right_iris_y = right_iris.y * h
    
    # Calculate horizontal eye center and width for each eye
    left_eye_center_x = (left_outer_x + left_inner_x) / 2
    left_eye_width = abs(left_outer_x - left_inner_x)
    
    right_eye_center_x = (right_outer_x + right_inner_x) / 2
    right_eye_width = abs(right_outer_x - right_inner_x)
    
    # Calculate horizontal iris offset from eye center (normalized by eye width)
    left_h_offset_ratio = abs(left_iris_x - left_eye_center_x) / left_eye_width if left_eye_width > 0 else 0
    right_h_offset_ratio = abs(right_iris_x - right_eye_center_x) / right_eye_width if right_eye_width > 0 else 0
    
    # === VERTICAL GAZE DETECTION (Up/Down) ===
    # Find top and bottom points of each eye
    left_eye_points_y = [(landmarks[i].y * h, i) for i in LEFT_EYE_INDICES]
    left_top_y = min(left_eye_points_y, key=lambda p: p[0])[0]     # Topmost
    left_bottom_y = max(left_eye_points_y, key=lambda p: p[0])[0]  # Bottommost
    
    right_eye_points_y = [(landmarks[i].y * h, i) for i in RIGHT_EYE_INDICES]
    right_top_y = min(right_eye_points_y, key=lambda p: p[0])[0]     # Topmost
    right_bottom_y = max(right_eye_points_y, key=lambda p: p[0])[0]  # Bottommost
    
    # Calculate vertical eye center and height for each eye
    left_eye_center_y = (left_top_y + left_bottom_y) / 2
    left_eye_height = abs(left_bottom_y - left_top_y)
    
    right_eye_center_y = (right_top_y + right_bottom_y) / 2
    right_eye_height = abs(right_bottom_y - right_top_y)
    
    # Calculate vertical iris offset from eye center (normalized by eye height)
    left_v_offset_ratio = abs(left_iris_y - left_eye_center_y) / left_eye_height if left_eye_height > 0 else 0
    right_v_offset_ratio = abs(right_iris_y - right_eye_center_y) / right_eye_height if right_eye_height > 0 else 0
    
    # === 3D DEPTH ANALYSIS (Z-coordinate) ===
    # When looking at camera, both iris z-coordinates should be similar
    left_iris_z = left_iris.z
    right_iris_z = right_iris.z
    iris_z_diff = abs(left_iris_z - right_iris_z)
    
    # Z-coordinate symmetry check (both irises at similar depth) - Made optional
    z_symmetry_ok = iris_z_diff < 0.05  # Relaxed depth difference
    
    # === SYMMETRY CHECK ===
    # When looking straight, both eyes should have similar offset ratios
    h_symmetry_diff = abs(left_h_offset_ratio - right_h_offset_ratio)
    v_symmetry_diff = abs(left_v_offset_ratio - right_v_offset_ratio)
    
    h_symmetry_ok = h_symmetry_diff < GAZE_SYMMETRY_THRESHOLD
    v_symmetry_ok = v_symmetry_diff < GAZE_SYMMETRY_THRESHOLD
    
    # === COMBINED GAZE ANALYSIS ===
    # Calculate average offsets
    avg_h_offset = (left_h_offset_ratio + right_h_offset_ratio) / 2
    avg_v_offset = (left_v_offset_ratio + right_v_offset_ratio) / 2
    
    # Both eyes must be centered independently (not just average)
    left_eye_centered = (left_h_offset_ratio < GAZE_MAX_INDIVIDUAL_EYE and 
                        left_v_offset_ratio < GAZE_MAX_INDIVIDUAL_EYE)
    right_eye_centered = (right_h_offset_ratio < GAZE_MAX_INDIVIDUAL_EYE and 
                         right_v_offset_ratio < GAZE_MAX_INDIVIDUAL_EYE)
    
    # Overall gaze check: PRIMARY conditions (horizontal + vertical + both eyes)
    horizontal_ok = avg_h_offset < GAZE_HORIZONTAL_THRESHOLD
    vertical_ok = avg_v_offset < GAZE_VERTICAL_THRESHOLD
    both_eyes_ok = left_eye_centered and right_eye_centered
    
    # OPTIONAL symmetry check - helps but not required
    symmetry_ok = h_symmetry_ok and v_symmetry_ok
    
    # Single frame check - RELAXED: only require primary conditions
    # Symmetry and z-depth are bonus but not mandatory
    frame_gaze_ok = horizontal_ok and vertical_ok and both_eyes_ok
    
    # === CONSISTENCY OVER MULTIPLE FRAMES ===
    # Add current frame result to history
    gaze_history.append(frame_gaze_ok)
    
    # Only consider gaze straight if it's been consistently good
    consistent_gaze = len(gaze_history) >= GAZE_CONSISTENCY_FRAMES and all(gaze_history)
    
    debug_info = {
        "left_h_offset": round(left_h_offset_ratio, 3),
        "right_h_offset": round(right_h_offset_ratio, 3),
        "avg_h_offset": round(avg_h_offset, 3),
        "h_threshold": GAZE_HORIZONTAL_THRESHOLD,
        "left_v_offset": round(left_v_offset_ratio, 3),
        "right_v_offset": round(right_v_offset_ratio, 3),
        "avg_v_offset": round(avg_v_offset, 3),
        "v_threshold": GAZE_VERTICAL_THRESHOLD,
        "h_symmetry_diff": round(h_symmetry_diff, 3),
        "v_symmetry_diff": round(v_symmetry_diff, 3),
        "iris_z_diff": round(iris_z_diff, 4),
        "horizontal_ok": horizontal_ok,
        "vertical_ok": vertical_ok,
        "left_eye_ok": left_eye_centered,
        "right_eye_ok": right_eye_centered,
        "both_eyes_ok": both_eyes_ok,
        "symmetry_ok": symmetry_ok,
        "z_symmetry_ok": z_symmetry_ok,
        "frame_gaze_ok": frame_gaze_ok,
        "consistent_gaze": consistent_gaze,
        "gaze_frames": f"{sum(gaze_history)}/{len(gaze_history)}"
    }
    
    return consistent_gaze, debug_info


def is_real_human(frame, landmarks, w, h):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detection_results = face_detection.process(rgb)

    if not detection_results.detections:
        return False, "No face detected by face detection model"

    detection = detection_results.detections[0]
    if detection.score[0] < 0.85:
        return False, "Low face detection confidence - possible photo"

    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x1, y1 = max(0, min(xs) - 20), max(0, min(ys) - 20)
    x2, y2 = min(w, max(xs) + 20), min(h, max(ys) + 20)

    face_crop = frame[y1:y2, x1:x2]
    if face_crop.size == 0:
        return False, "Invalid face crop"

    gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    if gray_face.shape[0] > 10 and gray_face.shape[1] > 10:
        sobel_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
        texture_variance = np.var(sobel_x) + np.var(sobel_y)

        if texture_variance < 80:
            return False, "Face appears too smooth - possible photo"

    return True, "Real human detected"

def detect_glasses_specifically(landmarks, frame, w, h, debug=False):
    """
    Production-ready glasses detection with context awareness
    Adapts sensitivity based on current verification step
    """
    import cv2
    import numpy as np
    
    # Get current scan context
    global current_scan_step
    
    # === CONTEXT-AWARE SENSITIVITY ADJUSTMENT ===
    # During head turns, use stricter thresholds to avoid false positives
    # During initial/final stages, use normal sensitivity
    
    if current_scan_step in ['verify_left', 'verify_right']:
        # STRICT MODE: Only detect very obvious glasses during turns
        edge_density_threshold = 0.20  # Much higher threshold
        min_edge_score_required = 3.0  # Need very strong evidence
        total_score_threshold = 15     # Very high bar
        enable_gradient_detection = False  # Disable sensitive detections
        enable_lens_boundary = False      # Disable to avoid turn artifacts
        min_reflections_required = 3      # Need more evidence during turns
    else:
        # NORMAL MODE: Standard sensitivity for static face
        edge_density_threshold = 0.12
        min_edge_score_required = 2.0
        total_score_threshold = 10
        enable_gradient_detection = True
        enable_lens_boundary = True
        min_reflections_required = 2
    
    # Get eye indices safely
    try:
        LEFT_EYE_IDX = list(set(i for c in mp_face_mesh.FACEMESH_LEFT_EYE for i in c))
        RIGHT_EYE_IDX = list(set(i for c in mp_face_mesh.FACEMESH_RIGHT_EYE for i in c))
    except:
        return False, {"error": "Could not access eye landmarks"} if debug else False
    
    def get_eye_region(indices, margin=20):
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        if not coords:
            return 0, 0, 0, 0
        xs, ys = zip(*coords)
        return (max(0, min(xs) - margin), 
                max(0, min(ys) - margin),
                min(w, max(xs) + margin), 
                min(h, max(ys) + margin))
    
    detection_scores = {
        'left_eye': 0,
        'right_eye': 0,
        'bridge': 0,
        'symmetry': 0
    }
    detection_details = []
    
    # Store features for symmetry check
    left_features = {}
    right_features = {}
    
    # === ENHANCED EYE ANALYSIS ===
    for eye_name, eye_indices, eye_features in [
        ("left", LEFT_EYE_IDX, left_features),
        ("right", RIGHT_EYE_IDX, right_features)
    ]:
        x1, y1, x2, y2 = get_eye_region(eye_indices, margin=35)
        if x2 <= x1 or y2 <= y1:
            continue
        
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        h_roi, w_roi = gray.shape
        
        # Store for symmetry analysis
        eye_features['mean'] = np.mean(gray)
        eye_features['std'] = np.std(gray)
        
        # === 1. MULTI-SCALE EDGE DETECTION ===
        # Use multiple scales to catch both obvious and subtle frames
        edge_scores = []
        
        # Enhance contrast for better edge detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        
        # Multiple edge detection passes
        for scale_factor in [0.5, 1.0, 1.5]:
            threshold_low = np.percentile(enhanced, 20) * scale_factor
            threshold_high = np.percentile(enhanced, 80) * scale_factor
            edges = cv2.Canny(enhanced, threshold_low, threshold_high)
            
            # Focus on frame regions
            if h_roi > 30:
                # Top region (upper frame)
                top_strip = edges[5:h_roi//3, :]
                # Bottom region (lower frame)
                bottom_strip = edges[2*h_roi//3:h_roi-5, :]
                # Side regions (frame sides)
                left_strip = edges[:, :w_roi//4]
                right_strip = edges[:, 3*w_roi//4:]
                
                # Calculate edge densities
                top_density = np.sum(top_strip > 0) / (top_strip.size + 1)
                bottom_density = np.sum(bottom_strip > 0) / (bottom_strip.size + 1)
                side_density = (np.sum(left_strip > 0) + np.sum(right_strip > 0)) / (left_strip.size + right_strip.size + 1)
                
                # CONTEXT-AWARE: Use dynamic threshold
                if top_density > edge_density_threshold or bottom_density > edge_density_threshold:
                    edge_scores.append(1)
                if side_density > edge_density_threshold * 0.7:
                    edge_scores.append(0.5)
        
        if sum(edge_scores) >= min_edge_score_required:
            detection_scores[f'{eye_name}_eye'] += 2
            detection_details.append(f"{eye_name}_frame_edges")
            eye_features['has_frame'] = True
        
        # === 2. REFLECTION AND GLARE ANALYSIS ===
        # More sophisticated reflection detection
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        v_channel = hsv[:, :, 2]  # Value channel
        
        # Dynamic threshold based on image statistics
        bright_threshold = min(250, np.percentile(v_channel, 97))
        very_bright = v_channel > bright_threshold
        
        # Analyze bright spot patterns
        if np.sum(very_bright) > 0:
            # Find connected components of bright areas
            num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
                very_bright.astype(np.uint8), connectivity=8
            )
            
            # Count valid reflections (small, concentrated bright spots)
            valid_reflections = 0
            for i in range(1, num_labels):
                area = stats[i, cv2.CC_STAT_AREA]
                if 5 < area < 150:  # Reasonable reflection size
                    valid_reflections += 1
            
            # CONTEXT-AWARE: Different requirements for turns
            if valid_reflections >= min_reflections_required:
                detection_scores[f'{eye_name}_eye'] += 1
                detection_details.append(f"{eye_name}_reflection")
                eye_features['has_reflection'] = True
        
        # === 3. GRADIENT PATTERN ANALYSIS (Only in normal mode) ===
        if enable_gradient_detection:
            # Calculate directional gradients
            dx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=5)
            dy = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=5)
            
            # Analyze gradient patterns
            gradient_magnitude = np.sqrt(dx**2 + dy**2)
            gradient_direction = np.arctan2(dy, dx)
            
            # Check for predominant horizontal lines (frame characteristic)
            horizontal_mask = np.abs(gradient_direction) < np.pi/8  # Within 22.5 degrees of horizontal
            vertical_mask = np.abs(gradient_direction - np.pi/2) < np.pi/8
            
            horizontal_strength = np.mean(gradient_magnitude[horizontal_mask]) if np.any(horizontal_mask) else 0
            vertical_strength = np.mean(gradient_magnitude[vertical_mask]) if np.any(vertical_mask) else 0
            
            eye_features['h_gradient'] = horizontal_strength
            eye_features['v_gradient'] = vertical_strength
            
            # Require strong horizontal dominance
            if horizontal_strength > vertical_strength * 1.5 and horizontal_strength > 15:
                detection_scores[f'{eye_name}_eye'] += 1
                detection_details.append(f"{eye_name}_gradient_horizontal")
        
        # === 4. LENS BOUNDARY DETECTION (Only in normal mode) ===
        if enable_lens_boundary:
            # Apply morphological operations to connect edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            gradient_magnitude = np.sqrt(dx**2 + dy**2) if enable_gradient_detection else np.zeros_like(gray)
            gradient_binary = (gradient_magnitude > np.percentile(gradient_magnitude, 80)).astype(np.uint8) if gradient_magnitude.any() else np.zeros_like(gray, dtype=np.uint8)
            closed = cv2.morphologyEx(gradient_binary, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                if len(contour) > 5:  # Need at least 5 points for ellipse
                    area = cv2.contourArea(contour)
                    if 400 < area < 4000:  # Reasonable lens size
                        # Check if contour is lens-like (circular/elliptical)
                        ellipse = cv2.fitEllipse(contour)
                        # Check aspect ratio
                        aspect_ratio = min(ellipse[1]) / max(ellipse[1]) if max(ellipse[1]) > 0 else 0
                        if aspect_ratio > 0.6:  # Circular/elliptical shape
                            detection_scores[f'{eye_name}_eye'] += 1.5
                            detection_details.append(f"{eye_name}_lens_boundary")
                            eye_features['has_lens_boundary'] = True
                            break
    
    # === SYMMETRY ANALYSIS ===
    # Glasses affect both eyes similarly
    if left_features and right_features:
        # Check feature symmetry
        if 'has_frame' in left_features and 'has_frame' in right_features:
            detection_scores['symmetry'] += 2
            detection_details.append("symmetric_frames")
        
        if 'has_reflection' in left_features and 'has_reflection' in right_features:
            detection_scores['symmetry'] += 1
            detection_details.append("symmetric_reflections")
        
        # Check gradient similarity (only in normal mode)
        if enable_gradient_detection and 'h_gradient' in left_features and 'h_gradient' in right_features:
            left_grad = left_features.get('h_gradient', 0)
            right_grad = right_features.get('h_gradient', 0)
            if left_grad > 10 and right_grad > 10:  # Both must have significant gradient
                gradient_diff = abs(left_grad - right_grad)
                gradient_avg = (left_grad + right_grad) / 2
                if gradient_avg > 0 and gradient_diff / gradient_avg < 0.4:  # Similar gradients
                    detection_scores['symmetry'] += 1
                    detection_details.append("symmetric_gradients")
    
    # === ENHANCED BRIDGE DETECTION ===
    NOSE_BRIDGE_TOP = 6
    NOSE_TIP = 1
    
    # Only check bridge in frontal views or with strong eye evidence during turns
    check_bridge = current_scan_step not in ['verify_left', 'verify_right'] or detection_scores.get('left_eye', 0) > 2
    
    if check_bridge:
        try:
            # Get bridge area coordinates
            bridge_top = (int(landmarks[NOSE_BRIDGE_TOP].x * w), int(landmarks[NOSE_BRIDGE_TOP].y * h))
            nose_tip = (int(landmarks[NOSE_TIP].x * w), int(landmarks[NOSE_TIP].y * h))
            
            # Expanded bridge ROI
            bridge_y1 = max(0, min(bridge_top[1], nose_tip[1]) - 20)
            bridge_y2 = min(h, max(bridge_top[1], nose_tip[1]) + 10)
            
            # Get eye positions for bridge width
            left_eye_x = get_eye_region(LEFT_EYE_IDX, margin=0)[2]
            right_eye_x = get_eye_region(RIGHT_EYE_IDX, margin=0)[0]
            
            bridge_x1 = max(0, left_eye_x - 20)
            bridge_x2 = min(w, right_eye_x + 20)
            
            if bridge_x2 > bridge_x1 and bridge_y2 > bridge_y1:
                bridge_roi = frame[bridge_y1:bridge_y2, bridge_x1:bridge_x2]
                if bridge_roi.size > 0:
                    bridge_gray = cv2.cvtColor(bridge_roi, cv2.COLOR_BGR2GRAY)
                    
                    # Multi-scale bridge detection
                    bridge_detected = False
                    
                    for scale in [0.7, 1.0, 1.3]:
                        threshold = np.percentile(bridge_gray, 50) * scale
                        edges = cv2.Canny(bridge_gray, threshold/2, threshold)
                        
                        # Look for horizontal continuity
                        horizontal_projection = np.sum(edges, axis=1)
                        # Stricter threshold during turns
                        min_projection = bridge_roi.shape[1] * (0.5 if current_scan_step in ['verify_left', 'verify_right'] else 0.4)
                        if np.max(horizontal_projection) > min_projection:
                            bridge_detected = True
                            break
                    
                    if bridge_detected:
                        detection_scores['bridge'] = 3
                        detection_details.append("nose_bridge")
        except:
            pass
    
    # === FINAL DECISION LOGIC - CONTEXT AWARE ===
    total_score = sum(detection_scores.values())
    
    # Calculate confidence based on evidence distribution
    confidence = 0.0
    has_glasses = False
    
    # Use dynamic threshold based on context
    if total_score >= total_score_threshold:
        has_glasses = True
        confidence = min(1.0, total_score / (total_score_threshold + 4))
    # Moderate detection with good symmetry
    elif detection_scores['symmetry'] >= 3 and total_score >= (total_score_threshold - 2):
        has_glasses = True
        confidence = min(0.8, total_score / total_score_threshold)
    # Bridge + eyes detection (only in normal mode)
    elif current_scan_step not in ['verify_left', 'verify_right']:
        if detection_scores['bridge'] >= 3 and (detection_scores['left_eye'] >= 2 and detection_scores['right_eye'] >= 2):
            has_glasses = True
            confidence = 0.7
    
    # Determine glasses type based on detection patterns
    glasses_type = "none"
    if has_glasses:
        if "lens_boundary" in str(detection_details):
            glasses_type = "transparent_or_frameless"
        elif "frame_edges" in str(detection_details):
            glasses_type = "framed_glasses"
        elif "reflection" in str(detection_details) and confidence > 0.7:
            glasses_type = "reflective_glasses"
        elif "gradient_horizontal" in str(detection_details):
            glasses_type = "possible_transparent"
        else:
            glasses_type = "eyewear_detected"
    
    if debug:
        return has_glasses, {
            "has_glasses": has_glasses,
            "glasses_type": glasses_type,
            "confidence": round(confidence, 2),
            "total_score": round(total_score, 2),
            "detection_scores": {k: round(v, 2) for k, v in detection_scores.items()},
            "detection_details": detection_details,
            "scan_context": current_scan_step,
            "mode": "strict" if current_scan_step in ['verify_left', 'verify_right'] else "normal",
            "thresholds": {
                "edge_density": edge_density_threshold,
                "min_edge_score": min_edge_score_required,
                "total_score": total_score_threshold
            }
        }
    
    return has_glasses

# Optional: Add temporal consistency check to reduce false positives
class GlassesDetectionBuffer:
    def __init__(self, buffer_size=5, confirmation_threshold=3):
        """
        Buffer to track glasses detection over multiple frames
        buffer_size: Number of frames to track
        confirmation_threshold: Number of positive detections needed
        """
        self.buffer = []
        self.buffer_size = buffer_size
        self.confirmation_threshold = confirmation_threshold
    
    def add_detection(self, has_glasses):
        """Add a detection result to the buffer"""
        self.buffer.append(has_glasses)
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)
    
    def is_confirmed(self):
        """Check if glasses are confirmed based on multiple frames"""
        if len(self.buffer) < self.confirmation_threshold:
            return False
        return sum(self.buffer) >= self.confirmation_threshold
    
    def reset(self):
        """Reset the buffer"""
        self.buffer = []

# Initialize buffer (add this at the top of your file with other state variables)
glasses_detection_buffer = GlassesDetectionBuffer(buffer_size=5, confirmation_threshold=3)

# Modified integration for main processing function
def check_glasses_with_confirmation(lm, frame, w, h):
    """
    Check for glasses with temporal consistency to avoid false positives
    """
    global glasses_detection_buffer
    
    # Get current frame detection
    has_glasses, debug_info = detect_glasses_specifically(lm, frame, w, h, debug=True)
    
    # Add to buffer
    glasses_detection_buffer.add_detection(has_glasses)
    
    # Check if confirmed over multiple frames
    confirmed_glasses = glasses_detection_buffer.is_confirmed()
    
    if confirmed_glasses:
        confidence = debug_info.get('confidence', 0)
        glasses_type = debug_info.get('glasses_type', 'unknown')
        
        if glasses_type == "sunglasses":
            message = "🚫 Sunglasses detected! Please remove your sunglasses."
        elif confidence > 0.7:
            message = "🚫 Glasses detected! Please remove your glasses for verification."
        else:
            message = "⚠️ Possible eyewear detected. Please ensure all glasses are removed."
        
        return True, message, debug_info
    
    return False, "", debug_info

def detect_face_obstruction(landmarks, frame, w, h, debug=False):
    from statistics import mean

    def get_region_box(indices, margin=15):
        coords = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]
        xs, ys = zip(*coords)
        x1 = max(0, min(xs) - margin)
        y1 = max(0, min(ys) - margin)
        x2 = min(w, max(xs) + margin)
        y2 = min(h, max(ys) + margin)
        return x1, y1, x2, y2

    LEFT_EYE_IDX = list(set(i for c in mp_face_mesh.FACEMESH_LEFT_EYE for i in c))
    RIGHT_EYE_IDX = list(set(i for c in mp_face_mesh.FACEMESH_RIGHT_EYE for i in c))
    NOSE_IDX = list(set(i for c in mp_face_mesh.FACEMESH_LIPS for i in c))
    MOUTH_IDX = list(set(i for c in mp_face_mesh.FACEMESH_LIPS for i in c))

    regions = {
        "left_eye": LEFT_EYE_IDX,
        "right_eye": RIGHT_EYE_IDX,
        "nose": NOSE_IDX,
        "mouth": MOUTH_IDX,
    }

    obstruction_score = 0
    debug_info = {}

    for region_name, idxs in regions.items():
        x1, y1, x2, y2 = get_region_box(idxs, margin=12)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, 70, 180)  # Balanced edge detection
        edge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
        if edge_density > 0.25:  # Balanced threshold
            obstruction_score += 1
            debug_info[f"{region_name}_edge_density"] = edge_density

        brightness_std = np.std(gray)
        if brightness_std > 35:  # Balanced threshold
            obstruction_score += 1
            debug_info[f"{region_name}_brightness_std"] = brightness_std

        if region_name in ["left_eye", "right_eye"]:
            h_, w_ = gray.shape
            if w_ > 20:
                half = gray[:, :w_//2], np.fliplr(gray[:, w_//2:])
                try:
                    corr = np.corrcoef(half[0].flatten(), half[1].flatten())[0, 1]
                    if not np.isnan(corr) and corr > 0.85:  # Balanced threshold
                        obstruction_score += 1
                        debug_info[f"{region_name}_symmetry"] = corr
                except:
                    pass

    has_obstruction = obstruction_score >= 3  # Balanced: Reject only significant obstructions
    if debug:
        debug_info.update({"score": obstruction_score, "final_decision": has_obstruction})
        return has_obstruction, debug_info

    return has_obstruction

def stabilize_frame(frame, landmarks, w, h):
    """Apply simple image stabilization using weighted averaging of recent frames"""
    global stabilizer_buffer
    
    # Store current frame with landmarks in buffer
    stabilizer_buffer.append({
        'frame': frame.copy(),
        'landmarks': landmarks,
        'timestamp': time.time()
    })
    
    # If we don't have enough frames yet, return original
    if len(stabilizer_buffer) < 3:
        return frame
    
    # Get face bounding box from current frame
    xs = [int(p.x * w) for p in landmarks]
    ys = [int(p.y * h) for p in landmarks]
    x1, y1 = max(0, min(xs) - 40), max(0, min(ys) - 60)
    x2, y2 = min(w, max(xs) + 40), min(h, max(ys) + 60)
    
    # Initialize stabilized frame
    stabilized_frame = frame.copy().astype(np.float32)
    
    # Apply weighted averaging to face region only
    total_weight = 1.0  # Current frame weight
    
    for i, buffer_item in enumerate(list(stabilizer_buffer)[:-1]):  # Exclude current frame
        old_frame = buffer_item['frame'].astype(np.float32)
        weight = 0.3 * (i + 1) / len(stabilizer_buffer)  # Give more weight to recent frames
        
        # Blend only the face region
        face_region = stabilized_frame[y1:y2, x1:x2]
        old_face_region = old_frame[y1:y2, x1:x2]
        
        # Ensure dimensions match
        if face_region.shape == old_face_region.shape:
            stabilized_frame[y1:y2, x1:x2] = cv2.addWeighted(
                face_region, 1.0, old_face_region, weight, 0
            )
            total_weight += weight
    
    # Normalize by total weight
    stabilized_frame = stabilized_frame / total_weight
    
    return stabilized_frame.astype(np.uint8)

def detect_head_turn_3d(landmarks, w, h):
    """
    Enhanced 3D head turn detection using facial landmarks and depth information.
    Returns: (turn_direction, turn_angle, confidence, debug_info)
    turn_direction: 'left', 'right', 'center'
    turn_angle: estimated rotation angle in degrees
    confidence: confidence score 0-1
    """
    # Key landmarks for 3D head orientation - using dynamic indices
    nose = landmarks[NOSE_INDEX]
    left_eye = landmarks[LEFT_EYE_OUTER]
    right_eye = landmarks[RIGHT_EYE_OUTER]
    left_face_edge = landmarks[LEFT_FACE_CONTOUR]
    right_face_edge = landmarks[RIGHT_FACE_CONTOUR]
    left_cheek = landmarks[LEFT_CHEEK]
    right_cheek = landmarks[RIGHT_CHEEK]
    
    # === Method 1: Face Asymmetry Analysis (2D + 3D) ===
    left_face_pos = np.array([left_face_edge.x * w, left_face_edge.y * h])
    right_face_pos = np.array([right_face_edge.x * w, right_face_edge.y * h])
    nose_pos = np.array([nose.x * w, nose.y * h])
    
    dist_nose_to_left = np.linalg.norm(nose_pos - left_face_pos)
    dist_nose_to_right = np.linalg.norm(nose_pos - right_face_pos)
    
    # Asymmetry ratio: > 1 means turning left, < 1 means turning right
    asymmetry_ratio = dist_nose_to_right / dist_nose_to_left if dist_nose_to_left > 0 else 1.0
    
    # === Method 2: 3D Eye Depth Analysis ===
    # When turning left, right eye moves forward (more negative z)
    # When turning right, left eye moves forward (more negative z)
    eye_depth_diff = right_eye.z - left_eye.z
    
    # === Method 3: Cheek Visibility Analysis (3D) ===
    # Check which cheek is more visible based on depth
    left_cheek_depth = left_cheek.z
    right_cheek_depth = right_cheek.z
    cheek_depth_diff = right_cheek_depth - left_cheek_depth
    
    # === Method 4: Nose Position Relative to Eyes ===
    eye_center_x = (left_eye.x + right_eye.x) * w / 2
    nose_x = nose.x * w
    face_width = abs(right_eye.x - left_eye.x) * w
    
    # Normalized nose offset (-1 to 1, where -1 is left, 1 is right)
    nose_offset_normalized = (nose_x - eye_center_x) / (face_width / 2) if face_width > 0 else 0
    
    # Estimate yaw angle in degrees
    yaw_estimate = np.arctan2(nose_offset_normalized * face_width / 4, w / 3) * (180 / np.pi)
    
    # === Combine All Methods for Robust 3D Detection ===
    turn_direction = 'center'
    turn_angle = abs(yaw_estimate)
    confidence = 0.0
    
    # Scoring system for each detection method
    left_votes = 0
    right_votes = 0
    confidence_scores = []
    
    # Vote 1: Asymmetry ratio (STRICT for anti-spoofing)
    if asymmetry_ratio > 1.12:  # Significant left turn required
        left_votes += 1
        confidence_scores.append(min((asymmetry_ratio - 1.0) * 2, 1.0))
    elif asymmetry_ratio < 0.88:  # Significant right turn required
        right_votes += 1
        confidence_scores.append(min((1.0 - asymmetry_ratio) * 2, 1.0))
    
    # Vote 2: Eye depth difference (STRICT - strong 3D indicator)
    if eye_depth_diff > 0.012:  # Clear left turn
        left_votes += 1
        confidence_scores.append(min(abs(eye_depth_diff) * 60, 1.0))
    elif eye_depth_diff < -0.012:  # Clear right turn
        right_votes += 1
        confidence_scores.append(min(abs(eye_depth_diff) * 60, 1.0))
    
    # Vote 3: Yaw estimate (STRICT - requires clear angle)
    if yaw_estimate < -5:  # Clear left rotation
        left_votes += 1
        confidence_scores.append(min(abs(yaw_estimate) / 25, 1.0))
    elif yaw_estimate > 5:  # Clear right rotation
        right_votes += 1
        confidence_scores.append(min(abs(yaw_estimate) / 25, 1.0))
    
    # Vote 4: Nose offset (STRICT - clear position change)
    if nose_offset_normalized < -0.12:  # Nose clearly moved left
        left_votes += 1
        confidence_scores.append(min(abs(nose_offset_normalized) * 3, 1.0))
    elif nose_offset_normalized > 0.12:  # Nose clearly moved right
        right_votes += 1
        confidence_scores.append(min(abs(nose_offset_normalized) * 3, 1.0))
    
    # Determine final direction based on votes (STRICT: Need multiple methods to agree)
    if left_votes >= 2:  # Need at least 2 methods to agree for anti-spoofing
        turn_direction = 'left'
        confidence = np.mean(confidence_scores) if confidence_scores else 0.5
    elif right_votes >= 2:  # Need at least 2 methods to agree for anti-spoofing
        turn_direction = 'right'
        confidence = np.mean(confidence_scores) if confidence_scores else 0.5
    else:
        turn_direction = 'center'
        confidence = 0.2
    
    turn_angle = max(abs(yaw_estimate), turn_angle)
    
    debug_info = {
        "asymmetry_ratio": round(asymmetry_ratio, 3),
        "eye_depth_diff": round(eye_depth_diff, 4),
        "cheek_depth_diff": round(cheek_depth_diff, 4),
        "yaw_estimate": round(yaw_estimate, 2),
        "nose_offset_norm": round(nose_offset_normalized, 3),
        "turn_angle": round(turn_angle, 2),
        "turn_direction": turn_direction,
        "confidence": round(confidence, 3),
        "left_votes": left_votes,
        "right_votes": right_votes
    }
    
    return turn_direction, turn_angle, confidence, debug_info


def capture_3d_face_snapshot(landmarks, w, h):
    """Capture a 3D snapshot of key facial landmarks for profile building"""
    # Use dynamically defined key facial landmarks
    snapshot = {
        'timestamp': time.time(),
        'landmarks_3d': []
    }
    
    for idx in KEY_FACIAL_LANDMARKS:
        lm = landmarks[idx]
        snapshot['landmarks_3d'].append({
            'index': idx,
            'x': lm.x * w,
            'y': lm.y * h,
            'z': lm.z
        })
    
    return snapshot


def verify_3d_scan_completeness(face_3d_profile):
    """
    Verify that the captured 3D profile has sufficient depth variation
    to confirm real 3D movement (not a static photo)
    """
    if len(face_3d_profile) < 3:
        return False, "Insufficient 3D snapshots"
    
    # Extract z-coordinates for nose across all snapshots
    nose_z_values = [snapshot['landmarks_3d'][0]['z'] for snapshot in face_3d_profile if snapshot['landmarks_3d']]
    
    if len(nose_z_values) < 3:
        return False, "Not enough depth data"
    
    # Calculate z-coordinate variation
    z_variation = max(nose_z_values) - min(nose_z_values)
    
    if z_variation < MIN_3D_DEPTH_VARIATION:
        return False, f"Insufficient 3D movement (depth variation: {z_variation:.4f})"
    
    return True, f"Valid 3D scan (depth variation: {z_variation:.4f})"


def detect_circular_movement(landmarks, w, h):
    """
    Detect circular head movement by tracking angle variations.
    Returns: (has_movement, angle, angle_variation, debug_info)
    """
    global head_angles_history
    
    # Get current head angle using the same method as head turn detection
    turn_direction, turn_angle, confidence, turn_debug = detect_head_turn_3d(landmarks, w, h)
    
    # Store angle in history
    head_angles_history.append(turn_angle)
    
    # Need enough history to detect movement (reduced from 10 to 5 - very quick)
    if len(head_angles_history) < 5:
        return False, turn_angle, 0, {"status": "collecting_data", "frames": len(head_angles_history)}
    
    # Calculate angle variation (range of angles)
    angle_variation = max(head_angles_history) - min(head_angles_history)
    
    # Check if there's sufficient variation (person moved head around)
    has_sufficient_movement = angle_variation >= CIRCULAR_MOVEMENT_THRESHOLD
    
    debug_info = {
        "current_angle": round(turn_angle, 2),
        "angle_variation": round(angle_variation, 2),
        "threshold": CIRCULAR_MOVEMENT_THRESHOLD,
        "has_movement": has_sufficient_movement,
        "frames_tracked": len(head_angles_history)
    }
    
    return has_sufficient_movement, turn_angle, angle_variation, debug_info


def reset_all_states():
    global blink_times, prev_nose, has_moved, captured, countdown_started, countdown_start_time, stabilizer_buffer, gaze_history
    global scan_mode_active, scan_start_time, scan_completed_steps, current_scan_step, movement_frame_counter, face_3d_profile, frame_count, head_angles_history
    global capture_ready_frames, left_turn_frames, right_turn_frames, glasses_detection_buffer, frame_count
    
    blink_times.clear()
    prev_nose = None  # Will store (x, y, z) in 3D
    has_moved = False
    captured = False
    countdown_started = False
    countdown_start_time = None
    stabilizer_buffer.clear()
    gaze_history.clear()
    
    # Reset 3D scan states
    scan_mode_active = False
    scan_start_time = None
    scan_completed_steps = {
        'left_turn_verified': False,
        'right_turn_verified': False,
        'blink_verified': False
    }
    current_scan_step = 'initialize'
    movement_frame_counter = 0
    face_3d_profile.clear()
    head_angles_history.clear()
    
    # Reset capture ready counter
    capture_ready_frames = 0
    
    # Reset turn verification counters
    left_turn_frames = 0
    right_turn_frames = 0
    
    # Reset frame counter
    frame_count = 0
    
    # Reset glasses detection buffer
    glasses_detection_buffer.reset()

# def process_frame_and_get_status(frame):
#     global prev_nose, has_moved, captured

#     h, w = frame.shape[:2]
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = face_mesh.process(rgb)

#     if not results.multi_face_landmarks:
#         return {"status": "❌ No face detected", "color": "red"}

#     if len(results.multi_face_landmarks) > 1:
#         reset_all_states()
#         return {"status": "❌ Multiple faces detected - only one person allowed", "color": "red"}

#     face = results.multi_face_landmarks[0]
#     lm = face.landmark

#     xs = [int(p.x * w) for p in lm]
#     ys = [int(p.y * h) for p in lm]
#     x1, y1 = max(0, min(xs) - 40), max(0, min(ys) - 60)
#     x2, y2 = min(w, max(xs) + 40), min(h, max(ys) + 60)
#     crop = frame[y1:y2, x1:x2]

#     is_human, reason = is_real_human(frame, lm, w, h)
#     if not is_human:
#         reset_all_states()
#         return {"status": f"🚫 Not a real human: {reason}", "color": "red"}

#     left_ear = compute_ear(lm, LEFT_EYE, w, h)
#     right_ear = compute_ear(lm, RIGHT_EYE, w, h)
#     ear = (left_ear + right_ear) / 2.0
#     eye_open = ear > EAR_THRESHOLD
#     blink_ok = detect_blinks(ear, EAR_THRESHOLD)

#     nose = lm[NOSE_INDEX]
#     nose_xy = (int(nose.x * w), int(nose.y * h))
#     dx = dy = 0
#     if prev_nose:
#         dx = abs(nose_xy[0] - prev_nose[0])
#         dy = abs(nose_xy[1] - prev_nose[1])
#     prev_nose = nose_xy

#     if not has_moved and (dx > MOVEMENT_THRESHOLD or dy > MOVEMENT_THRESHOLD):
#         has_moved = True

#     stable = dx < STABILITY_THRESHOLD and dy < STABILITY_THRESHOLD
#     straight_ok = is_face_straight(lm, w, h)

#     frame_valid = blink_ok and has_moved  and eye_open and straight_ok

#     if frame_valid and not captured:
#         # Apply stabilization to get the best possible final image
#         stabilized_frame = stabilize_frame(frame, lm, w, h)
        
#         # Extract final crop from stabilized frame
#         final_crop = stabilized_frame[y1:y2, x1:x2]
        
#         return {
#             "status": "✅ Photo captured",
#             "color": "green",
#             "captured": True,
#             "final_crop": final_crop,
#             "should_reset": True
#         }

#     return {
#         "status": f"🟡 Waiting | Blink={blink_ok} Moved={has_moved} EyeOpen={eye_open} Straight={straight_ok}",
#         "color": "yellow"
#     }




def process_frame_and_get_status(frame):
    """
    Main processing function with simplified 3D face scan verification.
    
    Workflow:
    1. Initialize: Detect face and check if human
    2. Circular Movement: User moves head in any direction (circular motion)
    3. Center & Blink: Return to center, look at camera, and blink
    4. Capture: Take final photo with all validations
    """
    global prev_nose, has_moved, captured
    global scan_mode_active, scan_start_time, scan_completed_steps, current_scan_step, movement_frame_counter, face_3d_profile, frame_count

    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if not results.multi_face_landmarks:
        return {"status": "❌ No face detected", "color": "red", "scan_step": current_scan_step}

    if len(results.multi_face_landmarks) > 1:
        reset_all_states()
        return {"status": "❌ Multiple faces detected - only one person allowed", "color": "red", "scan_step": "error"}

    face = results.multi_face_landmarks[0]
    lm = face.landmark
    
    # Increment frame counter
    frame_count += 1

    # Check for glasses with enhanced multi-level detection
    has_glasses, glasses_debug = detect_glasses_specifically(lm, frame, w, h, debug=True)
    
    # Debug logging for glasses detection
    if has_glasses:
        glasses_type = glasses_debug.get('glasses_type', 'unknown')
        confidence = glasses_debug.get('confidence', 0)
        total_score = glasses_debug.get('total_score', 0)
        scores = glasses_debug.get('detection_scores', {})
        mode = glasses_debug.get('mode', 'normal')
        thresholds = glasses_debug.get('thresholds', {})
        
        print(f"🚫 {glasses_type.upper()} DETECTED! [{mode.upper()}] Score:{total_score:.1f}/{thresholds.get('total_score', 10)} [L:{scores.get('left_eye', 0):.1f} R:{scores.get('right_eye', 0):.1f} B:{scores.get('bridge', 0):.1f} S:{scores.get('symmetry', 0):.1f}] Conf:{confidence:.2f}")
        
        return {
            "status": f"🚫 {glasses_type.replace('_', ' ').title()} detected! Please remove your glasses for verification.",
            "color": "red",
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps,
            "glasses_debug": glasses_debug
        }
    else:
        # Log when no glasses detected (every 30 frames to avoid spam)
        if frame_count % 30 == 0:
            total_score = glasses_debug.get('total_score', 0)
            scores = glasses_debug.get('detection_scores', {})
            mode = glasses_debug.get('mode', 'normal')
            thresholds = glasses_debug.get('thresholds', {})
            print(f"✅ No eyewear [{mode.upper()}]. Score:{total_score:.1f}/{thresholds.get('total_score', 10)} [L:{scores.get('left_eye', 0):.1f} R:{scores.get('right_eye', 0):.1f}]")
    
    # Check for general face obstruction
    iobstructed, obs_debug = detect_face_obstruction(lm, frame, w, h, debug=True)
    if iobstructed:
        return {
            "status": "🚫 Face is not clear. Remove any glasses, hands, or objects covering your face.",
            "color": "red",
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps,
            "obstruction_debug": obs_debug
        }

    xs = [int(p.x * w) for p in lm]
    ys = [int(p.y * h) for p in lm]
    x1, y1 = max(0, min(xs) - 40), max(0, min(ys) - 60)
    x2, y2 = min(w, max(xs) + 40), min(h, max(ys) + 60)
    crop = frame[y1:y2, x1:x2]

    # --- Enhanced 3D Face distance estimation ---
    face_area = (x2 - x1) * (y2 - y1)
    frame_area = w * h
    face_ratio = face_area / frame_area
    
    # Use 3D depth information for more accurate distance estimation - dynamic landmarks
    avg_face_depth = np.mean([lm[i].z for i in DISTANCE_LANDMARKS])
    
    # Combined 2D area + 3D depth analysis for distance feedback (VERY LENIENT)
    too_close = face_ratio > 0.45 or avg_face_depth < -0.08
    too_far = face_ratio < 0.15 or avg_face_depth > 0.08

    # Thresholds for distance feedback
    if too_close:
        return {
            "status": "👃 Too close to camera. Please move back.",
            "color": "orange",
            "depth_info": {"avg_depth": round(avg_face_depth, 4), "face_ratio": round(face_ratio, 3)},
            "scan_step": current_scan_step
        }
    elif too_far:
        return {
            "status": "📏 Too far from camera. Please come closer.",
            "color": "orange",
            "depth_info": {"avg_depth": round(avg_face_depth, 4), "face_ratio": round(face_ratio, 3)},
            "scan_step": current_scan_step
        }

    # ========================================
    # STEP 1: INITIALIZE 3D SCAN MODE
    # ========================================
    if not scan_mode_active:
        # Check for liveness before starting scan
        is_human, reason = is_real_human(frame, lm, w, h)
        if not is_human:
            reset_all_states()
            return {"status": f"🚫 Not a real human: {reason}", "color": "red", "scan_step": "failed"}
        
        # Initialize scan mode
        scan_mode_active = True
        scan_start_time = time.time()
        current_scan_step = 'warmup'
        
        return {
            "status": "🎯 3D Scan Started! Get ready...",
            "color": "cyan",
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps
        }
    
    # Check for timeout
    if time.time() - scan_start_time > SCAN_TIMEOUT_SECONDS:
        reset_all_states()
        return {
            "status": "⏰ Scan timeout. Please start again.",
            "color": "red",
            "scan_step": "timeout"
        }
    
    # ========================================
    # STEP 2: WARMUP PHASE (0.2 seconds)
    # ========================================
    if current_scan_step == 'warmup':
        movement_frame_counter += 1
        
        # Warmup for 1 frame (~0.2 second) - Conservative reduction
        if movement_frame_counter < 1:
            return {
                "status": "🔄 Keep moving your head slightly...",
                "color": "yellow",
                "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps
            }
        else:
            # Move to LEFT turn verification (camera's left = user's right)
            current_scan_step = 'verify_left'
            return {
                "status": "👉 Please turn your head to the RIGHT (your right)",
                "color": "yellow",
                "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps
            }
    
    # ========================================
    # STEP 3: VERIFY LEFT TURN
    # ========================================
    elif current_scan_step == 'verify_left':
        global left_turn_frames
        
        # Detect head turn direction
        turn_direction, current_angle, confidence, turn_debug = detect_head_turn_3d(lm, w, h)
        
        # Capture 3D snapshots
        if movement_frame_counter % 2 == 0:
            snapshot = capture_3d_face_snapshot(lm, w, h)
            face_3d_profile.append(snapshot)
        
        movement_frame_counter += 1
        
        # STRICT: Check if user is ACTUALLY turning LEFT (anti-spoofing measures)
        is_turning_left = (
            turn_direction == 'left' and 
            current_angle >= MIN_TURN_ANGLE and 
            confidence >= MIN_TURN_CONFIDENCE and
            turn_debug.get('left_votes', 0) >= REQUIRED_MIN_VOTES  # Need multiple methods to agree
        )
        
        if is_turning_left:
            left_turn_frames += 1
            print(f"✓ LEFT VERIFIED! Frames: {left_turn_frames}/{REQUIRED_TURN_FRAMES}, Angle: {current_angle:.1f}°, Confidence: {confidence:.2f}, Votes: {turn_debug.get('left_votes', 0)}")
        else:
            if left_turn_frames > 0:
                print(f"✗ LEFT FAILED. Dir: {turn_direction}, Angle: {current_angle:.1f}° (need {MIN_TURN_ANGLE}°), Conf: {confidence:.2f} (need {MIN_TURN_CONFIDENCE}), Votes: {turn_debug.get('left_votes', 0)}/{REQUIRED_MIN_VOTES}")
            left_turn_frames = 0  # Hard reset for strict verification
        
        # LEFT turn verified (camera's left = user's right)!
        if left_turn_frames >= REQUIRED_TURN_FRAMES:
            scan_completed_steps['left_turn_verified'] = True
            current_scan_step = 'verify_right'
            left_turn_frames = 0  # Reset counter
            
            return {
                "status": "✅ Right turn verified! Now turn your head to the LEFT 👈 (your left)",
                "color": "green",
                "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps,
                "turn_debug": turn_debug
            }
        
        # Still waiting for LEFT turn - Show helpful feedback (user's perspective = RIGHT)
        progress = int((left_turn_frames / REQUIRED_TURN_FRAMES) * 100)
        
        # Get 3D depth info to show user
        left_eye_depth = lm[LEFT_EYE_OUTER].z
        right_eye_depth = lm[RIGHT_EYE_OUTER].z
        depth_diff = right_eye_depth - left_eye_depth
        
        if turn_direction == 'left':
            status_msg = f"👉 Good! Turning RIGHT... {progress}% | 3D Depth: {depth_diff:.3f}"
            color = "cyan" 
        elif turn_direction == 'right':
            status_msg = "👉 Turn to your RIGHT (you're going the wrong way)"
            color = "yellow"
        else:
            status_msg = f"👉 Please turn your head to the RIGHT (your right)"
            color = "yellow"
        
        return {
            "status": status_msg,
            "color": color,
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps,
            "turn_debug": turn_debug,
            "depth_3d": {
                "left_eye_z": round(left_eye_depth, 4),
                "right_eye_z": round(right_eye_depth, 4),
                "depth_diff": round(depth_diff, 4)
            }
        }
    
    # ========================================
    # STEP 4: VERIFY RIGHT TURN
    # ========================================
    elif current_scan_step == 'verify_right':
        global right_turn_frames
        
        # Detect head turn direction
        turn_direction, current_angle, confidence, turn_debug = detect_head_turn_3d(lm, w, h)
        
        # Capture 3D snapshots
        if movement_frame_counter % 2 == 0:
            snapshot = capture_3d_face_snapshot(lm, w, h)
            face_3d_profile.append(snapshot)
        
        movement_frame_counter += 1
        
        # STRICT: Check if user is ACTUALLY turning RIGHT (anti-spoofing measures)
        is_turning_right = (
            turn_direction == 'right' and 
            current_angle >= MIN_TURN_ANGLE and 
            confidence >= MIN_TURN_CONFIDENCE and
            turn_debug.get('right_votes', 0) >= REQUIRED_MIN_VOTES  # Need multiple methods to agree
        )
        
        if is_turning_right:
            right_turn_frames += 1
            print(f"✓ RIGHT VERIFIED! Frames: {right_turn_frames}/{REQUIRED_TURN_FRAMES}, Angle: {current_angle:.1f}°, Confidence: {confidence:.2f}, Votes: {turn_debug.get('right_votes', 0)}")
        else:
            if right_turn_frames > 0:
                print(f"✗ RIGHT FAILED. Dir: {turn_direction}, Angle: {current_angle:.1f}° (need {MIN_TURN_ANGLE}°), Conf: {confidence:.2f} (need {MIN_TURN_CONFIDENCE}), Votes: {turn_debug.get('right_votes', 0)}/{REQUIRED_MIN_VOTES}")
            right_turn_frames = 0  # Hard reset for strict verification
        
        # RIGHT turn verified (camera's right = user's left)!
        if right_turn_frames >= REQUIRED_TURN_FRAMES:
            scan_completed_steps['right_turn_verified'] = True
            current_scan_step = 'center_blink'
            right_turn_frames = 0  # Reset counter
            
            return {
                "status": "✅ Left turn verified! Perfect! Now look STRAIGHT at camera and BLINK TWICE 👁️👁️",
                "color": "green",
                "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps,
                "turn_debug": turn_debug
            }
        
        # Still waiting for RIGHT turn - Show helpful feedback (user's perspective = LEFT)
        progress = int((right_turn_frames / REQUIRED_TURN_FRAMES) * 100)
        
        # Get 3D depth info to show user
        left_eye_depth = lm[LEFT_EYE_OUTER].z
        right_eye_depth = lm[RIGHT_EYE_OUTER].z
        depth_diff = right_eye_depth - left_eye_depth
        
        if turn_direction == 'right':
            status_msg = f"👈 Good! Turning LEFT... {progress}% | 3D Depth: {depth_diff:.3f}"
            color = "cyan"
        elif turn_direction == 'left':
            status_msg = "👈 Turn to your LEFT (you're going the wrong way)"
            color = "yellow"
        else:
            status_msg = f"👈 Please turn your head to the LEFT (your left)"
            color = "yellow"
        
        return {
            "status": status_msg,
            "color": color,
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps,
            "turn_debug": turn_debug,
            "depth_3d": {
                "left_eye_z": round(left_eye_depth, 4),
                "right_eye_z": round(right_eye_depth, 4),
                "depth_diff": round(depth_diff, 4)
            }
        }
    
    # ========================================
    # STEP 5: CENTER, STRAIGHT, GAZE, AND BLINK
    # ========================================
    elif current_scan_step == 'center_blink':
        # Verify 3D scan completeness
        scan_valid, scan_message = verify_3d_scan_completeness(face_3d_profile)
        
        if not scan_valid:
            reset_all_states()
            return {
                "status": f"❌ 3D Scan failed: {scan_message}. Please try again.",
                "color": "red",
                "scan_step": "failed"
            }
        
        # Check all center conditions
        left_ear = compute_ear(lm, LEFT_EYE, w, h)
        right_ear = compute_ear(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        eye_open = ear > EAR_THRESHOLD
        blink_ok = detect_blinks(ear, EAR_THRESHOLD)
        
        straight_ok = is_face_straight(lm, w, h)
        gaze_ok, gaze_debug = is_looking_at_camera(lm, w, h)
        
        
        # Check if face is centered (straight)
        turn_direction, _, _, _ = detect_head_turn_3d(lm, w, h)
        
        # Capture center snapshot when face is straight
        if turn_direction == 'center' and straight_ok:
            snapshot = capture_3d_face_snapshot(lm, w, h)
            face_3d_profile.append(snapshot)
        
        # Check for blink
        if blink_ok and straight_ok and gaze_ok:
            scan_completed_steps['blink_verified'] = True
            current_scan_step = 'finalize'
            
            return {
                "status": "✅ 3D Scan Complete! Finalizing capture...",
                "color": "green",
                "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps,
                "scan_info": scan_message
            }
        
        # Provide simple, friendly feedback with blink count
        blink_count = len(blink_times)
        
        if not blink_ok:
            message = f"👁️ Please blink TWICE ({blink_count}/2 blinks detected)"
        elif not straight_ok:
            message = "📷 Keep your face STRAIGHT and blink"
        elif not gaze_ok:
            message = "👀 Look DIRECTLY at camera and blink"
        else:
            message = f"👁️ Perfect! Blink again ({blink_count}/2)"
        
        return {
            "status": message,
            "color": "yellow",
            "scan_step": current_scan_step,
            "scan_progress": scan_completed_steps,
            "gaze_debug": gaze_debug,
            "conditions": {"straight": straight_ok, "gaze": gaze_ok, "blink": blink_ok, "eye_open": eye_open}
        }
    
    # ========================================
    # STEP 6: FINALIZE AND CAPTURE
    # ========================================
    elif current_scan_step == 'finalize':
        global capture_ready_frames, captured
        
        # Check all final conditions - STRICT GAZE VERIFICATION
        left_ear = compute_ear(lm, LEFT_EYE, w, h)
        right_ear = compute_ear(lm, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        eye_open = ear > EAR_THRESHOLD
        
        straight_ok = is_face_straight(lm, w, h)
        gaze_ok, gaze_debug = is_looking_at_camera(lm, w, h)  # CRITICAL: Must be looking straight at camera
        
        
        # ALL conditions must be met - ESPECIALLY GAZE (looking straight at camera)
        all_conditions_met = eye_open and straight_ok and gaze_ok
        
        if all_conditions_met:
            capture_ready_frames += 1
            
            # Quick stability check (10 frames = ~2 seconds)
            if capture_ready_frames < REQUIRED_STABLE_FRAMES:
                progress = int((capture_ready_frames / REQUIRED_STABLE_FRAMES) * 100)
                return {
                    "status": f"👀 Perfect! Look straight at camera... {progress}%",
                    "color": "cyan",
                    "scan_step": current_scan_step,
                    "scan_progress": scan_completed_steps
                }
            
            # CAPTURE! (No countdown - immediate capture after stability)
            elif not captured:
                # Apply stabilization to get the best possible final image
                stabilized_frame = stabilize_frame(frame, lm, w, h)
                final_crop = stabilized_frame[y1:y2, x1:x2]
                
                # Final 3D scan verification
                scan_valid, scan_message = verify_3d_scan_completeness(face_3d_profile)
                        
                captured = True
            
            return {
                "status": "✅ 3D Verified Human! Photo captured",
                "color": "green",
                "captured": True,
                "final_crop": final_crop,
                "should_reset": True,
                "scan_completed": True,
                "scan_info": scan_message,
                "scan_progress": scan_completed_steps,
                "profile_snapshots": len(face_3d_profile)
            }
        
        else:
            # Conditions not met - reset counter and provide SPECIFIC feedback
            capture_ready_frames = 0
            
            if not eye_open:
                status_msg = "👁️ Please keep your eyes OPEN"
            elif not straight_ok:
                status_msg = "📷 Keep your face STRAIGHT - don't tilt"
            elif not gaze_ok:
                status_msg = "👀 Look DIRECTLY at the camera lens (center your eyes)"
            else:
                status_msg = "📸 Getting ready to capture..."
        
        return {
                "status": status_msg,
            "color": "yellow",
            "scan_step": current_scan_step,
                "scan_progress": scan_completed_steps,
                "gaze_debug": gaze_debug
        }
    
    # Default fallback
    return {
        "status": "🟡 Processing...",
        "color": "yellow",
        "scan_step": current_scan_step,
        "scan_progress": scan_completed_steps
    }


