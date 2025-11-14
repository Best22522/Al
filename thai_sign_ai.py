from pathlib import Path
import os
import sys
import cv2
import numpy as np
import tensorflow as tf
import pickle
from PIL import Image, ImageDraw, ImageFont

# Thai character mapping
thai_mapping = {
    'bor': 'บ',
    'dor': 'ด',
    'for': 'ฟ',
    'hor': 'ห',
    'ko': 'ก',
    'lor': 'ล',
    'mor': 'ม',
    'nor': 'น',
    'or': 'อ',
    'por': 'ป',
    'rao': 'ร',
    'sor': 'ส',
    'tor': 'ต',
    'wor': 'ว',
    'yor': 'ย'
}

try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_available = True
except Exception:
    mp = None
    mp_hands = None
    mp_available = False
    print("Note: 'mediapipe' not installed. Install with: pip install mediapipe")

try:
    import pyperclip
    have_pyperclip = True
except Exception:
    have_pyperclip = False
    print("Note: 'pyperclip' not installed. Copy to clipboard disabled. Install with: pip install pyperclip")

# Project root and dataset locations
PROJECT_ROOT = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_ROOT / "dataset"

# Mediapipe helpers
if mp_available:
    hands_detector = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    def detect_hand(image):
        if image is None:
            return None
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return hands_detector.process(image_rgb)

    def get_hand_bbox(results, image_width, image_height, margin=10):
        if results and results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                x_min = int(max(0, min(x_coords) * image_width - margin))
                y_min = int(max(0, min(y_coords) * image_height - margin))
                x_max = int(min(image_width, max(x_coords) * image_width + margin))
                y_max = int(min(image_height, max(y_coords) * image_height + margin))
                return (x_min, y_min, x_max, y_max)
        return None

def load_class_names():
    """Load class names from dataset cache"""
    train_dir = ROOT_DIR / "Training set"
    cache_path = train_dir / '_cached_data.pkl'
    
    if cache_path.exists():
        try:
            with open(cache_path, 'rb') as f:
                data_list, unique_labels = pickle.load(f)
            return unique_labels
        except Exception as e:
            print(f"Error loading cache: {e}")
    
    return []

def run_webcam_with_keras(keras_model, class_names, use_mediapipe=True, required_stable_frames=8):
    """
    Run webcam inference using a Keras/TensorFlow model.
    """
    cap = cv2.VideoCapture(0)
    last_char = None
    stable_char = None
    current_word = ""
    stable_frames = 0

    try:
        thai_font = ImageFont.truetype("C:/Users/BestyBest/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew.ttf", 40)
    except Exception:
        thai_font = ImageFont.load_default()

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    print("[INFO] Starting webcam (Press ESC to quit, R to reset, F to confirm, C to copy word, D to delete last word)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # Get bbox from mediapipe if available
        bbox = None
        if use_mediapipe and mp_available:
            results = detect_hand(frame)
            bbox = get_hand_bbox(results, w, h, margin=20)

        if bbox is None:
            # Center-crop fallback
            crop_size = min(h, w)
            x1, y1 = (w - crop_size) // 2, (h - crop_size) // 2
            x2, y2 = x1 + crop_size, y1 + crop_size
        else:
            x1, y1, x2, y2 = bbox
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            crop_size = min(h, w)
            roi = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        
        resized = cv2.resize(roi, (224, 224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (rgb - mean) / std
        inp_batch = np.expand_dims(inp, axis=0)

        preds = keras_model.predict(inp_batch, verbose=0)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        label = class_names[pred_idx]

        # Stability detection
        if label == last_char:
            stable_frames += 1
        else:
            stable_frames = 0
        last_char = label

        if stable_frames >= required_stable_frames and label not in ["No Hand", "OK"]:
            stable_char = label
            stable_frames = 0

        # Drawing overlay
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=2)
        draw.text((10, 30), f"สัญลักษณ์: {last_char}", font=thai_font, fill=(0, 255, 0))
        draw.text((10, 70), f"รอตรวจสอบ: {stable_char if stable_char else '-'}", font=thai_font, fill=(0, 255, 255))
        draw.text((10, 110), f"คำ: {current_word}", font=thai_font, fill=(255, 255, 0))

        draw.text((10, h - 50), "F=ยืนยัน | R=รีเซ็ท | C=คัดลอก | D=ลบ | ESC=ออก", font=thai_font, fill=(200, 200, 200))
        frame_out = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Thai Sign Recognition", frame_out)
        key = cv2.waitKey(3) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            current_word = ""
            stable_char = None
            print("[RESET] Word cleared")
        elif key == ord('c'):
            if current_word and have_pyperclip:
                try:
                    pyperclip.copy(current_word)
                    print(f"[COPIED] '{current_word}' copied to clipboard")
                except Exception:
                    print("Copy failed.")
        elif key == ord('f'):  # Press F to confirm
            if stable_char and stable_char in thai_mapping:
                thai_char = thai_mapping[stable_char]
                current_word += thai_char
                print(f"[CONFIRM] Added '{thai_char}' -> {current_word}")
                stable_char = None
                stable_frames = 0
            else:
                print("[INFO] No sign to confirm. Hold a sign for 8 frames first.")
        elif key == ord('d'):  # Press D to delete last character
            if current_word:
                current_word = current_word[:-1]  # Remove last character
                print(f"[DELETE] Last character removed -> {current_word}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Load class names from dataset cache
    print("Loading class names...")
    class_names = load_class_names()
    
    if len(class_names) == 0:
        print("No classes found. Exiting.")
        sys.exit(1)

    print(f"Loaded {len(class_names)} classes")

    # Load model only
    MODEL_H5 = PROJECT_ROOT / "dataset" / "model" / "best_sign_model.h5"

    if MODEL_H5.exists():
        print(f"Loading HDF5 model from {MODEL_H5}")
        model = tf.keras.models.load_model(str(MODEL_H5))
        print("Model loaded successfully!")
    else:
        print(f"Model file not found at {MODEL_H5}")
        sys.exit(1)

    # Run webcam demo
    print("\n[INFO] Starting Thai Sign Recognition Webcam Demo...")
    print("[INFO] Controls: ESC=quit, R=reset word, F=confirm, C=copy to clipboard, D=delete last character")
    try:
        run_webcam_with_keras(model, class_names, use_mediapipe=mp_available)
    except Exception as e:
        print("Webcam demo failed:", e)
        import traceback
        traceback.print_exc()