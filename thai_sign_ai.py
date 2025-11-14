from pathlib import Path
import os
import sys
import glob
import pickle
import zipfile
import xml.etree.ElementTree as ET

# Optional imports with friendly messages
try:
    import cv2
except Exception:
    print("Missing package 'opencv-python'. Install with: pip install opencv-python")
    raise

try:
    import numpy as np
except Exception:
    print("Missing package 'numpy'. Install with: pip install numpy")
    raise

try:
    from PIL import Image, UnidentifiedImageError
except Exception:
    print("Missing package 'Pillow'. Install with: pip install Pillow")
    raise

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    from tensorflow.keras.applications import MobileNetV2
except Exception:
    print("Missing package 'tensorflow'. Install with: pip install tensorflow")
    raise

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except Exception:
    print("Missing packages for plotting. Install with: pip install matplotlib seaborn")
    raise

try:
    from sklearn.metrics import classification_report, confusion_matrix
except Exception:
    print("Missing scikit-learn. Install with: pip install scikit-learn")
    raise

from tqdm import tqdm

# Optional: mediapipe for detection (not required for training)
try:
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_available = True
except Exception:
    mp = None
    mp_hands = None
    mp_available = False
    print("Note: 'mediapipe' not installed. Hand detection functions will be disabled. Install with: pip install mediapipe")

# Project root and dataset locations
PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_ZIP = PROJECT_ROOT / "dataset.zip"
ROOT_DIR = PROJECT_ROOT / "dataset"  # default dataset folder

# If a dataset.zip is present, attempt to unzip to ./dataset
if DATASET_ZIP.exists() and not ROOT_DIR.exists():
    try:
        print(f"Found {DATASET_ZIP}, extracting to {ROOT_DIR}...")
        with zipfile.ZipFile(DATASET_ZIP, 'r') as zf:
            zf.extractall(ROOT_DIR)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract {DATASET_ZIP}: {e}")

if not ROOT_DIR.exists():
    print(f"Dataset folder not found at {ROOT_DIR}.")
    print("Place your dataset folder at that location or provide a dataset.zip in the project folder.")
    # Continue but many operations will fail if dataset not present

# utility: parse XML annotation
def parse_xml_annotation(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        filename_node = root.find('filename')
        filename = filename_node.text if filename_node is not None else os.path.basename(xml_path).replace('.xml', '.jpg')

        obj = root.find('object')
        if obj is None:
            return None
        bbox = obj.find('bndbox')
        xmin = int(float(bbox.find('xmin').text))
        ymin = int(float(bbox.find('ymin').text))
        xmax = int(float(bbox.find('xmax').text))
        ymax = int(float(bbox.find('ymax').text))
        label = obj.find('name').text if obj.find('name') is not None else "unknown"
        return {'filename': filename, 'bbox': (xmin, ymin, xmax, ymax), 'label': label}
    except Exception as e:
        print(f"Error parsing {xml_path}: {e}")
        return None

# load dataset from XML annotations with caching
def load_dataset_from_xml(root_dir):
    root_dir = Path(root_dir)
    cache_path = root_dir / '_cached_data.pkl'
    if cache_path.exists():
        print(f"Loading data from cache: {cache_path}")
        with open(cache_path, 'rb') as f:
            data_list, unique_labels = pickle.load(f)
        return data_list, unique_labels

    xml_files = list(root_dir.glob("**/*.xml"))
    print(f"Found {len(xml_files)} XML files under {root_dir}")
    data_list = []
    for xml_path in tqdm(xml_files, desc="Parsing XML"):
        ann = parse_xml_annotation(xml_path)
        if ann is None:
            continue
        xml_dir = xml_path.parent
        parent_dir = xml_dir.parent
        base_name = Path(ann['filename']).stem
        found = False
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
            candidate = parent_dir / "Images (JPEG)" / f"{base_name}{ext}"
            if candidate.exists():
                ann['image_path'] = str(candidate)
                data_list.append(ann)
                found = True
                break
        if not found:
            # search recursively under parent_dir
            matches = list(parent_dir.glob(f"**/{base_name}.*"))
            if matches:
                ann['image_path'] = str(matches[0])
                data_list.append(ann)
            else:
                print(f"Warning: image for {ann['filename']} not found (xml: {xml_path})")

    unique_labels = sorted({d['label'] for d in data_list})
    with open(cache_path, 'wb') as f:
        pickle.dump((data_list, unique_labels), f)
    return data_list, unique_labels

def imread_unicode(path):
    from PIL import Image, UnidentifiedImageError
    import numpy as np
    import cv2
    try:
        # Open with Pillow (handles Windows Unicode paths reliably)
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        arr = np.array(img)            # RGB
        return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, FileNotFoundError, OSError) as e:
        print(f"Unable to read {path}: {e}")
        return None

# cropping helper
def crop_image_with_bbox(image_path, bbox, margin=10):
    try:
        img = imread_unicode(str(image_path))
        if img is None:
            return None
        h, w = img.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        xmin = max(0, xmin - margin)
        ymin = max(0, ymin - margin)
        xmax = min(w, xmax + margin)
        ymax = min(h, ymax + margin)
        cropped = img[ymin:ymax, xmin:xmax]
        return cropped
    except Exception as e:
        print(f"Error cropping {image_path}: {e}")
        return None

# Custom Keras data generator (unchanged logic)
class CroppedSignDataGenerator(keras.utils.Sequence):
    def __init__(self, data_list, class_names, batch_size=32, shuffle=True, margin=10, augment=False, target_size=(224,224)):
        self.data_list = data_list
        self.class_names = class_names
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.margin = margin
        self.augment = augment
        self.target_size = target_size
        self.label_to_idx = {label: idx for idx, label in enumerate(class_names)}
        self.indexes = np.arange(len(self.data_list))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.data_list) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self._generate_data(batch_indexes)
        return X, y

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _generate_data(self, batch_indexes):
        X = np.empty((len(batch_indexes), *self.target_size, 3), dtype=np.float32)
        y = np.empty((len(batch_indexes)), dtype=int)
        for i, idx in enumerate(batch_indexes):
            sample = self.data_list[idx]
            cropped_img = crop_image_with_bbox(sample['image_path'], sample['bbox'], self.margin)
            if cropped_img is None:
                cropped_img = np.zeros((*self.target_size,3), dtype=np.uint8)
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB)
            cropped_img = cv2.resize(cropped_img, self.target_size)
            cropped_img = cropped_img.astype(np.float32) / 255.0
            mean = np.array([0.485,0.456,0.406])
            std = np.array([0.229,0.224,0.225])
            cropped_img = (cropped_img - mean) / std
            X[i] = cropped_img
            y[i] = self.label_to_idx[sample['label']]
        return X, keras.utils.to_categorical(y, num_classes=len(self.class_names))

# build model helper
def build_mobilenet_classifier(num_classes, input_shape=(224,224,3)):
    base_model = MobileNetV2(input_shape=input_shape, include_top=False, weights='imagenet', pooling='avg')
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

# Mediapipe helpers (only if mediapipe installed)
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

    def is_real_ok_sign_from_results(results, image_width, image_height,
                                     touch_thr=0.12,
                                     extend_thr=0.35):
        if not results or not results.multi_hand_landmarks:
            return False

        for hand in results.multi_hand_landmarks:
            lm = hand.landmark

            # Normalize scale (size of hand)
            xs = [p.x for p in lm]
            ys = [p.y for p in lm]
            diag = np.sqrt((max(xs) - min(xs))**2 + (max(ys) - min(ys))**2)
            if diag == 0:
                return False

            # --- 1. Thumb tip & Index tip distance ---
            thumb = lm[4]
            index = lm[8]
            d_thumb_index = np.sqrt((thumb.x - index.x)**2 + (thumb.y - index.y)**2)

            if d_thumb_index > touch_thr * diag:
                return False  # they are not touching → not OK sign

            # --- 2. Check 3 extended fingers (12, 16, 20) ---
            wrist = lm[0]

            def extended(tip_id, base_id):
                tip = lm[tip_id]
                base = lm[base_id]
                d = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                return d > extend_thr * diag

            middle_ok = extended(12, 9)
            ring_ok = extended(16, 13)
            pinky_ok = extended(20, 17)

            if not (middle_ok and ring_ok and pinky_ok):
                return False  # fingers not extended → not OK sign

            # --- 3. Angle validation between thumb → index direction ---
            v1 = np.array([lm[4].x - lm[3].x, lm[4].y - lm[3].y])
            v2 = np.array([lm[8].x - lm[7].x, lm[8].y - lm[7].y])
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                return False

            cosine = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

            # Angle should be roughly facing each other → OK circle
            if cosine > -0.2:  # Allow slight variations
                return False

            return True

        return False
else:
    def detect_hand(image):
        raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe")

    def crop_hand_from_image(image_path, margin=10):
        raise RuntimeError("mediapipe not installed. Install with: pip install mediapipe")

# PyTorch imports and webcam function (if available)
try:
    import torch
    import torchvision.transforms as tv_transforms
    from PIL import ImageDraw, ImageFont
    import pyperclip
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    print("Note: 'torch' or related packages not installed. Webcam PyTorch inference will be disabled. Install with: pip install torch torchvision pyperclip")

device = None
if TORCH_AVAILABLE:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def run_webcam_no_mediapipe(model_path="thai_sign_model.pth", class_names=None, required_stable_frames=8):
    """
    Run webcam inference using a PyTorch model (no mediapipe). The function will:
     - try to load a scripted/traced model via torch.jit.load(model_path)
     - otherwise try to load state_dict into a model factory `get_pytorch_model(num_classes)` if provided in runtime
       (user must provide that factory in their environment, or replace this logic with their model class).
    Usage: call after class_names is populated in __main__:
        run_webcam_no_mediapipe("thai_sign_model.pth", class_names)
    """
    if not TORCH_AVAILABLE:
        print("Torch not available. Install torch to use run_webcam_no_mediapipe().")
        return

    if class_names is None:
        print("run_webcam_no_mediapipe requires class_names list. Pass class_names from your dataset.")
        return

    # Load model
    model = None
    if os.path.exists(model_path):
        # Prefer scripted model
        try:
            model = torch.jit.load(model_path, map_location=device)
            model.to(device)
            model.eval()
            print(f"Loaded scripted model from {model_path}")
        except Exception:
            # Try loading state_dict into user-provided model factory `get_pytorch_model`
            try:
                # User must provide get_pytorch_model(num_classes) in their environment
                from pytorch_model import get_pytorch_model  # optional: user file
                model = get_pytorch_model(len(class_names))
                state = torch.load(model_path, map_location=device)
                model.load_state_dict(state)
                model.to(device)
                model.eval()
                print(f"Loaded state_dict into model from {model_path}")
            except Exception as e:
                print("Failed to load PyTorch model. Provide a scripted model or implement pytorch_model.get_pytorch_model().")
                print("Error:", e)
                return
    else:
        print(f"Model file not found: {model_path}")
        return

    # Transforms
    transform_infer = tv_transforms.Compose([
        tv_transforms.ToTensor(),
        tv_transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(0)
    last_char = None
    stable_char = None
    current_word = ""
    stable_frames = 0

    # Font
    try:
        thai_font = ImageFont.truetype("C:/Users/BestyBest/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew.ttf", 40)
    except Exception:
        thai_font = ImageFont.load_default()

    print("[INFO] Starting webcam (Press ESC to quit, R to reset, C to copy word)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape
        # Crop center square ROI
        crop_size = min(h, w)
        x1, y1 = (w - crop_size)//2, (h - crop_size)//2
        x2, y2 = x1 + crop_size, y1 + crop_size
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224,224))

        # Dummy keypoints (zeros) if model expects extra tensor
        kp_tensor = torch.zeros(1, 63).to(device)

        img_pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        img_tensor = transform_infer(img_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = None
            try:
                outputs = model(img_tensor, kp_tensor)
            except Exception:
                try:
                    outputs = model(img_tensor)
                except Exception as e:
                    print("Model call failed. Ensure your model accepts (img_tensor[, kp_tensor]). Error:", e)
                    cap.release()
                    cv2.destroyAllWindows()
                    return
            if isinstance(outputs, (list,tuple)):
                outputs = outputs[0]
            if isinstance(outputs, torch.Tensor):
                pred = outputs.argmax(dim=1)
                label = class_names[pred.item()]
            else:
                print("Unexpected model output type:", type(outputs))
                cap.release()
                cv2.destroyAllWindows()
                return

        # Stability logic
        if label == last_char:
            stable_frames += 1
        else:
            stable_frames = 0
        last_char = label

        if stable_frames >= required_stable_frames and label not in ["No Hand", "OK"]:
            stable_char = label
            stable_frames = 0
        elif label == "OK" and stable_char:
            current_word += stable_char
            print(f"[CONFIRM] Added '{stable_char}' -> {current_word}")
            stable_char = None
        elif label == "OK" and not stable_char and current_word:
            try:
                pyperclip.copy(current_word)
                print(f"[COPIED] '{current_word}' copied to clipboard")
            except Exception:
                print("pyperclip not available or copy failed.")

        # Draw overlay
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        draw.text((10,30), f"สัญลักษณ์: {last_char}", font=thai_font, fill=(0,255,0))
        draw.text((10,70), f"รอตรวจสอบ: {stable_char if stable_char else '-'}", font=thai_font, fill=(0,255,255))
        draw.text((10,110), f"คำ: {current_word}", font=thai_font, fill=(255,255,0))
        draw.text((10,h-50), "R=รีเซ็ท | C=คัดลอก | ESC=ออก", font=thai_font, fill=(200,200,200))
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Thai Sign Recognition", frame)
        key = cv2.waitKey(3) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            current_word = ""
            print("[RESET] Word cleared")
        elif key == ord('c'):
            if current_word:
                try:
                    pyperclip.copy(current_word)
                    print(f"[COPIED] '{current_word}' copied to clipboard")
                except Exception:
                    print("pyperclip not available or copy failed.")

    cap.release()
    cv2.destroyAllWindows()
def run_webcam_with_keras(keras_model, class_names, use_mediapipe=True, required_stable_frames=8):
    """
    Run webcam inference using a Keras/TensorFlow model. Uses MediaPipe bbox if available,
    otherwise center-crops the frame. Press ESC to quit, R to reset, C to copy the current word.
    """
    try:
        from PIL import ImageDraw, ImageFont, Image
    except Exception:
        print("Pillow ImageDraw/ImageFont missing. Install with: pip install Pillow")
        return
    try:
        import pyperclip
        have_pyperclip = True
    except Exception:
        have_pyperclip = False

    cap = cv2.VideoCapture(0)
    last_char = None
    stable_char = None
    current_word = ""
    stable_frames = 0

    try:
        thai_font = ImageFont.truetype("C:/Users/BestyBest/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew.ttf", 40)
    except Exception:
        thai_font = ImageFont.load_default()

    mean = np.array([0.485,0.456,0.406])
    std = np.array([0.229,0.224,0.225])

    print("[INFO] Starting webcam (Press ESC to quit, R to reset, C to copy word)")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w = frame.shape[:2]

        # get bbox from mediapipe if available
        bbox = None
        if use_mediapipe and mp_available:
            results = detect_hand(frame)
            bbox = get_hand_bbox(results, w, h, margin=20)

        if bbox is None:
            # center-crop fallback
            crop_size = min(h, w)
            x1, y1 = (w - crop_size)//2, (h - crop_size)//2
            x2, y2 = x1 + crop_size, y1 + crop_size
        else:
            x1, y1, x2, y2 = bbox
            # clamp
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            roi = np.zeros((crop_size, crop_size, 3), dtype=np.uint8)
        resized = cv2.resize(roi, (224,224))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = (rgb - mean) / std
        inp_batch = np.expand_dims(inp, axis=0)

        preds = keras_model.predict(inp_batch)
        pred_idx = int(np.argmax(preds, axis=1)[0])
        label = class_names[pred_idx]

        # stability detection
        if label == last_char:
            stable_frames += 1
        else:
            stable_frames = 0
        last_char = label

        if stable_frames >= required_stable_frames and label not in ["No Hand", "OK"]:
            stable_char = label
            stable_frames = 0
        elif label == "OK" and stable_char:
            # verify real "OK" gesture with mediapipe landmarks (if available)
            confirm_ok = True
            if mp_available:
                results_now = detect_hand(frame)
                if not is_real_ok_sign_from_results(results_now, w, h):
                    confirm_ok = False
            if confirm_ok:
                current_word += stable_char
                print(f"[CONFIRM] Added '{stable_char}' -> {current_word}")
                stable_char = None
                stable_frames = 0  # reset after confirmation
            else:
                # ignore false "OK" prediction
                print("[INFO] 'OK' detected but real gesture not verified; waiting...")
        elif label == "OK" and not stable_char and current_word:
            # Copy word on second "OK"
            if have_pyperclip:
                try:
                    pyperclip.copy(current_word)
                    print(f"[COPIED] '{current_word}' copied to clipboard")
                except Exception:
                    print("Copy failed.")
            else:
                print("pyperclip not installed.")

        # drawing overlay
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)
        draw.text((10,30), f"สัญลักษณ์: {last_char}", font=thai_font, fill=(0,255,0))
        draw.text((10,70), f"รอตรวจสอบ: {stable_char if stable_char else '-'}", font=thai_font, fill=(0,255,255))
        draw.text((10,110), f"คำ: {current_word}", font=thai_font, fill=(255,255,0))
        draw.text((10,h-50), "R=รีเซ็ท | C=คัดลอก | ESC=ออก", font=thai_font, fill=(200,200,200))
        frame_out = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

        cv2.imshow("Thai Sign Recognition", frame_out)
        key = cv2.waitKey(3) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('r'):
            current_word = ""
            print("[RESET] Word cleared")
        elif key == ord('c'):
            if current_word and have_pyperclip:
                try:
                    pyperclip.copy(current_word)
                    print(f"[COPIED] '{current_word}' copied to clipboard")
                except Exception:
                    print("Copy failed.")

    cap.release()
    cv2.destroyAllWindows()
# Example usage guard
if __name__ == "__main__":
    # Load train/test sets if present
    train_dir = ROOT_DIR / "Training set"
    test_dir = ROOT_DIR / "Test set"

    train_data, class_names = ([], [])
    test_data = []

    if train_dir.exists():
        print("Loading training annotations...")
        train_data, class_names = load_dataset_from_xml(train_dir)
    else:
        print(f"Training directory not found: {train_dir}")

    if test_dir.exists():
        print("Loading test annotations...")
        test_data, _ = load_dataset_from_xml(test_dir)
    else:
        print(f"Test directory not found: {test_dir}")

    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}, Classes: {len(class_names)}")

    if len(class_names) == 0:
        print("No classes found. Exiting.")
        sys.exit(0)

    # Instantiate generators (example)
    batch_size = 32
    train_gen = CroppedSignDataGenerator(train_data, class_names, batch_size=batch_size, shuffle=True)
    val_gen = CroppedSignDataGenerator(test_data, class_names, batch_size=batch_size, shuffle=False)

    # Build and compile model (use saved model if present)
    MODEL_H5 = PROJECT_ROOT / "dataset" / "model" / "best_sign_model.h5"

    if MODEL_H5.exists():
        print(f"Loading HDF5 model from {MODEL_H5}")
        model = tf.keras.models.load_model(str(MODEL_H5))
    else:
        print(f"Model file not found at {MODEL_H5}")
        print("Building a new model as fallback.")
        model = build_mobilenet_classifier(len(class_names))

    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Model ready. Parameters:", model.count_params())

    # Run webcam demo directly with Keras model
    print("\n[INFO] Starting Thai Sign Recognition Webcam Demo...")
    try:
        run_webcam_with_keras(model, class_names, use_mediapipe=mp_available)
    except Exception as e:
        print("Webcam demo failed:", e)