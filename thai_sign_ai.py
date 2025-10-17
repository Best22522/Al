import os
import cv2
from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import mediapipe as mp
import numpy as np
import torchvision.utils as vutils
import pyperclip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- STEP 0: Convert .jpg to .jpeg ----------------
def convert_jpg_to_jpeg(root_dir="Datasetgoon"):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                jpg_path = os.path.join(root, f)
                jpeg_path = os.path.join(root, f[:-4] + ".jpeg")
                try:
                    img = Image.open(jpg_path).convert("RGB")
                    img.save(jpeg_path)
                    os.remove(jpg_path)
                    print(f"[CONVERT] {jpg_path} -> {jpeg_path}")
                except UnidentifiedImageError:
                    print(f"[SKIP] Corrupted image: {jpg_path}")

# ---------------- STEP 1: Dataset ----------------
train_dir = "Datasetgoon/Training set"
val_dir   = "Datasetgoon/Test set"

transform_train = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

from torchvision.datasets.folder import default_loader
def safe_loader(path):
    try:
        return default_loader(path)
    except UnidentifiedImageError:
        return Image.new('RGB', (224,224))

train_ds = datasets.ImageFolder(train_dir, transform=transform_train, loader=safe_loader)
val_ds   = datasets.ImageFolder(val_dir, transform=transform_val, loader=safe_loader)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)
class_names = train_ds.classes
print("Classes:", class_names)

# ---------------- STEP 2: Model with Keypoints ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MobileNetWithKeypoints(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.mobilenet_v2(pretrained=True).features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc_img = nn.Linear(1280, 256)
        self.fc_kp = nn.Linear(63, 64)
        self.fc_out = nn.Linear(256+64, num_classes)
    
    def forward(self, x_img, x_kp):
        x = self.backbone(x_img)
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc_img(x)
        x_kp = self.fc_kp(x_kp)
        x_combined = torch.cat([x, x_kp], dim=1)
        return self.fc_out(x_combined)

model = MobileNetWithKeypoints(len(class_names)).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---------------- STEP 3: Training Function ----------------
def train_model(num_epochs=20):
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device) 
            kp_dummy = torch.zeros(inputs.size(0), 63).to(device)
            optimizer.zero_grad()
            outputs = model(inputs, kp_dummy)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            _, preds = torch.max(outputs,1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects.double() / len(train_ds)

        # ---------- Validation ----------
        model.eval()
        val_loss, val_corrects = 0.0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                kp_dummy = torch.zeros(inputs.size(0), 63).to(device)
                outputs = model(inputs, kp_dummy)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs,1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)
        val_loss /= len(val_ds)
        val_acc = val_corrects.double() / len(val_ds)

        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())

    # Save model
    torch.save(model.state_dict(), "thai_sign_model.pth")
    print("✅ Model saved!")

    # ---------- Plot and Save ----------
    os.makedirs("training_results", exist_ok=True)

    plt.figure(figsize=(12,6))
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_results/accuracy_curve.png")
    plt.close()

    plt.figure(figsize=(12,6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig("training_results/loss_curve.png")
    plt.close()

    print("📊 Saved training curves in 'training_results/' folder")

    return history

# ---------------- STEP 4: Evaluate Model ----------------
def evaluate_model():
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            kp_dummy = torch.zeros(inputs.size(0), 63).to(device)
            outputs = model(inputs, kp_dummy)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
    test_loss = running_loss / len(val_ds)
    test_acc = running_corrects.double() / len(val_ds)
    print(f"📊 Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")
    return test_loss, test_acc

# ---------------- STEP 5: Classification Report ----------------
def classification_report_folder():
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            kp_dummy = torch.zeros(inputs.size(0), 63).to(device)
            outputs = model(inputs, kp_dummy)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    print("\n------------------ Task 2 Feature Extraction ------------------ \n")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

# ---------------- STEP 6: Confusion Matrix ----------------
def plot_confusion_matrix(loader, model, class_names, title="Confusion Matrix"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            kp_dummy = torch.zeros(inputs.size(0), 63).to(device)
            outputs = model(inputs, kp_dummy)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(title)
    os.makedirs("training_results", exist_ok=True)
    plt.savefig(f"training_results/{title.replace(' ', '_')}.png")
    plt.show()
    print(f"📊 Confusion matrix saved as training_results/{title.replace(' ', '_')}.png")

# ---------------- STEP 7: Mediapipe Webcam Inference ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def crop_and_resize_hand(frame, hand_landmarks, size=224, margin=20):
    h,w,_ = frame.shape
    x_min = int(min([lm.x for lm in hand_landmarks.landmark]) * w)
    x_max = int(max([lm.x for lm in hand_landmarks.landmark]) * w)
    y_min = int(min([lm.y for lm in hand_landmarks.landmark]) * h)
    y_max = int(max([lm.y for lm in hand_landmarks.landmark]) * h)
    x_min, y_min = max(0, x_min-margin), max(0, y_min-margin)
    x_max, y_max = min(w, x_max+margin), min(h, y_max+margin)
    cropped = frame[y_min:y_max, x_min:x_max]
    resized = cv2.resize(cropped, (size,size))
    return resized

def get_hand_keypoints(hand_landmarks):
    kp = []
    for lm in hand_landmarks.landmark:
        kp.extend([lm.x, lm.y, lm.z])
    return torch.tensor(kp, dtype=torch.float32).to(device)

def run_webcam():
    model.load_state_dict(torch.load("thai_sign_model.pth", map_location=device))
    model.eval()
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    cap = cv2.VideoCapture(0)
    last_char = None
    stable_char = None
    current_word = ""
    stable_frames = 0
    required_stable_frames = 8
    thai_font = ImageFont.truetype("C:/Users/BestyBest/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew.ttf", 40)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            h, w, c = frame.shape
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            label = "No Hand"
            ok_detected = False

            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

                    # Detect OK gesture manually
                    thumb_tip = handLms.landmark[4]
                    index_tip = handLms.landmark[8]
                    middle_tip = handLms.landmark[12]
                    thumb_index_dist = np.sqrt(
                        (thumb_tip.x - index_tip.x)**2 +
                        (thumb_tip.y - index_tip.y)**2 +
                        (thumb_tip.z - index_tip.z)**2
                    )

                    if thumb_index_dist < 0.05 and middle_tip.y < handLms.landmark[9].y:
                        label = "OK"
                        ok_detected = True
                    else:
                        cropped = crop_and_resize_hand(frame, handLms)
                        img_tensor = transform_infer(Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
                        kp_tensor = get_hand_keypoints(handLms).unsqueeze(0)
                        with torch.no_grad():
                            outputs = model(img_tensor, kp_tensor)
                            _, pred = outputs.max(1)
                            label = class_names[pred.item()]

                    # ---------- Stable logic ----------
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
                        print(f"[CONFIRM] Added '{stable_char}' to word -> {current_word}")
                        stable_char = None

                    elif label == "OK" and not stable_char and current_word:
                        pyperclip.copy(current_word)
                        print(f"[COPIED] '{current_word}' copied to clipboard")

            # Draw text on screen
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)
            draw.text((10,30), f"สัญลักษณ์: {last_char}", font=thai_font, fill=(0,255,0))
            draw.text((10,70), f"รอตรวจสอบ: {stable_char if stable_char else '-'}", font=thai_font, fill=(0,255,255))
            draw.text((10,110), f"คำ: {current_word}", font=thai_font, fill=(255,255,0))
            if ok_detected:
                draw.text((w-200, 30), "👌 OK Detected ✅", font=thai_font, fill=(0,255,0))
            draw.text((10,h-50), "R=รีเซ็ท | C=คัดลอก | ESC=ออก", font=thai_font, fill=(200,200,200))
            frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

            cv2.imshow("Thai Sign Recognition", frame)
            key = cv2.waitKey(3) & 0xFF
            if key == 27:
                break
            elif key == ord('r'):
                current_word = ""
                print("[RESET] Word cleared")
            elif key == ord('c'):
                if current_word:
                    pyperclip.copy(current_word)
                    print(f"[COPIED] '{current_word}' copied to clipboard")

    cap.release()
    cv2.destroyAllWindows()

# ---------------- MAIN ----------------
if __name__=="__main__":
    convert_jpg_to_jpeg()
    if os.path.exists("thai_sign_model.pth"):
        print("🔄 Loading existing model...")
        model.load_state_dict(torch.load("thai_sign_model.pth", map_location=device))
    else:
        print("🚀 Training model...")
        train_model(num_epochs=20)

    evaluate_model()                  # Test accuracy/loss
    classification_report_folder()    # Per-class metrics
    plot_confusion_matrix(val_loader, model, class_names, title="Confusion Matrix (Model 1)")

    run_webcam()                      # Live inference with OK gesture
