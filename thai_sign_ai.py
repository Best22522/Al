import os
import cv2
from PIL import Image, ImageFont, ImageDraw, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import numpy as np
import torchvision.utils as vutils
import pyperclip
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import transforms
import xml.etree.ElementTree as ET

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

# ---------------- STEP 1: Dataset with XML-based Cropping ----------------
train_dir = "Datasetgoon/Training set"
val_dir   = "Datasetgoon/Test set"

transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                         [0.229, 0.224, 0.225])
])

# ---------------- STEP 1a: Extract hand from XML ----------------
def extract_hand_and_keypoints(image_path, size=224, margin=20):
    """
    Reads image and its XML annotation, crops hand using <bndbox>, resizes,
    returns PIL image and dummy keypoints (zeros).
    """
    xml_path = image_path.rsplit('.',1)[0]+".xml"  # assumes XML has same name
    image_bgr = cv2.imread(image_path)
    if image_bgr is None or not os.path.exists(xml_path):
        return Image.new('RGB', (size,size)), np.zeros(63, dtype=np.float32)

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        obj = root.find('object')  # assume single hand
        if obj is None:
            raise ValueError("No object found in XML")
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Apply margin and clamp
        h, w, _ = image_bgr.shape
        xmin, ymin = max(0, xmin-margin), max(0, ymin-margin)
        xmax, ymax = min(w, xmax+margin), min(h, ymax+margin)

        cropped = image_bgr[ymin:ymax, xmin:xmax]
        cropped_resized = cv2.resize(cropped, (size,size))
        img_pil = Image.fromarray(cv2.cvtColor(cropped_resized, cv2.COLOR_BGR2RGB))

        # Dummy keypoints since we rely on XML bbox
        keypoints = np.zeros(63, dtype=np.float32)
        return img_pil, keypoints
    except Exception as e:
        print(f"[ERROR] {image_path}: {e}")
        return Image.new('RGB', (size,size)), np.zeros(63, dtype=np.float32)

# ---------------- STEP 1b: Dataset Class ----------------
class HandDatasetWithKeypoints(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.dataset = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for f in os.listdir(folder_path):
                    if f.lower().endswith((".jpg", ".jpeg")):
                        img_path = os.path.join(folder_path, f)
                        xml_path = img_path.rsplit('.',1)[0]+".xml"
                        if os.path.exists(xml_path):
                            self.dataset.append((img_path, folder))  # folder = class name

        # Map class names to integers
        self.class_to_idx = {c:i for i,c in enumerate(sorted(set([s[1] for s in self.dataset])))}
        self.samples = [(s[0], self.class_to_idx[s[1]]) for s in self.dataset]

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img_pil, keypoints = extract_hand_and_keypoints(img_path)
        if self.transform:
            img_pil = self.transform(img_pil)
        return img_pil, torch.tensor(keypoints), label

    def __len__(self):
        return len(self.samples)

# ---------------- STEP 1c: Load datasets ----------------
train_ds = HandDatasetWithKeypoints(train_dir, transform=transform_train)
val_ds   = HandDatasetWithKeypoints(val_dir, transform=transform_val)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

class_names = sorted(train_ds.class_to_idx, key=lambda k: train_ds.class_to_idx[k])
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
        for inputs, keypoints, labels in train_loader:
            inputs, keypoints, labels = inputs.to(device), keypoints.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, keypoints)
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
            for inputs, keypoints, labels in val_loader:
                inputs, keypoints, labels = inputs.to(device), keypoints.to(device), labels.to(device)
                outputs = model(inputs, keypoints)
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

    torch.save(model.state_dict(), "thai_sign_model.pth")
    print("✅ Model saved!")

    # ---------- Plot ----------
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
    running_loss, running_corrects = 0.0, 0
    with torch.no_grad():
        for inputs, keypoints, labels in val_loader:
            inputs, keypoints, labels = inputs.to(device), keypoints.to(device), labels.to(device)
            outputs = model(inputs, keypoints)
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
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, keypoints, labels in val_loader:
            inputs, keypoints, labels = inputs.to(device), keypoints.to(device), labels.to(device)
            outputs = model(inputs, keypoints)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\n------------------ Classification Report ------------------\n")
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

# ---------------- STEP 6: Confusion Matrix ----------------
def plot_confusion_matrix(loader, model, class_names, title="Confusion Matrix"):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, keypoints, labels in loader:
            inputs, keypoints, labels = inputs.to(device), keypoints.to(device), labels.to(device)
            outputs = model(inputs, keypoints)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(title)
    os.makedirs("training_results", exist_ok=True)
    plt.savefig(f"training_results/{title.replace(' ', '_')}.png")
    plt.show()

#---------------- ถึงนี่ ----------------

# ---------------- STEP 7: Webcam Inference (MediaPipe Removed) ----------------
def run_webcam_no_mediapipe():
    model.load_state_dict(torch.load("thai_sign_model.pth", map_location=device))
    model.eval()
    transform_infer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    cap = cv2.VideoCapture(0)
    last_char, stable_char, current_word = None, None, ""
    stable_frames = 0
    required_stable_frames = 8

    try:
        thai_font = ImageFont.truetype("C:/Users/BestyBest/AppData/Local/Microsoft/Windows/Fonts/THSarabunNew.ttf", 40)
    except:
        thai_font = ImageFont.load_default()

    print("[INFO] Starting webcam (Press ESC to quit, R to reset, C to copy word)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        h, w, _ = frame.shape

        # ---------------- Crop center square of frame as hand ROI ----------------
        crop_size = min(h, w)
        x1, y1 = (w - crop_size)//2, (h - crop_size)//2
        x2, y2 = x1 + crop_size, y1 + crop_size
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (224,224))

        # Dummy keypoints (zeros) since no landmarks
        kp_tensor = torch.zeros(1, 63).to(device)

        # Predict
        img_tensor = transform_infer(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor, kp_tensor)
            _, pred = outputs.max(1)
            label = class_names[pred.item()]

        # Stability check
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

        # Draw
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        draw.rectangle([x1,y1,x2,y2], outline=(0,255,0), width=2)  # ROI box
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

    evaluate_model()
    classification_report_folder()
    plot_confusion_matrix(val_loader, model, class_names, title="Confusion Matrix (Model 1)")