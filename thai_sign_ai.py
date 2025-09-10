# thai_sign_ai_with_keypoints.py

import os
import cv2
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import mediapipe as mp

########################################
# STEP 0: Convert all .jpg to .jpeg
########################################
def convert_jpg_to_jpeg(root_dir="Dataset"):
    for root, _, files in os.walk(root_dir):
        for f in files:
            if f.lower().endswith(".jpg"):
                jpg_path = os.path.join(root, f)
                jpeg_path = os.path.join(root, f[:-4] + ".jpeg")
                try:
                    img = Image.open(jpg_path)
                    img = img.convert("RGB")
                    img.save(jpeg_path)
                    os.remove(jpg_path)
                    print(f"[CONVERT] {jpg_path} -> {jpeg_path}")
                except UnidentifiedImageError:
                    print(f"[SKIP] Corrupted image: {jpg_path}")

convert_jpg_to_jpeg()

########################################
# STEP 1: Load Dataset
########################################
train_dir = "Dataset/Training set"
val_dir   = "Dataset/Test set"

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# Custom safe loader to skip corrupted images
from torchvision.datasets.folder import default_loader
def safe_loader(path):
    try:
        return default_loader(path)
    except UnidentifiedImageError:
        print(f"[WARN] Skipping unreadable image: {path}")
        return Image.new('RGB', (224,224))  # dummy black image

train_ds = datasets.ImageFolder(train_dir, transform=transform, loader=safe_loader)
val_ds   = datasets.ImageFolder(val_dir, transform=transform, loader=safe_loader)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

class_names = train_ds.classes
print("Classes found:", class_names)

########################################
# STEP 2: Build Model (MobileNetV2)
########################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.last_channel, len(class_names))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

########################################
# STEP 3: Training Loop
########################################
def train_model(num_epochs=5):
    for epoch in range(num_epochs):
        model.train()
        running_loss, running_corrects = 0.0, 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_ds)
        epoch_acc = running_corrects.double() / len(train_ds)

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # Save model
    torch.save(model.state_dict(), "thai_sign_model.pth")
    print("✅ Model saved as thai_sign_model.pth")

########################################
# STEP 4: Hand Keypoints & Fingers Up
########################################
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
TIP_IDS = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb
    if hand_landmarks.landmark[TIP_IDS[0]].x < hand_landmarks.landmark[TIP_IDS[0]-1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for id in range(1,5):
        if hand_landmarks.landmark[TIP_IDS[id]].y < hand_landmarks.landmark[TIP_IDS[id]-2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

########################################
# STEP 5: Real-time Webcam Prediction
########################################
def run_webcam():
    # Load trained model
    model.load_state_dict(torch.load("thai_sign_model.pth", map_location=device))
    model.eval()

    transform_infer = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
    ])

    cap = cv2.VideoCapture(0)

    with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            # Draw hand landmarks
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    # Fingers up
                    finger_state = fingers_up(handLms)
                    cv2.putText(frame, f"Fingers: {finger_state}", (10,100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

            # CNN Prediction
            img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tensor = transform_infer(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                _, pred = outputs.max(1)
                label = class_names[pred.item()].replace("_", " ")

            cv2.putText(frame, f"Prediction: {label}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Thai Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

########################################
# MAIN
########################################
if __name__ == "__main__":
    if os.path.exists("thai_sign_model.pth"):
        print("🔄 Loading existing model for continued training...")
        model.load_state_dict(torch.load("thai_sign_model.pth", map_location=device))
    
    print("🚀 Training model...")
    train_model(num_epochs=1)   # continue training
    
    print("📷 Starting webcam...")
    run_webcam()

