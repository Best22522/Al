# eng_sign_ai_with_keypoints.py

import os
import cv2
from PIL import Image, UnidentifiedImageError
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import time 
from sklearn.model_selection import train_test_split
import shutil

print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))
    print("GPU Memory:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

MODEL_PATH = "thai_sign_model.pth"

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
val_dir = "Dataset/Test set"

# Move device definition before transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Update transform definition without .to(device)
transform = transforms.Compose([
    transforms.Resize((224, 224), antialias=True),
    transforms.RandomHorizontalFlip(p=0.3),  # Reduced probability
    transforms.RandomRotation(10),  # Reduced rotation
    transforms.RandomAffine(
        degrees=0, 
        translate=(0.1, 0.1),
        scale=(0.9, 1.1)  # Add slight scaling
    ),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

mp_hands = mp.solutions.hands

def crop_hand(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
        result = hands.process(rgb)
        if result.multi_hand_landmarks:
            h, w, _ = img.shape
            bbox = [w, h, 0, 0]  # min_x, min_y, max_x, max_y
            
            for lm in result.multi_hand_landmarks[0].landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                bbox[0] = min(bbox[0], x)
                bbox[1] = min(bbox[1], y)
                bbox[2] = max(bbox[2], x)
                bbox[3] = max(bbox[3], y)
            
            # Add some padding
            pad = 20
            min_x = max(bbox[0] - pad, 0)
            min_y = max(bbox[1] - pad, 0)
            max_x = min(bbox[2] + pad, w)
            max_y = min(bbox[3] + pad, h)

            cropped = img[min_y:max_y, min_x:max_x]
            return Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
        else:
            # If no hand detected, return full image
            return Image.fromarray(rgb)

def safe_loader(path):
    try:
        return crop_hand(path)
    except UnidentifiedImageError:
        print(f"[WARN] Skipping unreadable image: {path}")
        return Image.new('RGB', (224, 224))

def check_and_create_datasets():
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    # Verify structure
    train_classes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
    val_classes = [d for d in os.listdir(val_dir) if os.path.isdir(os.path.join(val_dir, d))]
    
    print(f"Found {len(train_classes)} training classes: {sorted(train_classes)}")
    print(f"Found {len(val_classes)} test classes: {sorted(val_classes)}")

check_and_create_datasets()

train_ds = datasets.ImageFolder(train_dir, transform=transform, loader=safe_loader)
val_ds   = datasets.ImageFolder(val_dir, transform=transform, loader=safe_loader)

train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, pin_memory=True, num_workers=4)
val_loader   = DataLoader(val_ds, batch_size=128, shuffle=False, pin_memory=True, num_workers=4)

class_names = train_ds.classes
print("Classes found:", class_names)

########################################
# STEP 2: Build Model (MobileNetV2)
########################################
def create_model(new_model=True):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Use the same architecture regardless of new_model flag
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(model.last_channel, 512),
        nn.ReLU(),
        nn.BatchNorm1d(512),
        nn.Dropout(p=0.3),
        nn.Linear(512, len(class_names))
    )
    
    return model.to(device)

# Update model initialization
if os.path.exists(MODEL_PATH):
    print("🔄 Loading existing model...")
    model = create_model()  # Remove new_model parameter
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("🆕 Creating new model instead...")
        if os.path.exists(MODEL_PATH):
            os.rename(MODEL_PATH, "old_" + MODEL_PATH)
            print(f"Renamed existing model to old_{MODEL_PATH}")
else:
    print("🆕 Creating new model...")
    model = create_model()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

########################################
# STEP 3: Training Loop
########################################
def evaluate_model(model, data_loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(data_loader.dataset)
    epoch_acc = running_corrects.double() / len(data_loader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_labels

def plot_confusion_matrix(true_labels, pred_labels, classes):
    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Optimize training loop for better GPU performance
def train_model(num_epochs=1):
    scaler = torch.cuda.amp.GradScaler()  # Enable automatic mixed precision
    
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'epoch_times': []  # Add timing tracking
    }
    
    total_start_time = time.time()  # Start timing total training
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start timing epoch
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        train_loss = running_loss / len(train_ds)
        train_acc = running_corrects.double() / len(train_ds)
        
        # Validation phase
        val_loss, val_acc, val_preds, val_labels = evaluate_model(model, val_loader)
        
        # Store metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc.item())
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc.item())
        
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        history['epoch_times'].append(epoch_time)
        
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Training   - Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
        print(f"Validation - Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        print(f"Time: {epoch_time:.2f} seconds")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)  # Use MODEL_PATH constant
            print(f"✨ New best model saved as {MODEL_PATH}!")
    
    total_time = time.time() - total_start_time
    
    # Final evaluation and confusion matrix
    final_val_loss, final_val_acc, final_preds, final_labels = evaluate_model(model, val_loader)
    plot_confusion_matrix(final_labels, final_preds, class_names)
    
    print("\n🎯 Final Results:")
    print(f"Best Validation Accuracy: {best_val_acc:.4f}")
    print(f"Total Training Time: {total_time/60:.2f} minutes")
    print(f"Average Epoch Time: {np.mean(history['epoch_times']):.2f} seconds")
    print("✅ Model saved as eng.pth")
    print("📊 Confusion matrix saved as confusion_matrix.png")
    
    return history

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
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
        
    # Load trained model
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print(f"✅ Model loaded from {MODEL_PATH}")
    
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

            # Update display text
            cv2.putText(frame, f"Letter: {label}", (10,50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("English Sign Recognition", frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

    cap.release()
    cv2.destroyAllWindows()

########################################
# GPU Statistics
########################################
def print_gpu_stats():
    if torch.cuda.is_available():
        print("\nGPU Statistics:")
        print(f"Memory Allocated: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
        print(f"Memory Cached: {torch.cuda.memory_reserved(0)/1e9:.2f} GB")
        print(f"GPU Utilization: {torch.cuda.utilization(0)}%")

########################################
# Check Dataset Structure
########################################
def check_dataset_structure():
    print("\nChecking dataset structure...")
    
    # Check directories
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Training directory not found: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"Test directory not found: {val_dir}")
    
    # Get class folders
    test_folders = [d for d in os.listdir(val_dir) 
                   if os.path.isdir(os.path.join(val_dir, d))]
    
    if not test_folders:
        raise FileNotFoundError(f"No class folders found in {val_dir}")
    
    print(f"✓ Found {len(test_folders)} test class folders")
    
    # Count images in each folder
    total_images = 0
    for folder in test_folders:
        folder_path = os.path.join(val_dir, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
        total_images += len(images)
        print(f"  - {folder}: {len(images)} images")
    
    print(f"✓ Total of {total_images} test images found")
    print(f"✓ Class folders: {sorted(test_folders)}")

def create_test_set(train_dir="Dataset/Training set", val_dir="Dataset/Test set", split=0.2):
    os.makedirs(val_dir, exist_ok=True)
    
    for class_name in os.listdir(train_dir):
        class_dir = os.path.join(train_dir, class_name)
        if os.path.isdir(class_dir):
            val_class_dir = os.path.join(val_dir, class_name)
            os.makedirs(val_class_dir, exist_ok=True)
            
            # Get all images in class
            images = [f for f in os.listdir(class_dir) 
                     if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            
            if not images:
                continue
                
            # Calculate number of validation images needed
            num_val = max(1, int(len(images) * split))
            
            # Randomly select images for validation
            val_images = np.random.choice(images, size=num_val, replace=False)
            
            # Move validation images
            for img in val_images:
                src = os.path.join(class_dir, img)
                dst = os.path.join(val_class_dir, img)
                try:
                    shutil.copy2(src, dst)  # Use copy instead of move
                    print(f"Copied {img} to test set")
                except Exception as e:
                    print(f"Error copying {img}: {e}")
    
    print("✅ Test set created successfully")

########################################
# MAIN
########################################
if __name__ == "__main__":
    check_dataset_structure()
    
    if os.path.exists(MODEL_PATH):
        print(f"🔄 Loading existing model from {MODEL_PATH}...")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    
    print("🚀 Training model...")
    history = train_model(num_epochs=1)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(history['epoch_times'], marker='o')
    plt.title('Training Time per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    print("📷 Starting webcam...")
    run_webcam()

