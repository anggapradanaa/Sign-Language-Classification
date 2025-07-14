# train_model.py
# Script untuk training model SIBI classifier

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import json
import warnings
warnings.filterwarnings('ignore')

# Set random seed untuk reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version:", tf.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))

# ==================== KONFIGURASI DATASET ====================
# Path ke dataset SIBI
DATASET_PATH = r"D:\Perkuliahan\Data Science and Machine Learning\Sistem Isyarat Bahasa Indonesia (SIBI)\SIBI"

# Parameter untuk model
IMG_SIZE = (224, 224)  # Ukuran input untuk MobileNetV2
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# Cek apakah path dataset ada
if not os.path.exists(DATASET_PATH):
    print(f"ERROR: Path dataset tidak ditemukan: {DATASET_PATH}")
    print("Pastikan path dataset benar!")
    exit()
else:
    print(f"Dataset ditemukan di: {DATASET_PATH}")

# ==================== EKSPLORASI DATASET ====================
def explore_dataset(dataset_path):
    """Fungsi untuk mengeksplorasi struktur dataset"""
    print("=== EKSPLORASI DATASET ===")
    
    # Dapatkan daftar folder (kelas)
    classes = sorted([d for d in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, d))])
    
    print(f"Jumlah kelas: {len(classes)}")
    print(f"Kelas: {classes}")
    
    # Hitung jumlah gambar per kelas
    class_counts = {}
    total_images = 0
    
    for class_name in classes:
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        class_counts[class_name] = len(image_files)
        total_images += len(image_files)
    
    print(f"\nTotal gambar: {total_images}")
    print("\nDistribusi gambar per kelas:")
    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count} gambar")
    
    return classes, class_counts

# ==================== PREPROCESSING DATA ====================
def load_and_preprocess_data(dataset_path, classes, img_size):
    """Load dan preprocessing data gambar"""
    print("=== LOADING DAN PREPROCESSING DATA ===")
    
    images = []
    labels = []
    
    for class_idx, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        image_files = [f for f in os.listdir(class_path) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"Processing kelas {class_name}: {len(image_files)} gambar")
        
        for image_file in image_files:
            try:
                # Load gambar
                img_path = os.path.join(class_path, image_file)
                img = cv2.imread(img_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Resize gambar
                img = cv2.resize(img, img_size)
                
                # Normalisasi pixel values ke [0, 1]
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue
    
    return np.array(images), np.array(labels)

# ==================== TRANSFER LEARNING MODEL ====================
def create_transfer_learning_model(num_classes, img_size):
    """Membuat model transfer learning dengan MobileNetV2"""
    print("=== MEMBUAT MODEL TRANSFER LEARNING ===")
    
    # Load pre-trained MobileNetV2
    base_model = MobileNetV2(
        input_shape=img_size + (3,),
        include_top=False,
        weights='imagenet'
    )
    
    # Freeze base model layers
    base_model.trainable = False
    
    # Add custom classification layers
    model = keras.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.2),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    print(f"Model summary:")
    model.summary()
    
    return model

# ==================== MAIN TRAINING PROCESS ====================
def main():
    # Eksplorasi dataset
    classes, class_counts = explore_dataset(DATASET_PATH)
    num_classes = len(classes)
    
    # Simpan class names untuk digunakan nanti
    class_names = {i: class_name for i, class_name in enumerate(classes)}
    with open('models/class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    # Visualisasi distribusi kelas
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Distribusi Jumlah Gambar per Kelas SIBI')
    plt.xlabel('Kelas (Huruf)')
    plt.ylabel('Jumlah Gambar')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/class_distribution.png')
    plt.show()
    
    # Load data
    X, y = load_and_preprocess_data(DATASET_PATH, classes, IMG_SIZE)
    
    print(f"\nShape data gambar: {X.shape}")
    print(f"Shape data label: {y.shape}")
    print(f"Rentang nilai pixel: {X.min():.3f} - {X.max():.3f}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"\nPembagian dataset:")
    print(f"Training: {X_train.shape[0]} gambar ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"Validation: {X_val.shape[0]} gambar ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"Testing: {X_test.shape[0]} gambar ({X_test.shape[0]/len(X)*100:.1f}%)")
    
    # Data augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Buat model
    model = create_transfer_learning_model(num_classes, IMG_SIZE)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=5,
        min_lr=0.00001
    )
    
    checkpoint = ModelCheckpoint(
        'models/best_sibi_model.keras',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max'
    )
    
    callbacks = [early_stopping, reduce_lr, checkpoint]
    
    # Training
    print("=== TRAINING MODEL ===")
    history = model.fit(
        train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluasi
    print("=== EVALUASI MODEL ===")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Prediksi dan classification report
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=classes))
    
    # Simpan classification report
    report = classification_report(y_test, y_pred_classes, target_names=classes, output_dict=True)
    with open('models/classification_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Visualisasi hasil
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss plot
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('plots/training_history.png')
    plt.show()
    
    # Confusion Matrix
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig('plots/confusion_matrix.png')
    plt.show()
    
    # Save final model
    model.save('models/sibi_classifier_final.keras')
    
    # Simpan training history
    import pickle
    with open('models/training_history.pkl', 'wb') as f:
        pickle.dump(history.history, f)
    
    print("\n=== TRAINING SELESAI ===")
    print(f"Final Test Accuracy: {test_accuracy:.4f}")
    print("Model dan file pendukung telah disimpan di folder 'models/'")

if __name__ == "__main__":
    # Buat folder jika belum ada
    os.makedirs('models', exist_ok=True)
    os.makedirs('plots', exist_ok=True)
    
    main()