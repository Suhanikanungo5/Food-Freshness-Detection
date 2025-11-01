import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



DATA_DIR = r"C:\Users\Suhani Kanungo\Downloads\archive (5)\Dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32  # BATCH ZIZE***
EPOCHS = 15      # EPOCH ***

print(f"Using TensorFlow version: {tf.__version__}")

# LOAD DATASET.***

print("Loading training data...")
# Training set (80% of the data)
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'  # BINARY FOR 2 CLASSES***
)

print("Loading validation data...")
# Validation set (the other 20%)
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Get class names (e.g., ['Fresh', 'Rotten'])
class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#DATA AUGMENTATION ***

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)

# BUILD MODEL CNN***

print("Building Model 1: Baseline CNN...")

model_1 = keras.Sequential([
    # 1. Add Data Augmentation layers
    data_augmentation,
    
    # 2. Rescaling layer ***
    layers.Rescaling(1./255),

    # --- Convolutional Block 1 ---
    
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    
    # --- Convolutional Block 2 ---
    layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- Convolutional Block 3 ---
    layers.Conv2D(128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    # --- Classification Head ---
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    
    layers.Dropout(0.5),  # Dropout (0.5) is our regularization ***
    
    layers.Dense(1, activation='sigmoid') # Sigmoid for 2 classes***
], name="baseline_cnn")


# Part 5: Compile the Model

model_1.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)


#  Model Summary ***

print("\n--- Model 1 Summary ---")
model_1.summary()
print("-----------------------\n")


# Part 7: Train the Model

print("Starting model training...")
history_1 = model_1.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("Model training complete.")


# Part 8: Evaluate the Model

print("Evaluating model and plotting results...")

# 1. Plot Accuracy and Loss Graphs



acc = history_1.history['accuracy']
val_acc = history_1.history['val_accuracy']
loss = history_1.history['loss']
val_loss = history_1.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.suptitle('Model 1 (Baseline CNN) Performance')
plt.savefig('model_1_performance_graph.png') # Save the graph
plt.show()

# 2. Generate Confusion Matrix


y_true = []
y_pred_probs = []

# Iterate over the validation dataset
for images, labels in validation_dataset:
    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(model_1.predict(images).flatten())

# Convert probabilities (0.0 to 1.0) to class labels (0 or 1)
y_pred = (np.array(y_pred_probs) > 0.5).astype(int)

# Calculate the confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Model 1 Confusion Matrix")
plt.savefig('model_1_confusion_matrix.png') # Save the matrix
plt.show()

print("Script finished. Check the pop-up plots and saved .png files.")

