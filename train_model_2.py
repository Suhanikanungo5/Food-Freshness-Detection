import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

#PATHUPDTAE***
DATA_DIR = r"C:\Users\Suhani Kanungo\Downloads\archive (5)\Dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32 # BATCHSIZE ***
EPOCHS = 15  #EPOCH***

print(f"Using TensorFlow version: {tf.__version__}")

# Part 2: Load Datasets 

print("Loading training data...")
 #TRAINING_SET 80%***
train_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

print("Loading validation data...")
#VALIDATION_SET 20%***
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

# Get class names
class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# DATA_AUGMENTATION***

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)

# MODEL_2,TRANSFER_LEARNING(MOBILENET V2)***

# This part is NEW and DIFFERENT
print("Building Model 2: Transfer Learning (MobileNetV2)...")


# LOADING PRETRAINED MODEL ***
# include_top=False means we DON'T load its final 1000-class classifier***
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

#FREEZE THE BASE MODEL ***

base_model.trainable = False

#CREATING NEW MODEL ON TOP ***
model_2 = keras.Sequential([
    # Add our augmentation
    data_augmentation,
    
    # SPECIAL PREPROCESSING LAYER(MOBILENET V2 )***
    # This scales pixel values from [0, 255] to [-1, 1]
    layers.Lambda(tf.keras.applications.mobilenet_v2.preprocess_input),
    # ADDING FROZEN BASE MODEL***
    base_model,
    
    # New Classification Head***
    # PART TO BE TRAINED ***
    
    # FLATTENING OUTPUT OF BASE MODEL***
    layers.GlobalAveragePooling2D(),
    
   
    layers.Dropout(0.2), #DROPOUT FOR REGULARIZATION***
    
    # FINAL CLASSIFICATION LAYER***
    layers.Dense(1, activation='sigmoid') 
], name="transfer_learning_mobilenetv2")

# MODEL COMPILATION***
# We use a SMALLER learning rate for fine-tuning

model_2.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

#MODEL SUMMARY ***
print("\n--- Model 2 Summary ---")
model_2.summary()
print("-----------------------\n")


# Part 7: Train the Model

print("Starting Model 2 training...")
history_2 = model_2.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS
)
print("Model 2 training complete.")


# Part 8: Save the Final Model


print("Saving model to file...")
model_2.save("food_freshness_model_v2.keras")
print("Model saved as food_freshness_model_v2.keras")


# Part 9: Evaluate the Model (Same as before)

print("Evaluating model and plotting results...")

# 1. Plot Accuracy and Loss Graphs
acc = history_2.history['accuracy']
val_acc = history_2.history['val_accuracy']
loss = history_2.history['loss']
val_loss = history_2.history['val_loss']
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
plt.suptitle('Model 2 (Transfer Learning) Performance')
plt.savefig('model_2_performance_graph.png')
plt.show()

# 2. Generate Confusion Matrix
y_true = []
y_pred_probs = []

for images, labels in validation_dataset:
    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(model_2.predict(images).flatten())

y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Model 2 Confusion Matrix")
plt.savefig('model_2_confusion_matrix.png')
plt.show()

print("Script finished. Model 2 is saved and plots are generated.")