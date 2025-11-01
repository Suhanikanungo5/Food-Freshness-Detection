import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


DATA_DIR = r"C:\Users\Suhani Kanungo\Downloads\archive (5)\Dataset"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
INITIAL_EPOCHS = 10  # EPOCH SAME*** STAGE 1 
FINETUNE_EPOCHS = 10 #  STAGE 2 
TOTAL_EPOCHS = INITIAL_EPOCHS + FINETUNE_EPOCHS

print(f"Using TensorFlow version: {tf.__version__}")


# Part 2: Load Datasets

print("Loading training data...")
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
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

class_names = train_dataset.class_names
print(f"Classes found: {class_names}")

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)

#Data Augmentation ***

data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="data_augmentation"
)


# Part 4: Build Model***

print("Building model...")

# LOADING PRERAINED MODEL***
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)

# 2. Freeze the base model (for Stage 1)***
base_model.trainable = False

# 3. Create our new model
inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False) # 'training=False' is important when base is frozen
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)
model_3 = keras.Model(inputs, outputs, name="finetuning_mobilenetv2")

# COMPILE FOR STAGE 1 
model_3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model 3 Summary (Before Fine-Tuning) ---")
model_3.summary()
print("--------------------------------------------\n")


# Part 6: Train Stage 1 (Feature Extraction)

print(f"--- Starting Stage 1 Training ({INITIAL_EPOCHS} epochs) ---")
history_stage_1 = model_3.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=INITIAL_EPOCHS
)
print("--- Stage 1 Training Complete ---")


# NEW - Un-freeze for Fine-Tuning***

print("\n--- Preparing for Stage 2 (Fine-Tuning) ---")
base_model.trainable = True

# We'll only un-freeze the top layers.
# Let's un-freeze from layer 100 onwards (MobileNetV2 has 154)***
fine_tune_at = 100

# Freeze all layers *before* the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

print(f"Un-freezing top {len(base_model.layers) - fine_tune_at} layers of base model.")


# Part 8: NEW - Re-compile for Fine-Tuning***


model_3.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5), # 0.00001
    loss='binary_crossentropy',
    metrics=['accuracy']
)

print("\n--- Model 3 Summary (After Fine-Tuning Setup) ---")
model_3.summary()
print("--------------------------------------------\n")


# Part 9: NEW - Train Stage 2 (Fine-Tuning)

print(f"--- Starting Stage 2 Training ({FINETUNE_EPOCHS} epochs) ---")
# We continue training from where we left off
history_stage_2 = model_3.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=TOTAL_EPOCHS,
    initial_epoch=history_stage_1.epoch[-1] # Start from the last epoch
)
print("--- Stage 2 Fine-Tuning Complete ---")


# Part 10: Save the Final Model

print("Saving fine-tuned model to file...")
model_3.save("food_freshness_model_v3_finetuned.keras")
print("Model saved as food_freshness_model_v3_finetuned.keras")


#  Evaluate the Final Model

print("Evaluating final model and plotting results...")

# Combine the history from both training stages
acc = history_stage_1.history['accuracy'] + history_stage_2.history['accuracy']
val_acc = history_stage_1.history['val_accuracy'] + history_stage_2.history['val_accuracy']
loss = history_stage_1.history['loss'] + history_stage_2.history['loss']
val_loss = history_stage_1.history['val_loss'] + history_stage_2.history['val_loss']

# Plot the combined graphs
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.axvline(INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='lower right')
plt.title('Combined Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.axvline(INITIAL_EPOCHS - 1, color='gray', linestyle='--', label='Start Fine-Tuning')
plt.legend(loc='upper right')
plt.title('Combined Training and Validation Loss')
plt.suptitle('Model 3 (Fine-Tuning) Performance')
plt.savefig('model_3_performance_graph.png')
plt.show()

# Generate Confusion Matrix
y_true = []
y_pred_probs = []
for images, labels in validation_dataset:
    y_true.extend(labels.numpy().flatten())
    y_pred_probs.extend(model_3.predict(images).flatten())

y_pred = (np.array(y_pred_probs) > 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues)
plt.title("Model 3 (Fine-Tuned) Confusion Matrix")
plt.savefig('model_3_confusion_matrix.png')
plt.show()

print("Script finished. Model 3 is saved and plots are generated.")
