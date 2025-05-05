import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import os
import numpy as np
from collections import Counter

# Define parameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 150
NUM_CLASSES = 29  # Updated to match your 29 classes

# Enhanced data augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.15,
    shear_range=0.1,
    brightness_range=[0.8, 1.2],
    horizontal_flip=False,
    validation_split=0.2,
    fill_mode='nearest'
)

# Load data
train_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
print(f"Total training images: {train_generator.samples}")

validation_generator = train_datagen.flow_from_directory(
    'Data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)
print(f"Total validation images: {validation_generator.samples}")
# Correct class weights calculation
def get_class_weights(generator):
    class_counts = Counter(generator.classes)
    total_samples = sum(class_counts.values())
    num_classes = len(class_counts)
    return {i: total_samples / (num_classes * count) for i, count in class_counts.items()}

class_weights = get_class_weights(train_generator)

# Improved model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(2,2),
    BatchNormalization(),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Dropout(0.3),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(29, activation='softmax')
])
# Enhanced model compilation
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)  # Increased from 2e-5
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
)

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, min_delta=0.001),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', mode='max')
]
# Add this before training:
print("Class distribution:")
print(Counter(train_generator.classes))

# Visualize samples
import matplotlib.pyplot as plt
x, y = next(train_generator)
plt.imshow(x[0])
plt.title(f"Class: {y[0]}")
plt.show()

# Train model with class weights
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks,
    class_weight=class_weights
)
# ==================== TEST EVALUATION ==================== 
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    'Test_Data',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False  # Critical for accurate metrics
)

# Detailed evaluation
loss, accuracy, top3_acc = model.evaluate(test_generator)
print(f"\nTest Accuracy: {accuracy:.2%}")
print(f"Top-3 Accuracy: {top3_acc:.2%}")

# Generate confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

test_preds = model.predict(test_generator)
y_true = test_generator.classes
y_pred = np.argmax(test_preds, axis=1)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(15,10))
sns.heatmap(cm, annot=True, fmt='d', 
            xticklabels=test_generator.class_indices.keys(),
            yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.savefig('Model/confusion_matrix.png')  # Save for later analysis

# Save final model
model.save('Model/sign_language_model.h5')

# Save class labels
with open('Model/labels.txt', 'w') as f:
    for class_name, class_idx in train_generator.class_indices.items():
        f.write(f"{class_idx} {class_name}\n")

# Save training history
import pickle
with open('Model/training_history.pkl', 'wb') as f:
    pickle.dump(history.history, f)