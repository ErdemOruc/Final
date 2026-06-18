import os 
import tensorflow as tf                                   
import matplotlib.pyplot as plt                           
import matplotlib.image as mpimg                           
from tensorflow.keras.models import Sequential, Model      
from tensorflow.keras.optimizers import Adam               
from tensorflow.keras.callbacks import EarlyStopping               
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization  
from tensorflow.keras.applications import DenseNet121

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "old_damaged_data", "train")
VAL_DIR = os.path.join(BASE_DIR, "old_damaged_data", "val")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "OldDamagedWeight.keras")

train_data = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

train_data = train_data.map(lambda x, y: (x / 255.0, y))

val_data = tf.keras.preprocessing.image_dataset_from_directory(
    VAL_DIR,
    labels='inferred',
    label_mode='categorical',
    image_size=(256, 256),
    batch_size=32)

val_data = val_data.map(lambda x, y: (x / 255.0, y))

conv_base = DenseNet121(
    weights='imagenet',
    include_top = False,
    input_shape=(256,256,3),
    pooling='avg'
)

conv_base.trainable = False

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.RandomZoom(0.2),
])

model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(256, 256, 3)))
model.add(data_augmentation)
model.add(conv_base)
model.add(BatchNormalization())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.35))
model.add(BatchNormalization())
model.add(Dense(120, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

print("\n--- PHASE 1 STARTING (Top Layers) ---")
history = model.fit(
    train_data, 
    epochs=25, 
    validation_data=val_data, 
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)

print("\n--- PHASE 1 COMPLETE ---")

evaluation = model.evaluate(val_data)

print("Validation Loss:", evaluation[0])
print("Validation Accuracy:", evaluation[1])

model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")